"""Web Search Agent — 웹 검색 + 벡터 DB 캐싱 + Hybrid 검색"""
import json
import os
import time
from datetime import datetime, timedelta
from typing import List, Dict
from collections import Counter

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

from models.state import AgentState, WebSearchResult
from prompts.web_search_prompt import (
    WEB_SEARCH_QUERY_GENERATION_PROMPT,            # 기존 단일 라운드 (폴백용)
    WEB_SEARCH_QUERY_GENERATION_GENERAL_PROMPT,    # 라운드 1: 일반
    WEB_SEARCH_QUERY_GENERATION_ACADEMIC_PROMPT,   # 라운드 2: 학술/특허
    WEB_SEARCH_QUERY_GENERATION_JOBS_PROMPT,       # 라운드 3: 채용
)
from config import (
    OPENAI_API_KEY, TAVILY_API_KEY, LLM_MODEL, LLM_TEMPERATURE,
    MAX_SEARCH_RESULTS, MAX_QUERIES, SEARCH_DEPTH, TIME_RANGE, SEARCH_DELAY,
    TARGET_COMPANIES, RELIABILITY_SCORES, SEARCH_ROUND_CONFIG,
)

# Rate limit 에러 키워드
_RATE_LIMIT_KEYWORDS = ["excessive requests", "rate limit", "too many requests", "blocked"]


def _init_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=OPENAI_API_KEY,
    )


def _init_tavily():
    return TavilyClient(api_key=TAVILY_API_KEY)


def _generate_search_queries(state: AgentState, llm) -> List[str]:
    """LLM을 사용하여 검색 쿼리 생성.
    재수집 시에는 이전 쿼리를 회피하도록 프롬프트에 previous_queries 전달.
    """
    previous_queries = state.get("search_queries_used", [])
    prev_queries_str = ", ".join(previous_queries) if previous_queries else "없음"

    prompt = WEB_SEARCH_QUERY_GENERATION_PROMPT.format(
        target_technologies=", ".join(state["target_technologies"]),
        supervisor_feedback=state.get("supervisor_feedback", "없음"),
        previous_queries=prev_queries_str,
    )
    response = llm.invoke([HumanMessage(content=prompt)])

    # JSON 배열 파싱
    queries: List[str] = []
    try:
        content = response.content
        # JSON 블록 추출
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        parsed = json.loads(content.strip())
        if isinstance(parsed, list):
            queries = parsed
    except (json.JSONDecodeError, IndexError):
        pass

    # 파싱 실패 시 기본 쿼리 생성
    if not queries:
        for tech in state["target_technologies"]:
            for company in ["SK hynix", "Samsung", "Micron"]:
                queries.extend([
                    f"{company} {tech} development status 2025",
                    f"{company} {tech} mass production roadmap",
                    f"{company} {tech} patent filing 2024 2025",
                ])

    # 이전 쿼리와 중복되는 쿼리 제거 (재수집 시 동일 결과 방지)
    previous_set = set(q.strip().lower() for q in previous_queries)
    deduped = [q for q in queries if q.strip().lower() not in previous_set]
    removed = len(queries) - len(deduped)
    if removed > 0:
        print(f"   이전 쿼리와 중복된 {removed}개 제거 (재수집 다양성 확보)")

    # 소스 유형별 보강 쿼리 — supervisor_feedback에 특정 키워드가 있으면 자동 추가
    feedback = state.get("supervisor_feedback", "").lower()
    augmented: List[str] = []
    techs = state.get("target_technologies", [])

    if "특허" in feedback or "patent" in feedback:
        for tech in techs:
            augmented.extend([
                f"{tech} patent application 2024 2025",
                f"{tech} 특허 출원",
                f"{tech} USPTO patent filing",
            ])
        print(f"   📎 '특허' 키워드 감지 → 특허 전용 쿼리 {len(augmented)}개 추가")

    if "논문" in feedback or "paper" in feedback or "학술" in feedback:
        paper_queries = []
        for tech in techs:
            paper_queries.extend([
                f"{tech} IEEE paper 2025",
                f"{tech} ISSCC 2025",
                f"{tech} arxiv semiconductor research",
            ])
        augmented.extend(paper_queries)
        print(f"   📎 '논문' 키워드 감지 → 학술 전용 쿼리 {len(paper_queries)}개 추가")

    # 보강 쿼리도 이전 쿼리와 중복 제거
    augmented = [q for q in augmented if q.strip().lower() not in previous_set]
    deduped.extend(augmented)

    return deduped


def _parse_query_json(content: str) -> List[str]:
    """LLM 응답에서 JSON 배열 쿼리 파싱"""
    try:
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        parsed = json.loads(content.strip())
        if isinstance(parsed, list):
            return [str(q) for q in parsed if isinstance(q, (str, int))]
    except (json.JSONDecodeError, IndexError):
        pass
    return []


def _generate_queries_for_round(
    round_name: str,
    state: AgentState,
    llm,
    previous_queries: List[str],
) -> List[str]:
    """라운드 이름에 맞춘 특화 쿼리 생성"""
    techs = state.get("target_technologies", [])
    prev_str = ", ".join(previous_queries) if previous_queries else "없음"
    feedback = state.get("supervisor_feedback", "없음")

    if round_name == "general":
        prompt = WEB_SEARCH_QUERY_GENERATION_GENERAL_PROMPT.format(
            target_technologies=", ".join(techs),
            supervisor_feedback=feedback,
            previous_queries=prev_str,
        )
        fallback = [
            f"{c} {t} mass production roadmap 2025"
            for t in techs for c in ["SK hynix", "Samsung", "Micron"]
        ]
    elif round_name == "academic_patent":
        prompt = WEB_SEARCH_QUERY_GENERATION_ACADEMIC_PROMPT.format(
            target_technologies=", ".join(techs),
            previous_queries=prev_str,
        )
        fallback = []
        for t in techs:
            fallback.append(f"{t} architecture paper 2025")
            fallback.append(f"{t} patent application filing")
    elif round_name == "jobs":
        prompt = WEB_SEARCH_QUERY_GENERATION_JOBS_PROMPT.format(
            target_technologies=", ".join(techs),
            previous_queries=prev_str,
        )
        fallback = [
            f"SK hynix {techs[0] if techs else 'HBM4'} engineer hiring 2025",
            f"Samsung {techs[1] if len(techs) > 1 else 'PIM'} design engineer job",
            f"Micron {techs[2] if len(techs) > 2 else 'CXL'} controller engineer",
        ]
    else:
        return []

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        queries = _parse_query_json(response.content)
    except Exception as e:
        print(f"   ⚠ '{round_name}' 라운드 쿼리 생성 실패: {e}")
        queries = []

    if not queries:
        queries = fallback

    # 이전 쿼리 중복 제거
    previous_set = set(q.strip().lower() for q in previous_queries)
    return [q for q in queries if q.strip().lower() not in previous_set]


def _classify_source(url: str, title: str, content: str) -> str:
    """URL과 내용 기반 소스 유형 분류 (도메인 매핑 강화)"""
    url_lower = url.lower()
    title_lower = title.lower()
    combined = f"{url_lower} {title_lower}"

    # 1. 논문 도메인
    if any(d in url_lower for d in [
        "arxiv.org", "ieee.org", "ieeexplore.ieee.org", "nature.com",
        "sciencedirect.com", "scholar.google.com", "semanticscholar.org",
        "acm.org", "springer.com",
    ]):
        return "paper"
    # 2. 특허 도메인
    if any(d in url_lower for d in [
        "patents.google.com", "uspto.gov", "patentscope.wipo.int",
        "kipris.or.kr", "freepatentsonline.com",
    ]):
        return "patent"
    if "patent" in title_lower and "application" in title_lower:
        return "patent"
    # 3. 채용 공고 도메인
    if any(d in url_lower for d in [
        "linkedin.com/jobs", "indeed.com", "greenhouse.io", "lever.co",
        "careers.skhynix.com", "careers.samsung.com", "careers.micron.com",
        "glassdoor.com",
    ]):
        return "job_posting"
    if any(k in title_lower for k in ["hiring", "채용", "engineer job", "engineer - "]):
        return "job_posting"
    # 4. 어닝콜/IR
    if any(d in url_lower for d in [
        "seekingalpha.com", "investor.skhynix.com", "ir.samsung.com",
        "investors.micron.com",
    ]):
        return "earnings_call"
    if any(k in title_lower for k in ["earnings call", "transcript", "어닝콜", "investor presentation"]):
        return "earnings_call"
    # 5. 학회
    if any(d in combined for d in ["isscc", "hotchips", "iedm", "vlsi", "dac.com", "conference"]):
        return "conference"
    # 6. 공식 발표
    if any(d in url_lower for d in [
        "news.skhynix.com", "news.samsung.com", "micron.com/about",
        "skhynix.com/press", "samsung.com/newsroom",
    ]):
        return "official"
    if any(d in url_lower for d in ["skhynix.com", "samsung.com/semiconductor", "micron.com"]):
        return "official"
    return "news"


def _calculate_bias_check(results: List[dict]) -> List[str]:
    """편향 방지 점검 로그 생성"""
    logs = []

    # 1. 소스 유형 분포
    category_counts = Counter(r.get("category", "news") for r in results)
    logs.append(f"[소스 유형 분포] {dict(category_counts)}")
    if len(category_counts) >= 3:
        logs.append("[소스 다양성] PASS — 3종 이상 소스 유형 확보")
    else:
        logs.append(f"[소스 다양성] FAIL — {len(category_counts)}종만 확보, 3종 이상 필요")

    # 2. 기업별 정보량
    company_counts = {"SK하이닉스": 0, "삼성전자": 0, "마이크론": 0}
    for r in results:
        content = f"{r.get('title', '')} {r.get('content', '')}".lower()
        if any(k in content for k in ["sk hynix", "sk하이닉스", "skhynix"]):
            company_counts["SK하이닉스"] += 1
        if any(k in content for k in ["samsung", "삼성"]):
            company_counts["삼성전자"] += 1
        if any(k in content for k in ["micron", "마이크론"]):
            company_counts["마이크론"] += 1

    logs.append(f"[기업별 정보량] {company_counts}")
    values = [v for v in company_counts.values() if v > 0]
    if values and max(values) <= min(values) * 2:
        logs.append("[기업 대칭성] PASS — 최다/최소 비율 2:1 이내")
    else:
        logs.append("[기업 대칭성] FAIL — 정보량 불균형 감지")

    # 3. 총 수집량
    logs.append(f"[총 수집량] {len(results)}건")

    return logs


def _run_search_round(
    round_cfg: dict,
    queries: List[str],
    tavily,
    seen_urls: set,
) -> List[dict]:
    """단일 라운드의 Tavily 검색 실행.
    round_cfg의 include_domains/time_range를 적용하여 검색하고 신규 결과만 반환.
    """
    round_name = round_cfg["name"]
    include_domains = round_cfg.get("include_domains")
    time_range = round_cfg.get("time_range", TIME_RANGE)

    print(f"\n   ── 라운드 [{round_name}] 시작 ({len(queries)}개 쿼리" +
          (f", domains={len(include_domains)}개" if include_domains else ", 도메인 제한 없음") + ")")

    round_results: List[dict] = []
    consecutive_rate_limit = 0
    MAX_CONSECUTIVE_RATE_LIMIT = 3

    for i, query in enumerate(queries, 1):
        if i > 1:
            time.sleep(SEARCH_DELAY)

        # Tavily search 파라미터 (include_domains는 None일 때 전달 안 함)
        search_kwargs = {
            "query": query,
            "search_depth": SEARCH_DEPTH,
            "max_results": 5,
            "time_range": time_range,
        }
        if include_domains:
            search_kwargs["include_domains"] = include_domains

        try:
            response = tavily.search(**search_kwargs)
            consecutive_rate_limit = 0

            new_in_query = 0
            for result in response.get("results", []):
                url = result.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                title = result.get("title", "")
                content = result.get("content", "")
                category = _classify_source(url, title, content)

                web_result = {
                    "source": url,
                    "title": title,
                    "content": content,
                    "date": "",
                    "reliability": RELIABILITY_SCORES.get(category, 0.5),
                    "category": category,
                }
                round_results.append(web_result)
                new_in_query += 1

        except Exception as e:
            error_msg = str(e)
            is_rate_limit = any(kw in error_msg.lower() for kw in _RATE_LIMIT_KEYWORDS)
            if is_rate_limit:
                consecutive_rate_limit += 1
                print(f"      ⚠ Rate limit ({consecutive_rate_limit}/{MAX_CONSECUTIVE_RATE_LIMIT}): {query[:40]}...")
                if consecutive_rate_limit >= MAX_CONSECUTIVE_RATE_LIMIT:
                    print(f"      ❌ 라운드 [{round_name}] 조기 종료 (rate limit)")
                    break
                time.sleep(SEARCH_DELAY * 4)
            else:
                print(f"      ⚠ 검색 실패: {query[:50]}... — {error_msg[:60]}")
            continue

    print(f"   라운드 [{round_name}] 완료: {len(round_results)}건 수집")
    return round_results


def web_search_agent_node(state: AgentState) -> dict:
    """Web Search Agent 메인 노드 — 다중 라운드 검색 (include_domains 기반 소스 다양화)"""
    llm = _init_llm()
    tavily = _init_tavily()

    print("\n🔍 [Web Search Agent] 다중 라운드 검색 시작...")

    # 재수집인지 확인 — 기존 수집 URL 집합 추출
    existing_results = state.get("web_results", []) or []
    existing_urls = {r.get("source", "") for r in existing_results if r.get("source")}
    previous_queries = state.get("search_queries_used", []) or []
    is_retry = len(existing_results) > 0
    if is_retry:
        print(f"   🔁 재수집 모드: 기존 {len(existing_results)}건 / {len(existing_urls)} URL 중복 차단")

    # ── 라운드별 쿼리 생성 + 실행 ──
    all_results: List[dict] = []
    all_queries_used: List[str] = []
    seen_urls = set(existing_urls)

    for round_cfg in SEARCH_ROUND_CONFIG:
        round_name = round_cfg["name"]
        round_queries = _generate_queries_for_round(round_name, state, llm, previous_queries + all_queries_used)

        # 라운드별 쿼리 수 상한 적용
        cap = round_cfg.get("queries_per_round", 9)
        if len(round_queries) > cap:
            round_queries = round_queries[:cap]

        if not round_queries:
            print(f"   ⏭ 라운드 [{round_name}] 건너뜀 (쿼리 없음)")
            continue

        round_results = _run_search_round(round_cfg, round_queries, tavily, seen_urls)
        all_results.extend(round_results)
        all_queries_used.extend(round_queries)

        # 전체 쿼리 상한 체크 — MAX_QUERIES 초과 시 이후 라운드 건너뜀
        if len(all_queries_used) >= MAX_QUERIES:
            print(f"   ⚠ 전체 쿼리 상한 {MAX_QUERIES} 도달 → 이후 라운드 건너뜀")
            break

    # ── 결과 요약 ──
    if is_retry:
        print(f"\n   📊 수집 완료: 신규 {len(all_results)}건 (기존 {len(existing_results)}건 + 신규 = 총 {len(existing_results) + len(all_results)}건)")
    else:
        print(f"\n   📊 수집 완료: {len(all_results)}건 (총 {len(all_queries_used)}개 쿼리 × {len(SEARCH_ROUND_CONFIG)}라운드)")

    if not all_results:
        print("   ⚠ 수집 결과가 0건입니다.")
        print("   💡 확인 사항:")
        print("      1. Tavily API 키가 유효한 Production 키인지 확인 (tvly-... 형식)")
        print("      2. https://app.tavily.com 에서 사용량 한도 확인")
        print(f"      3. config.py의 SEARCH_DEPTH가 '{SEARCH_DEPTH}'인지 확인")

    # 편향 방지 점검 (재수집 시 누적 기준)
    combined_for_bias = existing_results + all_results
    bias_log = _calculate_bias_check(combined_for_bias)
    for log in bias_log:
        print(f"   {log}")

    return {
        "web_results": all_results,
        "search_queries_used": all_queries_used,
        "bias_check_log": bias_log,
        "vector_db_status": f"cached: {len(all_results)}건 수집 완료 ({len(SEARCH_ROUND_CONFIG)}라운드)",
        "messages": [
            f"[Web Search Agent] {len(all_results)}건 수집, "
            f"{len(all_queries_used)}개 쿼리 × {len(SEARCH_ROUND_CONFIG)}라운드"
        ],
    }
