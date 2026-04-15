"""TRL Evaluator Agent — NASA TRL 9단계 기준 기술 성숙도 판정 전담"""
import json
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from models.state import AgentState
from prompts.trl_evaluator_prompt import TRL_EVALUATION_PROMPT
from config import OPENAI_API_KEY, LLM_MODEL

# 기대 판정 수 (3기술 × 3기업)
EXPECTED_TRL_COUNT = 9

# TRL Evaluator에 전달할 최대 자료 수 (재시도 누적 방지)
MAX_WEB_RESULTS_FOR_TRL = 60


def _init_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.1,   # TRL 판정은 일관성이 중요 → 낮은 temperature
        api_key=OPENAI_API_KEY,
        max_tokens=3000,
    )


def _deduplicate_web_results(web_results: List[dict]) -> List[dict]:
    """URL 기준 중복 제거 후 신뢰도 내림차순 정렬, 최대 MAX_WEB_RESULTS_FOR_TRL 반환"""
    seen_urls: set = set()
    unique: List[dict] = []
    for r in web_results:
        url = r.get("source", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        unique.append(r)

    # 신뢰도 높은 순으로 정렬 후 상한 적용
    unique.sort(key=lambda x: x.get("reliability", 0.5), reverse=True)
    return unique[:MAX_WEB_RESULTS_FOR_TRL]


def _format_web_results_from_list(results: List[dict]) -> str:
    """이미 정제된 결과 목록을 LLM 입력용 텍스트로 포맷팅"""
    if not results:
        return "수집된 자료 없음"
    formatted = []
    for i, r in enumerate(results, 1):
        reliability = r.get("reliability", 0.5)
        category = r.get("category", "news")
        formatted.append(
            f"[{i}] ({category}, 신뢰도: {reliability})\n"
            f"제목: {r.get('title', 'N/A')}\n"
            f"출처: {r.get('source', 'N/A')}\n"
            f"내용: {r.get('content', 'N/A')[:500]}\n"
        )
    return "\n---\n".join(formatted)


def _parse_trl_json(content: str) -> List[dict]:
    """LLM 응답에서 TRL JSON 파싱 (코드블록 자동 제거)"""
    content = content.strip()

    # 코드블록(```json ... ``` 또는 ``` ... ```) 제거
    if content.startswith("```"):
        lines = content.splitlines()
        # 첫 줄(```json / ```)과 마지막 줄(```) 제거
        inner_lines = lines[1:]
        if inner_lines and inner_lines[-1].strip() == "```":
            inner_lines = inner_lines[:-1]
        content = "\n".join(inner_lines)

    try:
        result = json.loads(content.strip())
        if isinstance(result, list):
            return result
        # 딕셔너리 단건 반환 방어
        if isinstance(result, dict):
            return [result]
    except json.JSONDecodeError as e:
        print(f"   ⚠ TRL JSON 파싱 실패: {e}")
        print(f"   응답 앞부분: {content[:300]}")

    return []


def _backfill_missing_fields(assessments: List[dict]) -> List[dict]:
    """LLM이 누락한 필수 필드(is_estimated, estimation_basis, counter_evidence)를 자동 보정.
    Supervisor가 해당 필드 누락으로 NOT_OK를 반복 발동하는 것을 방지한다.
    """
    for item in assessments:
        trl = item.get("trl_level", 0)
        # is_estimated 누락 → TRL 4~6이면 True, 아니면 False
        if "is_estimated" not in item:
            item["is_estimated"] = bool(4 <= trl <= 6)
        # is_estimated=True인데 estimation_basis 누락/null/빈 문자열 → 기본값 삽입
        if item.get("is_estimated") and not item.get("estimation_basis"):
            item["estimation_basis"] = "간접 지표 기반 추정 (상세 근거 미제공)"
        # 직접 판정이면 estimation_basis는 null
        if not item.get("is_estimated"):
            item["estimation_basis"] = item.get("estimation_basis") or None
        # counter_evidence 누락/null/빈 문자열 → 기본값 삽입
        if not item.get("counter_evidence"):
            item["counter_evidence"] = "반박 근거 미제공"
        # evidence 누락 → 빈 리스트
        if "evidence" not in item or not isinstance(item.get("evidence"), list):
            item["evidence"] = []
    return assessments


def _validate_trl_assessments(assessments: List[dict]) -> List[str]:
    """TRL 판정 결과 유효성 검사 — 경고 메시지 반환"""
    warnings = []

    if len(assessments) < EXPECTED_TRL_COUNT:
        warnings.append(
            f"⚠ 판정 수 부족: {len(assessments)}개 (기대: {EXPECTED_TRL_COUNT}개)"
        )

    for t in assessments:
        trl = t.get("trl_level", 0)
        # TRL 4~6 간접 추정인데 is_estimated가 False인 경우
        if 4 <= trl <= 6 and not t.get("is_estimated", False):
            warnings.append(
                f"⚠ {t.get('company', '?')}/{t.get('technology', '?')}: "
                f"TRL {trl}은 간접 추정 범위이나 is_estimated=False"
            )
        # is_estimated=True인데 estimation_basis가 없는 경우
        if t.get("is_estimated") and not t.get("estimation_basis"):
            warnings.append(
                f"⚠ {t.get('company', '?')}/{t.get('technology', '?')}: "
                f"is_estimated=True이나 estimation_basis 누락"
            )
        # counter_evidence 누락
        if not t.get("counter_evidence"):
            warnings.append(
                f"⚠ {t.get('company', '?')}/{t.get('technology', '?')}: "
                f"counter_evidence 누락"
            )

    return warnings


def trl_evaluator_agent_node(state: AgentState) -> dict:
    """TRL Evaluator Agent 메인 노드"""
    llm = _init_llm()

    print("\n🔬 [TRL Evaluator Agent] TRL 판정 시작...")

    web_results = state.get("web_results", [])
    deduped = _deduplicate_web_results(web_results)
    print(f"   입력 자료: {len(web_results)}건 원본 → {len(deduped)}건 (중복 제거 + 상한 {MAX_WEB_RESULTS_FOR_TRL}건)")
    # 이미 dedup된 결과를 직접 포맷 (내부에서 중복 제거 재실행 방지)
    formatted_results = _format_web_results_from_list(deduped)

    prompt = TRL_EVALUATION_PROMPT.format(
        target_technologies=", ".join(state["target_technologies"]),
        web_results=formatted_results,
        supervisor_feedback=state.get("supervisor_feedback", "없음"),
    )

    # LLM 호출
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content
    except Exception as e:
        print(f"   ❌ LLM 호출 실패: {e}")
        return {
            "trl_assessments": [],
            "messages": [f"[TRL Evaluator Agent] LLM 호출 실패: {e}"],
        }

    # JSON 파싱
    trl_assessments = _parse_trl_json(content)

    # 누락 필드 자동 보정 (is_estimated, estimation_basis, counter_evidence)
    before_count = sum(
        1 for t in trl_assessments
        if ("is_estimated" not in t) or (not t.get("counter_evidence"))
        or (t.get("is_estimated") and not t.get("estimation_basis"))
    )
    trl_assessments = _backfill_missing_fields(trl_assessments)
    if before_count > 0:
        print(f"   🔧 필수 필드 자동 보정: {before_count}개 항목")

    # 유효성 검사
    validation_warnings = _validate_trl_assessments(trl_assessments)
    for w in validation_warnings:
        print(f"   {w}")

    # 판정 결과 요약 출력
    completeness = "✅" if len(trl_assessments) == EXPECTED_TRL_COUNT else f"⚠ {len(trl_assessments)}/{EXPECTED_TRL_COUNT}"
    print(f"   TRL 판정 완료: {completeness}")

    for trl in trl_assessments:
        estimated_flag = " ⚠간접추정" if trl.get("is_estimated") else ""
        print(
            f"   - {trl.get('company', '?'):8s} / {trl.get('technology', '?'):5s}: "
            f"TRL {trl.get('trl_level', '?')} "
            f"(신뢰도 {trl.get('confidence', '?')}){estimated_flag}"
        )

    return {
        "trl_assessments": trl_assessments,
        "messages": [
            f"[TRL Evaluator Agent] TRL 판정 {len(trl_assessments)}건 완료 "
            f"({'완결' if len(trl_assessments) == EXPECTED_TRL_COUNT else '불완전'})"
        ],
    }
