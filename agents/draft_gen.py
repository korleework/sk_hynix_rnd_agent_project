"""Draft Generation Agent — TRL 판정 결과 기반 보고서 초안 작성"""
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from models.state import AgentState
from prompts.draft_gen_prompt import DRAFT_GENERATION_PROMPT
from config import OPENAI_API_KEY, LLM_MODEL


def _init_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.3,   # 분석 정확도와 창의성 사이 균형
        api_key=OPENAI_API_KEY,
        max_tokens=4096,
    )


def _format_web_results(web_results: List[dict]) -> str:
    """웹 검색 결과를 LLM 입력용 텍스트로 포맷팅"""
    if not web_results:
        return "수집된 자료 없음"
    formatted = []
    for i, r in enumerate(web_results, 1):
        reliability = r.get("reliability", 0.5)
        category = r.get("category", "news")
        formatted.append(
            f"[{i}] ({category}, 신뢰도: {reliability})\n"
            f"제목: {r.get('title', 'N/A')}\n"
            f"출처: {r.get('source', 'N/A')}\n"
            f"내용: {r.get('content', 'N/A')[:500]}\n"
        )
    return "\n---\n".join(formatted)


def _format_trl_assessments(trl_assessments: List[dict]) -> str:
    """TRL 판정 결과를 LLM 입력용 텍스트로 포맷팅"""
    if not trl_assessments:
        return "TRL 판정 결과 없음 — TRL Evaluator Agent 오류"
    lines = []
    for t in trl_assessments:
        estimated_flag = " [간접추정]" if t.get("is_estimated") else ""
        basis = f" | 추정근거: {t.get('estimation_basis', '-')}" if t.get("is_estimated") else ""
        evidence = ", ".join(t.get("evidence", [])) or "-"
        counter = t.get("counter_evidence") or "-"
        lines.append(
            f"- {t.get('company', '?')} / {t.get('technology', '?')}: "
            f"TRL {t.get('trl_level', '?')} (신뢰도 {t.get('confidence', '?')}){estimated_flag}{basis}\n"
            f"  근거: {evidence}\n"
            f"  반박: {counter}"
        )
    return "\n".join(lines)


def draft_generation_node(state: AgentState) -> dict:
    """Draft Generation Agent 메인 노드"""
    llm = _init_llm()

    print("\n📝 [Draft Generation Agent] 보고서 초안 작성 시작...")

    web_results = state.get("web_results", [])
    trl_assessments = state.get("trl_assessments", [])

    formatted_results = _format_web_results(web_results)
    formatted_trl = _format_trl_assessments(trl_assessments)

    print(f"   입력 자료: {len(web_results)}건의 수집 자료, TRL 판정 {len(trl_assessments)}건")

    prompt = DRAFT_GENERATION_PROMPT.format(
        target_technologies=", ".join(state["target_technologies"]),
        trl_assessments=formatted_trl,
        web_results=formatted_results,
        supervisor_feedback=state.get("supervisor_feedback", "없음"),
    )

    # LLM 호출
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        draft_report = response.content.strip()
    except Exception as e:
        print(f"   ❌ LLM 호출 실패: {e}")
        return {
            "draft_report": f"[오류] LLM 호출 실패: {e}",
            "messages": [f"[Draft Generation Agent] LLM 호출 실패: {e}"],
        }

    print(f"   보고서 초안 길이: {len(draft_report)}자")

    return {
        "draft_report": draft_report,
        "messages": [
            f"[Draft Generation Agent] 보고서 초안 {len(draft_report)}자 작성 완료"
        ],
    }
