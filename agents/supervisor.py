"""Supervisor — 업무 배분, 품질 게이트, 최종 승인"""
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from models.state import AgentState
from prompts.supervisor_prompt import (
    SUPERVISOR_REVIEW_SEARCH_PROMPT,
    SUPERVISOR_REVIEW_SEARCH_RELAXED_PROMPT,
    SUPERVISOR_REVIEW_TRL_PROMPT,
    SUPERVISOR_REVIEW_DRAFT_PROMPT,
    SUPERVISOR_FINAL_CHECK_PROMPT,
)
from config import OPENAI_API_KEY, LLM_MODEL, MAX_ITERATIONS

# 기대 TRL 판정 수
EXPECTED_TRL_COUNT = 9


def _init_llm():
    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.1,   # 판정은 일관성이 중요
        api_key=OPENAI_API_KEY,
    )


# ──────────────────────────────────────────────
# 검증 함수들
# ──────────────────────────────────────────────

def _review_search_results(state: AgentState, llm, relaxed: bool = False) -> str:
    """수집 결과 검증.
    relaxed=True이면 완화된 기준(RELAXED_PROMPT)으로 판정 — 2회차 이후 자동 발동.
    """
    web_results = state.get("web_results", [])

    summary_lines = []
    for i, r in enumerate(web_results[:50], 1):
        summary_lines.append(
            f"[{i}] ({r.get('category', 'news')}) {r.get('title', 'N/A')[:80]}"
        )

    template = SUPERVISOR_REVIEW_SEARCH_RELAXED_PROMPT if relaxed else SUPERVISOR_REVIEW_SEARCH_PROMPT
    prompt = template.format(
        bias_check_log="\n".join(state.get("bias_check_log", [])),
        total_results=len(web_results),
        search_queries=", ".join(state.get("search_queries_used", [])[:10]),
        web_results_summary="\n".join(summary_lines),
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def _review_trl(state: AgentState, llm) -> str:
    """TRL 판정 결과 검증 — 결정적 검증 우선(deterministic-first).

    프로그램 체크가 모두 PASS면 LLM 호출 없이 자동 OK를 반환.
    프로그램 체크에서 경고가 나온 경우에만 LLM이 최종 판정.
    이는 LLM-as-judge의 환각(hallucination) 거부를 차단한다.
    """
    trl_assessments = state.get("trl_assessments", [])

    # ── 1단계: 결정적(프로그래밍) 검증 ──
    trl_lines = []
    validation_warnings = []

    for t in trl_assessments:
        estimated = " ⚠간접추정" if t.get("is_estimated") else ""
        basis = f" | 추정근거: {t.get('estimation_basis') or '없음'}" if t.get("is_estimated") else ""
        # counter_evidence를 요약에 명시 — LLM 환각 방지
        counter = t.get("counter_evidence") or "없음"
        counter_str = f" | 반박근거: {counter[:80]}"
        trl_lines.append(
            f"- {t.get('company', '?')} / {t.get('technology', '?')}: "
            f"TRL {t.get('trl_level', '?')} (신뢰도 {t.get('confidence', '?')}){estimated}{basis}{counter_str}"
        )
        # 프로그래밍 경고 수집
        trl = t.get("trl_level", 0)
        if 4 <= trl <= 6 and not t.get("is_estimated"):
            validation_warnings.append(
                f"⚠ {t.get('company', '?')}/{t.get('technology', '?')}: TRL {trl}이나 is_estimated=False"
            )
        if t.get("is_estimated") and not t.get("estimation_basis"):
            validation_warnings.append(
                f"⚠ {t.get('company', '?')}/{t.get('technology', '?')}: estimation_basis 누락"
            )
        if not t.get("counter_evidence"):
            validation_warnings.append(
                f"⚠ {t.get('company', '?')}/{t.get('technology', '?')}: counter_evidence 누락"
            )

    # ── 2단계: 결정적 게이트 — 9건 완결 + 경고 0건이면 LLM 호출 생략 ──
    if len(trl_assessments) == EXPECTED_TRL_COUNT and not validation_warnings:
        print("   🟢 결정적 검증 통과 (9건 완결 + 필수 필드 모두 채움) → LLM 재검증 생략")
        return "OK (deterministic)"

    # ── 3단계: 경고 또는 불완결 시에만 LLM 재검증 ──
    prompt = SUPERVISOR_REVIEW_TRL_PROMPT.format(
        trl_count=len(trl_assessments),
        trl_assessments="\n".join(trl_lines) if trl_lines else "판정 결과 없음",
        validation_warnings="\n".join(validation_warnings) if validation_warnings else "경고 없음",
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def _review_draft(state: AgentState, llm) -> str:
    """초안 검증"""
    draft = state.get("draft_report", "")
    trl = state.get("trl_assessments", [])

    trl_summary = ""
    for t in trl:
        estimated = " ⚠간접추정" if t.get("is_estimated") else ""
        trl_summary += (
            f"- {t.get('company', '?')} / {t.get('technology', '?')}: "
            f"TRL {t.get('trl_level', '?')} (신뢰도 {t.get('confidence', '?')}){estimated}\n"
        )

    prompt = SUPERVISOR_REVIEW_DRAFT_PROMPT.format(
        draft_report=draft[:3000],
        trl_assessments=trl_summary or "없음",
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


def _review_formatting(state: AgentState, llm) -> str:
    """최종 규격 확인 — 결정적 검증 우선(deterministic-first).

    format_check 문자열이 "PASS"로 시작하면 LLM 호출 없이 자동 OK 반환.
    (LLM-as-judge의 환각 거부를 차단하기 위함)
    """
    format_check = state.get("format_check", "미확인")
    final_report = state.get("final_report", "") or ""

    # ── 결정적 게이트 ──
    # format_check가 "PASS"로 시작하고 최종 보고서가 존재하면 자동 OK
    fc_stripped = format_check.lstrip()
    if fc_stripped.upper().startswith("PASS") and len(final_report) > 500:
        print("   🟢 결정적 규격 검증 통과 (format_check=PASS) → LLM 재검증 생략")
        return "OK (deterministic)"

    # ── 경고/실패 시에만 LLM 재검증 ──
    prompt = SUPERVISOR_FINAL_CHECK_PROMPT.format(
        final_report=final_report[:3000],
        format_check=format_check,
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ──────────────────────────────────────────────
# 메인 노드
# ──────────────────────────────────────────────

def supervisor_node(state: AgentState) -> dict:
    """Supervisor 메인 노드 — 현재 phase에 따라 검증 수행 및 다음 phase 결정"""
    llm = _init_llm()
    phase = state.get("current_phase", "web_search")
    iteration = state.get("iteration_count", 0)

    print(f"\n👔 [Supervisor] 현재 단계: {phase} (반복 {iteration}/{MAX_ITERATIONS})")

    # ── web_search Phase ──
    if phase == "web_search":
        if not state.get("web_results"):
            print("   → Web Search Agent에 수집 지시")
            return {
                "current_phase": "web_search",
                "supervisor_feedback": "첫 수집 시작",
                "messages": ["[Supervisor] Web Search Agent에 수집 지시"],
            }

        # 2회차 이후에는 완화된 기준으로 자동 전환
        use_relaxed = iteration >= 2
        if use_relaxed:
            print(f"   ⚙ 반복 {iteration}회차 — 완화된 검증 기준 적용")
        feedback = _review_search_results(state, llm, relaxed=use_relaxed)
        print(f"   검증 결과: {feedback[:100]}...")

        if feedback.startswith("OK"):
            print("   ✅ 수집 검증 통과 → TRL Evaluator 단계로 전환")
            return {
                "current_phase": "trl_eval",
                "supervisor_feedback": "OK",
                "iteration_count": 0,   # 페이즈 전환 시 카운터 리셋
                "messages": ["[Supervisor] 수집 검증 OK → TRL Evaluator Agent 단계로 전환"],
            }
        else:
            new_iteration = iteration + 1
            # Safety Valve 직전에 완화 기준으로 재검증 (아직 완화 검증이 수행되지 않은 경우만)
            if new_iteration >= MAX_ITERATIONS:
                if not use_relaxed:
                    print("   ⚠ Safety Valve 직전 — 완화 기준 재검증 시도")
                    relaxed_feedback = _review_search_results(state, llm, relaxed=True)
                    print(f"   완화 재검증 결과: {relaxed_feedback[:100]}...")
                    if relaxed_feedback.startswith("OK"):
                        print("   ✅ 완화 기준 통과 → TRL 단계로 정상 전환")
                        return {
                            "current_phase": "trl_eval",
                            "supervisor_feedback": "OK (완화 기준)",
                            "iteration_count": 0,
                            "messages": ["[Supervisor] 완화 기준 OK → TRL Evaluator 단계로 전환"],
                        }
                print(f"   ⚠ 최대 반복 도달 ({MAX_ITERATIONS}회) → 강제 TRL 단계 진행")
                return {
                    "current_phase": "trl_eval",
                    "supervisor_feedback": "Safety Valve: 최대 반복 초과, 현재 결과로 진행",
                    "iteration_count": 0,   # 페이즈 전환 시 카운터 리셋
                    "messages": ["[Supervisor] Safety Valve 발동 — 강제 TRL Evaluator 단계 진행"],
                }
            print("   ❌ 수집 검증 미통과 → 재수집 지시")
            return {
                "current_phase": "web_search",
                "supervisor_feedback": feedback,
                "iteration_count": new_iteration,
                "messages": [f"[Supervisor] 수집 NOT OK → 재수집 지시: {feedback[:80]}"],
            }

    # ── trl_eval Phase ──
    elif phase == "trl_eval":
        trl_assessments = state.get("trl_assessments", [])

        # TRL 판정이 없으면 TRL Evaluator 실행 지시
        if not trl_assessments:
            print("   → TRL Evaluator Agent에 판정 지시")
            return {
                "current_phase": "trl_eval",
                "supervisor_feedback": state.get("supervisor_feedback", "TRL 판정 시작"),
                "messages": ["[Supervisor] TRL Evaluator Agent에 TRL 판정 지시"],
            }

        # TRL 판정 결과 검증
        feedback = _review_trl(state, llm)
        print(f"   TRL 검증 결과: {feedback[:100]}...")

        if feedback.startswith("OK"):
            print("   ✅ TRL 검증 통과 → Draft 단계로 전환")
            return {
                "current_phase": "draft",
                "supervisor_feedback": "OK",
                "iteration_count": 0,   # 페이즈 전환 시 카운터 리셋
                "messages": ["[Supervisor] TRL 검증 OK → Draft Generation 단계로 전환"],
            }
        else:
            new_iteration = iteration + 1
            if new_iteration >= MAX_ITERATIONS:
                print(f"   ⚠ 최대 반복 도달 → 강제 Draft 진행")
                return {
                    "current_phase": "draft",
                    "supervisor_feedback": "Safety Valve: 현재 TRL 판정으로 Draft 진행",
                    "iteration_count": 0,   # 페이즈 전환 시 카운터 리셋
                    "messages": ["[Supervisor] Safety Valve 발동 — 강제 Draft 단계 진행"],
                }
            print("   ❌ TRL 검증 미통과 → 재판정 지시")
            return {
                "current_phase": "trl_eval",
                "supervisor_feedback": feedback,
                "iteration_count": new_iteration,
                "trl_assessments": [],   # 리셋하여 재판정 유도
                "messages": [f"[Supervisor] TRL NOT OK → 재판정 지시: {feedback[:80]}"],
            }

    # ── draft Phase ──
    elif phase == "draft":
        if not state.get("draft_report"):
            print("   → Draft Generation Agent에 초안 작성 지시")
            return {
                "current_phase": "draft",
                "supervisor_feedback": state.get("supervisor_feedback", "초안 작성 시작"),
                "messages": ["[Supervisor] Draft Generation Agent에 초안 작성 지시"],
            }

        feedback = _review_draft(state, llm)
        print(f"   검증 결과: {feedback[:100]}...")

        if feedback.startswith("OK"):
            print("   ✅ 초안 검증 통과 → Formatting 단계로 전환")
            return {
                "current_phase": "formatting",
                "supervisor_feedback": "OK",
                "iteration_count": 0,   # 페이즈 전환 시 카운터 리셋
                "messages": ["[Supervisor] 초안 검증 OK → Formatting 단계로 전환"],
            }
        else:
            new_iteration = iteration + 1
            if new_iteration >= MAX_ITERATIONS:
                print("   ⚠ 최대 반복 도달 → 강제 Formatting 진행")
                return {
                    "current_phase": "formatting",
                    "supervisor_feedback": "Safety Valve: 현재 초안으로 포맷 진행",
                    "iteration_count": 0,   # 페이즈 전환 시 카운터 리셋
                    "messages": ["[Supervisor] Safety Valve 발동 — 강제 Formatting 진행"],
                }
            print("   ❌ 초안 검증 미통과 → 수정 지시")
            return {
                "current_phase": "draft",
                "supervisor_feedback": feedback,
                "iteration_count": new_iteration,
                "draft_report": None,   # 초안 리셋하여 재작성 유도
                "messages": [f"[Supervisor] 초안 NOT OK → 수정 지시: {feedback[:80]}"],
            }

    # ── formatting Phase ──
    elif phase == "formatting":
        if not state.get("final_report"):
            print("   → Formatting Node에 규격화 지시")
            return {
                "current_phase": "formatting",
                "supervisor_feedback": "검증 OK: request formatting",
                "messages": ["[Supervisor] Formatting Node에 규격화 지시"],
            }

        feedback = _review_formatting(state, llm)
        print(f"   최종 확인: {feedback[:100]}...")

        if feedback.startswith("OK"):
            print("   ✅ 최종 확인 완료 → END")
            return {
                "current_phase": "done",
                "supervisor_feedback": "OK — 최종 승인",
                "iteration_count": 0,
                "messages": ["[Supervisor] 생성 확인 후 → END"],
            }
        else:
            new_iteration = iteration + 1
            if new_iteration >= MAX_ITERATIONS:
                print("   ⚠ 최대 반복 도달 → 강제 종료")
                return {
                    "current_phase": "done",
                    "supervisor_feedback": "Safety Valve: 규격 미충족이나 최대 반복 도달로 종료",
                    "iteration_count": 0,
                    "messages": ["[Supervisor] Safety Valve 발동 — 강제 종료"],
                }
            print("   ❌ 규격 미충족 → 재포맷")
            return {
                "current_phase": "formatting",
                "supervisor_feedback": feedback,
                "iteration_count": new_iteration,
                "final_report": None,
                "messages": [f"[Supervisor] 규격 미충족 → 재포맷: {feedback[:80]}"],
            }

    # done
    return {
        "current_phase": "done",
        "supervisor_feedback": "완료",
        "messages": ["[Supervisor] 파이프라인 완료"],
    }


def supervisor_route(state: AgentState) -> str:
    """Supervisor의 조건부 라우팅 함수.

    supervisor_node가 실행된 후 업데이트된 state를 받아 라우팅을 결정한다.
    supervisor_node가 current_phase를 업데이트하면 그 값이 반영된 상태로 호출된다.
    """
    phase = state.get("current_phase", "web_search")

    if phase == "done":
        return "end"
    elif phase == "web_search":
        return "web_search"
    elif phase == "trl_eval":
        return "trl_eval"
    elif phase == "draft":
        return "draft"
    elif phase == "formatting":
        return "formatting"

    # 알 수 없는 phase는 안전하게 종료
    print(f"   ⚠ 알 수 없는 phase: {phase} → 강제 종료")
    return "end"
