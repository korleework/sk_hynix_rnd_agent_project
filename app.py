"""
SK하이닉스 차세대 반도체 R&D 전략 분석 에이전트
- LangGraph Supervisor 패턴
- Web Search Agent → TRL Evaluator Agent → Draft Generation Agent → Formatting Node
"""
import os
import sys
from datetime import datetime

from langgraph.graph import StateGraph, END

from models.state import AgentState
from agents.supervisor import supervisor_node, supervisor_route
from agents.web_search import web_search_agent_node
from agents.trl_evaluator import trl_evaluator_agent_node
from agents.draft_gen import draft_generation_node
from nodes.formatter import formatting_node
from config import MAX_ITERATIONS, TARGET_TECHNOLOGIES


def build_graph() -> StateGraph:
    """LangGraph 워크플로우 구성"""

    workflow = StateGraph(AgentState)

    # ── 노드 등록 ──
    workflow.add_node("supervisor",             supervisor_node)
    workflow.add_node("web_search_agent",       web_search_agent_node)
    workflow.add_node("trl_evaluator_agent",    trl_evaluator_agent_node)
    workflow.add_node("draft_generation_agent", draft_generation_node)
    workflow.add_node("formatting_node",        formatting_node)

    # ── 엔트리 포인트 ──
    workflow.set_entry_point("supervisor")

    # ── Supervisor → 다음 노드 (조건부 라우팅) ──
    workflow.add_conditional_edges(
        "supervisor",
        supervisor_route,
        {
            "web_search":  "web_search_agent",
            "trl_eval":    "trl_evaluator_agent",
            "draft":       "draft_generation_agent",
            "formatting":  "formatting_node",
            "end":         END,
        }
    )

    # ── 각 에이전트/노드 → Supervisor 복귀 ──
    workflow.add_edge("web_search_agent",       "supervisor")
    workflow.add_edge("trl_evaluator_agent",    "supervisor")
    workflow.add_edge("draft_generation_agent", "supervisor")
    workflow.add_edge("formatting_node",        "supervisor")

    return workflow.compile()


def run(query: str = None):
    """에이전트 실행"""
    if query is None:
        query = (
            "SK하이닉스의 차세대 반도체 기술(HBM4, PIM, CXL)에 대한 "
            "경쟁사(삼성전자, 마이크론) 대비 기술 성숙도(TRL)를 분석하고, "
            "R&D 투자 우선순위에 대한 전략적 시사점을 도출하세요."
        )

    print("=" * 60)
    print("🚀 SK하이닉스 R&D 전략 분석 에이전트 시작")
    print("=" * 60)
    print(f"쿼리: {query}")
    print(f"대상 기술: {', '.join(TARGET_TECHNOLOGIES)}")
    print(f"최대 반복: {MAX_ITERATIONS}회")
    print("파이프라인: Web Search → TRL Evaluator → Draft → Formatting")
    print("=" * 60)

    # 그래프 빌드
    app = build_graph()

    # 초기 상태
    initial_state = {
        "query": query,
        "target_technologies": TARGET_TECHNOLOGIES,
        "web_results": [],
        "search_queries_used": [],
        "bias_check_log": [],
        "vector_db_status": "",
        "trl_assessments": [],
        "draft_report": None,
        "iteration_count": 0,
        "max_iterations": MAX_ITERATIONS,
        "supervisor_feedback": "",
        "current_phase": "web_search",
        "final_report": None,
        "format_check": None,
        "messages": [],
    }

    # 실행
    final_state = None
    for step in app.stream(initial_state, {"recursion_limit": 40}):
        for node_name, output in step.items():
            if "messages" in output:
                for msg in output["messages"]:
                    print(f"  📌 {msg}")
        final_state = step

    print("\n" + "=" * 60)
    print("✅ 파이프라인 완료")
    print("=" * 60)

    # 최종 결과 출력
    if final_state:
        last_node = list(final_state.keys())[0]
        last_output = final_state[last_node]

        if "final_report" in last_output and last_output["final_report"]:
            report = last_output["final_report"]
            today = datetime.now().strftime('%Y-%m-%d')
            print(f"\n📄 최종 보고서 길이: {len(report)}자")
            print(f"📁 저장 위치: outputs/report_{today}.md")

    return final_state


if __name__ == "__main__":
    # 커맨드라인에서 쿼리 전달 가능
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        run(query=user_query)
    else:
        run()
