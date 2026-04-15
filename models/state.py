"""AgentState 및 데이터 모델 정의"""
from typing import TypedDict, List, Optional, Literal, Annotated
from pydantic import BaseModel, Field
import operator


# ─────────────────────────────────────────────
# 데이터 모델 (Pydantic)
# ─────────────────────────────────────────────

class WebSearchResult(BaseModel):
    """웹 검색 결과 단위"""
    source: str = Field(description="URL")
    title: str = Field(description="문서 제목")
    content: str = Field(description="문서 내용")
    date: str = Field(default="", description="발행일")
    reliability: float = Field(default=0.6, description="출처 신뢰도 0.0~1.0")
    category: str = Field(
        default="news",
        description="paper|patent|news|job_posting|earnings_call|conference|official"
    )


class TRLAssessment(BaseModel):
    """기술별·회사별 TRL 판정 결과"""
    technology: str = Field(description="HBM4 / PIM / CXL")
    company: str = Field(description="SK하이닉스 / 삼성전자 / 마이크론")
    trl_level: int = Field(description="TRL 1~9")
    confidence: float = Field(description="판정 신뢰도 0.0~1.0")
    evidence: List[str] = Field(default_factory=list, description="근거 목록")
    is_estimated: bool = Field(default=False, description="TRL 4~6 간접 추정 여부")
    estimation_basis: Optional[str] = Field(default=None, description="간접 추정 근거 유형")
    counter_evidence: Optional[str] = Field(default=None, description="반박 근거")


class DraftReport(BaseModel):
    """보고서 초안 (가이드 지정 목차 구조)"""
    summary: str = Field(description="SUMMARY (1/2페이지 이내)")
    section_1_background: str = Field(description="1. 분석 배경")
    section_2_tech_status: str = Field(description="2. 분석 대상 기술 현황")
    section_3_competitor: str = Field(description="3. 경쟁사 동향 분석")
    section_4_implications: str = Field(description="4. 전략적 시사점")
    references: List[str] = Field(default_factory=list, description="REFERENCE 목록")


# ─────────────────────────────────────────────
# AgentState (LangGraph State)
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    """LangGraph 전체 상태"""
    # 입력
    query: str
    target_technologies: List[str]

    # Web Search Agent 출력
    web_results: Annotated[List[dict], operator.add]  # WebSearchResult dicts
    search_queries_used: Annotated[List[str], operator.add]
    bias_check_log: Annotated[List[str], operator.add]
    vector_db_status: str

    # TRL Evaluator Agent 출력
    trl_assessments: List[dict]  # TRLAssessment dicts (TRL Evaluator Agent 전담)

    # Draft Generation Agent 출력
    draft_report: Optional[str]  # 마크다운 형태의 보고서 초안

    # Supervisor 제어
    iteration_count: int
    max_iterations: int
    supervisor_feedback: str
    current_phase: str  # "web_search" | "trl_eval" | "draft" | "formatting" | "done"

    # Formatting Node 출력
    final_report: Optional[str]  # 최종 보고서 (마크다운)
    format_check: Optional[str]

    # 메시지 로그
    messages: Annotated[List[str], operator.add]
