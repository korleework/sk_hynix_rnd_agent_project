"""프로젝트 설정 파일"""
import os
import warnings
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# API 키 유효성 경고
if not OPENAI_API_KEY:
    warnings.warn("⚠ OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
if not TAVILY_API_KEY:
    warnings.warn("⚠ TAVILY_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

# LLM 설정
LLM_MODEL = "gpt-4o-mini"  # 비용 절감을 위해 gpt-4o-mini 사용
# LLM_MODEL = "gpt-4o" 
LLM_TEMPERATURE = 0.2  # 분석 정확도를 위해 낮은 temperature

# 검색 설정
MAX_SEARCH_RESULTS = 10
MAX_QUERIES = 18          # 쿼리 수 상한 (rate limit 방지)
SEARCH_DEPTH = "basic"    # "basic" | "advanced" — free 티어는 basic 권장
TIME_RANGE = "month"      # Tavily 유효값: "day"|"week"|"month"|"year" ("6m" 은 무효)
SEARCH_DELAY = 1.5        # 쿼리 간 대기 시간(초) — rate limit 방지

# ─────────────────────────────────────────────
# 다중 라운드 검색 설정 (include_domains 기반 소스 다양화)
# ─────────────────────────────────────────────
# 3라운드 구성: 일반 → 학술/특허 → 채용
# 총 쿼리 수는 MAX_QUERIES를 넘지 않도록 라운드별로 할당
SEARCH_ROUND_CONFIG = [
    {
        "name": "general",         # 라운드 1: 뉴스/공식발표/어닝콜
        "queries_per_round": 9,    # 9개 쿼리 (3사 × 3기술)
        "include_domains": None,   # 도메인 제한 없음
        "time_range": "month",
    },
    {
        "name": "academic_patent", # 라운드 2: 학술/특허
        "queries_per_round": 6,    # 6개 쿼리 (3기술 × 2 타입)
        "include_domains": [
            "arxiv.org",
            "ieee.org",
            "ieeexplore.ieee.org",
            "patents.google.com",
            "scholar.google.com",
            "semanticscholar.org",
            "nature.com",
            "sciencedirect.com",
        ],
        "time_range": "year",      # 논문/특허는 1년 범위 확대
    },
    {
        "name": "jobs",            # 라운드 3: 채용 공고 (TRL 4~6 간접 지표)
        "queries_per_round": 3,    # 3개 쿼리 (3사)
        "include_domains": [
            "linkedin.com",
            "indeed.com",
            "greenhouse.io",
            "lever.co",
            "careers.skhynix.com",
            "careers.samsung.com",
            "careers.micron.com",
        ],
        "time_range": "month",
    },
]

# Supervisor 설정
MAX_ITERATIONS = 3  # 최대 재작업 횟수

# 임베딩 설정
EMBEDDING_MODEL = "BAAI/bge-m3"

# 대상 기술 및 기업
TARGET_TECHNOLOGIES = ["HBM4", "PIM", "CXL"]
TARGET_COMPANIES = ["SK하이닉스", "삼성전자", "마이크론"]

# 출처 신뢰도 스코어링
RELIABILITY_SCORES = {
    "paper": 1.0,
    "official": 1.0,
    "patent": 0.8,
    "earnings_call": 0.8,
    "news": 0.6,
    "conference": 0.7,
    "blog": 0.4,
}
