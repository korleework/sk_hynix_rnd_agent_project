"""Web Search Agent 프롬프트 템플릿"""

WEB_SEARCH_QUERY_GENERATION_PROMPT = """당신은 반도체 R&D 정보를 전문적으로 수집하는 리서처입니다.
주어진 기술과 기업에 대해 최적의 검색 쿼리를 생성하세요.

## 대상 기술
{target_technologies}

## 대상 기업
SK하이닉스, 삼성전자, 마이크론

## 추가 지시 (Supervisor 피드백)
{supervisor_feedback}

## 쿼리 생성 규칙
1. 각 기술 × 각 기업 조합(3기술 × 3기업 = 9조합)에 대해 정확히 2개씩 쿼리 생성
2. 영문 쿼리를 기본으로 하고 핵심 키워드 위주로 작성
3. 소스 다양성 확보를 위해 아래 유형을 균형 있게 분배:
   - 기술 현황 + 양산: "[company] [tech] development mass production 2025"
   - 특허 + 채용 + 어닝콜: "[company] [tech] patent hiring earnings 2024 2025"
4. 확증 편향 방지: 3사에 동일한 쿼리 구조를 대칭적으로 적용

## 재수집 시 규칙 (supervisor_feedback이 NOT_OK인 경우)
- 아래 이전 쿼리들은 이미 사용했으므로 **동일한 쿼리를 생성하지 마세요**
- Supervisor의 피드백에서 부족하다고 지적된 소스 유형(특허, 논문 등)에 집중하세요
- 쿼리 다양화를 위해 키워드, 연도, 문구 패턴을 변형하세요
  예) "HBM4 patent 2025" → "HBM4 patent application USPTO filing", "HBM4 IP roadmap"
- 이전 쿼리: {previous_queries}

## 출력 형식
검색 쿼리를 JSON 배열로만 반환하세요. 설명 없이 JSON만 출력합니다.
예시: ["SK hynix HBM4 mass production 2025", "Samsung HBM4 development status", ...]

반드시 정확히 18개의 쿼리를 생성하세요 (9조합 × 2개).
"""

# ─────────────────────────────────────────────
# 라운드별 특화 쿼리 생성 프롬프트 (include_domains 기반 다중 라운드 검색)
# ─────────────────────────────────────────────

WEB_SEARCH_QUERY_GENERATION_GENERAL_PROMPT = """당신은 반도체 R&D 정보를 전문적으로 수집하는 리서처입니다.
**일반 웹 검색용** 쿼리를 생성하세요. 뉴스·공식발표·어닝콜·기술 현황을 타깃으로 합니다.

## 대상 기술
{target_technologies}

## 대상 기업
SK하이닉스, 삼성전자, 마이크론

## 추가 지시 (Supervisor 피드백)
{supervisor_feedback}

## 이전 사용 쿼리 (재수집 시 회피 대상)
{previous_queries}

## 쿼리 생성 규칙
- 3기술 × 3기업 = 9조합에 대해 **정확히 1개씩 총 9개 쿼리** 생성
- 뉴스/양산/공식발표/어닝콜 중심 키워드 (예: "mass production", "roadmap", "earnings")
- 영문 쿼리를 기본으로 핵심 키워드 위주
- 이전 쿼리와 동일하면 안 됨

## 출력 형식
JSON 배열만 출력. 설명 텍스트 금지.
예시: ["SK hynix HBM4 mass production 2025", "Samsung HBM4 roadmap earnings", ...]

정확히 9개 쿼리.
"""

WEB_SEARCH_QUERY_GENERATION_ACADEMIC_PROMPT = """당신은 반도체 R&D 정보를 전문적으로 수집하는 리서처입니다.
**학술/특허 전용 검색**용 쿼리를 생성하세요. 아래 도메인에서만 검색됩니다:
arxiv.org, ieee.org, patents.google.com, scholar.google.com, semanticscholar.org, nature.com, sciencedirect.com

## 대상 기술
{target_technologies}

## 쿼리 생성 규칙
- 3기술에 대해 **논문 전용 1개 + 특허 전용 1개 = 총 6개 쿼리** 생성
- 논문용: "[tech] architecture paper 2025", "[tech] IEEE ISSCC circuit design"
- 특허용: "[tech] patent application memory", "[tech] USPTO filing"
- 기업명은 필요 시만 포함 (논문/특허 검색은 기술 키워드 중심)
- 영문 쿼리 필수

## 이전 사용 쿼리 (재수집 시 회피 대상)
{previous_queries}

## 출력 형식
JSON 배열만 출력. 설명 텍스트 금지.
예시: ["HBM4 architecture paper 2025", "HBM4 patent application USPTO", "PIM IEEE ISSCC 2025", ...]

정확히 6개 쿼리 (3기술 × 2타입).
"""

WEB_SEARCH_QUERY_GENERATION_JOBS_PROMPT = """당신은 반도체 R&D 정보를 전문적으로 수집하는 리서처입니다.
**채용 공고 전용 검색**용 쿼리를 생성하세요. 아래 도메인에서만 검색됩니다:
linkedin.com, indeed.com, greenhouse.io, lever.co, careers.skhynix.com, careers.samsung.com, careers.micron.com

TRL 4~6 간접 추정 지표로 활용됩니다 (어떤 기술에 엔지니어를 모집하는지 = 개발 활성도).

## 대상 기술
{target_technologies}

## 대상 기업
SK하이닉스, 삼성전자, 마이크론

## 쿼리 생성 규칙
- 3기업에 대해 **정확히 1개씩 총 3개 쿼리** 생성
- 기술 키워드 + "engineer", "hiring", "job" 조합
- 예: "SK hynix HBM engineer hiring 2025", "Samsung PIM design engineer job", "Micron CXL controller engineer"

## 이전 사용 쿼리 (재수집 시 회피 대상)
{previous_queries}

## 출력 형식
JSON 배열만 출력. 설명 텍스트 금지.

정확히 3개 쿼리.
"""


BIAS_CHECK_PROMPT = """아래 수집 결과의 편향을 점검하세요.

## 수집 결과 통계
- 시간 분포: {time_distribution}
- 소스 유형 분포: {source_distribution}
- 기업별 정보량: {company_distribution}

## 점검 항목
1. 시간 편향: 6개월 이내 정보가 60% 이상인가?
2. 소스 편향: 3종 이상의 소스 유형이 포함되어 있는가?
3. 기업 편향: 3사 정보량이 대칭적인가? (최다 vs 최소 비율이 2:1 이내)

각 항목에 대해 PASS/FAIL과 설명을 제공하세요.
"""
