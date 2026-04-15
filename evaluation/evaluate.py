"""
Retrieval 평가 스크립트 — Hit Rate@K, MRR 산출
참조: langchain-v1/14-Retriever/10-RetrieverEvaluation.ipynb

사용법:
  python evaluation/evaluate.py --create-sample   # 샘플 eval 데이터 생성 + 실제 retrieval 수행
  python evaluation/evaluate.py                   # 평가 수행 (Hit Rate@5, MRR)
"""
import json
import math
import os
import re
from collections import Counter
from typing import Dict, List


# ─────────────────────────────────────────────
# 평가 지표
# ─────────────────────────────────────────────

def hit_rate_at_k(
    queries: List[str],
    retrieved_docs: List[List[str]],
    relevant_docs: List[List[str]],
    k: int = 5,
) -> float:
    """Hit Rate@K — 상위 K개 중 정답이 하나라도 포함된 쿼리 비율"""
    hits = 0
    for retrieved, relevant in zip(retrieved_docs, relevant_docs):
        top_k = retrieved[:k]
        if any(doc in relevant for doc in top_k):
            hits += 1
    return hits / len(queries) if queries else 0.0


def mrr(
    queries: List[str],
    retrieved_docs: List[List[str]],
    relevant_docs: List[List[str]],
) -> float:
    """MRR (Mean Reciprocal Rank) — 정답의 역순위 평균"""
    reciprocal_ranks = []
    for retrieved, relevant in zip(retrieved_docs, relevant_docs):
        rr = 0.0
        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


# ─────────────────────────────────────────────
# 경량 BM25 retriever — 외부 의존성 없음
# ─────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25:
    """BM25 Okapi — 외부 라이브러리 없이 순수 파이썬 구현."""

    def __init__(self, corpus: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.N = len(corpus)
        self.avgdl = sum(len(d) for d in corpus) / self.N if self.N else 0.0
        self.doc_freqs = [Counter(d) for d in corpus]
        df: Counter = Counter()
        for d in corpus:
            for term in set(d):
                df[term] += 1
        self.idf = {
            term: math.log(1 + (self.N - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        freqs = self.doc_freqs[doc_idx]
        dl = sum(freqs.values())
        score = 0.0
        for term in query_tokens:
            if term not in self.idf:
                continue
            f = freqs.get(term, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            score += self.idf[term] * (f * (self.k1 + 1)) / denom
        return score

    def rank(self, query: str, doc_ids: List[str]) -> List[str]:
        """쿼리와의 BM25 점수 내림차순으로 doc_ids 정렬."""
        q = _tokenize(query)
        scored = [(doc_ids[i], self.score(q, i)) for i in range(self.N)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in scored]


# ─────────────────────────────────────────────
# 샘플 말뭉치 — 정답(relevant) + 분산 문서(distractor)로 구성
# ─────────────────────────────────────────────

SAMPLE_CORPUS: Dict[str, str] = {
    # ─ HBM4 (정답 — 쿼리 키워드 일부 유지 + 의역 혼합) ─
    "skhynix_hbm4_roadmap":
        "SK hynix HBM4 roadmap 12-high stacked DRAM bandwidth interconnect 2026 tape-out plan",
    "hbm4_mass_production_news":
        "HBM4 mass production ramp begins 2025 bandwidth doubling first mover SK hynix supply",
    "samsung_hbm4_dev":
        "Samsung HBM4 development 1c node validation status qualification timeline stacked DRAM",
    "samsung_memory_strategy":
        "Samsung DS division earnings memory HBM4 strategy development leadership roadmap status",
    # ─ HBM 관련 혼동 유발 (쿼리 키워드와 높은 overlap) ─
    "hbm3e_production_sk":
        "SK hynix HBM3E mass production 2025 shipping NVIDIA H200 customer qualification ramp volume",
    "samsung_hbm3e_struggle":
        "Samsung HBM3E development status qualification NVIDIA yield process 2024 2025 memory",
    "micron_hbm3e_ramp":
        "Micron HBM3E ramp shipping 8-high 24GB NVIDIA qualification 2025 production status development",
    "hbm_market_share":
        "HBM market share SK hynix Samsung Micron 2025 mass production revenue forecast DRAM memory",
    "nvidia_hbm4_requirements":
        "NVIDIA announces HBM4 requirements Rubin architecture 2025 2026 mass production qualification",
    # ─ CXL (정답 — 일부 키워드 유지) ─
    "micron_cxl_product":
        "Micron CXL memory expander CZ120 DDR5 enterprise product launch scalable solution",
    "cxl_memory_expander_review":
        "CXL memory expander product benchmark comparison Samsung Micron latency evaluation review",
    "cxl3_spec_update":
        "CXL 3.0 specification switching fabric coherency memory pooling disaggregation update",
    "cxl_memory_pooling_whitepaper":
        "CXL memory pooling data center rack disaggregation hyperscaler whitepaper adoption TCO",
    # ─ CXL 혼동 유발 ─
    "samsung_cxl_smart_ssd":
        "Samsung CXL 2.0 Smart SSD memory semantic product launch enterprise data center DDR5 expander",
    "skhynix_cxl_niagara":
        "SK hynix Niagara CXL memory pooling solution demo rack server 2024 data center product",
    "cxl_vs_nvlink":
        "CXL 3.0 vs NVLink fabric data center interconnect comparison memory expander pooling bandwidth",
    "intel_cxl_sapphire":
        "Intel Sapphire Rapids CXL 1.1 support memory expander product data center enterprise 2023",
    # ─ PIM (정답 — 일부 키워드 유지) ─
    "skhynix_aimx_announcement":
        "SK hynix AiMX accelerator card GDDR6-AiM LPDDR5 compute-in-DRAM inference LLM serving",
    "pim_semiconductor_overview":
        "PIM processing-in-memory technology survey architecture bank-level compute near-DRAM taxonomy",
    # ─ PIM 혼동 유발 (쿼리의 'SK hynix'·'PIM'·'technology' 많이 공유) ─
    "samsung_hbm_pim_paper":
        "Samsung HBM-PIM ISSCC IEEE function-in-memory bank parallel LLM inference accelerator technology",
    "micron_pim_research":
        "Micron compute-near-memory PIM research UPMEM bit-serial DRAM technology",
    "samsung_lpddr_pim":
        "Samsung LPDDR-PIM mobile in-memory computing technology announcement LLM inference edge",
    "skhynix_pim_patent":
        "SK hynix PIM patent filing USPTO compute-in-memory architecture technology 2024 processor",
    # ─ 완전 분산 ─
    "ddr5_market_forecast":
        "DDR5 market forecast server client demand pricing cycle",
    "nvidia_gpu_roadmap":
        "NVIDIA GPU Blackwell roadmap H100 H200 training inference",
    "tsmc_advanced_packaging":
        "TSMC advanced packaging CoWoS SoIC 3D IC chiplet integration",
    "ai_datacenter_trends":
        "AI data center power consumption cooling cluster hyperscaler capex",
}


def _build_retrieval(eval_items: List[dict], top_n: int = 10) -> List[dict]:
    """각 query에 대해 SAMPLE_CORPUS 전체를 BM25로 랭킹하여 retrieved_docs 채움."""
    doc_ids = list(SAMPLE_CORPUS.keys())
    tokenized_corpus = [_tokenize(SAMPLE_CORPUS[d]) for d in doc_ids]
    bm25 = BM25(tokenized_corpus)

    out = []
    for item in eval_items:
        ranked = bm25.rank(item["query"], doc_ids)[:top_n]
        out.append({**item, "retrieved_docs": ranked})
    return out


# ─────────────────────────────────────────────
# 평가 러너
# ─────────────────────────────────────────────

def evaluate_retrieval(eval_data_path: str = "evaluation/eval_data.json", k: int = 5):
    with open(eval_data_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    queries = [d["query"] for d in eval_data]
    retrieved = [d["retrieved_docs"] for d in eval_data]
    relevant = [d["relevant_docs"] for d in eval_data]

    hr = hit_rate_at_k(queries, retrieved, relevant, k=k)
    m = mrr(queries, retrieved, relevant)

    print(f"{'='*40}")
    print(f"Retrieval 평가 결과")
    print(f"{'='*40}")
    print(f"  평가 쿼리 수: {len(queries)}")
    print(f"  Hit Rate@{k}: {hr:.4f}")
    print(f"  MRR:          {m:.4f}")
    print(f"{'='*40}")

    print(f"\n쿼리별 상세:")
    for i, d in enumerate(eval_data):
        top_k = d["retrieved_docs"][:k]
        hit = any(doc in d["relevant_docs"] for doc in top_k)
        rr = 0.0
        for rank, doc in enumerate(d["retrieved_docs"], 1):
            if doc in d["relevant_docs"]:
                rr = 1.0 / rank
                break
        print(f"  Q{i+1}: {'Hit ✅' if hit else 'Miss ❌'} | RR={rr:.3f} | {d['query'][:50]}")

    return {"hit_rate": hr, "mrr": m}


def create_sample_eval_data(output_path: str = "evaluation/eval_data.json"):
    """샘플 평가 데이터 생성 + BM25 retriever로 retrieved_docs 자동 채움."""
    sample_data = [
        {
            "query": "SK hynix HBM4 mass production 2025",
            "relevant_docs": ["skhynix_hbm4_roadmap", "hbm4_mass_production_news"],
        },
        {
            "query": "Samsung HBM4 development status",
            "relevant_docs": ["samsung_hbm4_dev", "samsung_memory_strategy"],
        },
        {
            "query": "Micron CXL memory expander product",
            "relevant_docs": ["micron_cxl_product", "cxl_memory_expander_review"],
        },
        {
            "query": "SK hynix AiMX PIM technology",
            "relevant_docs": ["skhynix_aimx_announcement", "pim_semiconductor_overview"],
        },
        {
            "query": "CXL 3.0 memory pooling data center",
            "relevant_docs": ["cxl3_spec_update", "cxl_memory_pooling_whitepaper"],
        },
    ]

    # BM25 retrieval 수행하여 retrieved_docs 채움
    filled = _build_retrieval(sample_data, top_n=10)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(filled, f, ensure_ascii=False, indent=2)

    print(f"📝 샘플 평가 데이터 생성 + BM25 retrieval 완료: {output_path}")
    print(f"   말뭉치: {len(SAMPLE_CORPUS)}개 문서 | 쿼리: {len(filled)}개")
    print(f"   이제 `python evaluation/evaluate.py` 실행 시 실측 Hit Rate@5 / MRR 산출\n")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--create-sample":
        create_sample_eval_data()
        # 생성 직후 바로 평가 실행 (편의 기능)
        print()
        evaluate_retrieval()
    else:
        try:
            evaluate_retrieval()
        except FileNotFoundError:
            print("평가 데이터가 없습니다. 먼저 --create-sample로 샘플을 생성하세요.")
            print("  python evaluation/evaluate.py --create-sample")
