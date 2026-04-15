"""Formatting Node — 보고서 규격화 (노드/도구, 에이전트 아님)"""
import os
import re
from datetime import datetime

# 프로젝트 루트 기준 절대경로 (formatter.py → nodes/ → project root)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from models.state import AgentState

# 고정 출력 파일명
OUTPUT_FILENAME = "ai-mini_output_2반_이현수.pdf"

# PDF 변환에 사용할 한글 CSS 스타일
_PDF_CSS = """
@charset "UTF-8";

/* ── 한글 폰트 우선순위: macOS → Windows → Linux → 웹폰트 ── */
body {
    font-family: "Apple SD Gothic Neo", "Malgun Gothic", "Noto Sans KR",
                 "NanumGothic", "나눔고딕", sans-serif;
    font-size: 10.5pt;
    line-height: 1.75;
    color: #1a1a1a;
    margin: 0;
    padding: 0;
}

@page {
    size: A4;
    margin: 25mm 20mm 25mm 20mm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-size: 9pt;
        color: #888;
    }
}

/* ── 제목 ── */
h1 {
    font-size: 18pt;
    font-weight: 700;
    color: #003366;
    border-bottom: 3px solid #003366;
    padding-bottom: 6px;
    margin-top: 0;
    margin-bottom: 16px;
}

h2 {
    font-size: 13pt;
    font-weight: 700;
    color: #004080;
    border-left: 4px solid #004080;
    padding-left: 10px;
    margin-top: 24px;
    margin-bottom: 10px;
    page-break-after: avoid;
}

h3 {
    font-size: 11pt;
    font-weight: 600;
    color: #333;
    margin-top: 16px;
    margin-bottom: 6px;
    page-break-after: avoid;
}

/* ── 인용/메타 정보 ── */
blockquote {
    background-color: #f0f4f8;
    border-left: 4px solid #4a90d9;
    margin: 12px 0;
    padding: 8px 14px;
    color: #555;
    font-size: 9.5pt;
    border-radius: 0 4px 4px 0;
}
blockquote p { margin: 0; }

/* ── 표 ── */
table {
    border-collapse: collapse;
    width: 100%;
    margin: 14px 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}
th {
    background-color: #003366;
    color: white;
    padding: 7px 10px;
    text-align: center;
    font-weight: 600;
}
td {
    border: 1px solid #ccc;
    padding: 6px 10px;
    vertical-align: top;
}
tr:nth-child(even) td {
    background-color: #f7f9fc;
}

/* ── 목록 ── */
ul, ol {
    padding-left: 20px;
    margin: 6px 0;
}
li { margin-bottom: 3px; }

/* ── 코드 ── */
code {
    background-color: #f4f4f4;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 9pt;
    font-family: "Courier New", monospace;
}

pre {
    background-color: #f4f4f4;
    padding: 10px 14px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 8.5pt;
    page-break-inside: avoid;
}

/* ── 구분선 ── */
hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 18px 0;
}

/* ── SUMMARY 강조 ── */
h2:has(+ *) + * { margin-top: 8px; }

/* ── 출처(REFERENCE) 섹션 ── */
h2#reference, h2#REFERENCE {
    font-size: 11pt;
    color: #666;
    border-left-color: #aaa;
}

p { margin: 6px 0 10px 0; }

strong { color: #222; }
em { color: #555; }
"""


def _check_format(report: str) -> str:
    """보고서 규격 체크리스트 검증"""
    checks = []

    # 1. SUMMARY 포함 여부
    if "## SUMMARY" in report or "## Summary" in report:
        try:
            summary_start = (
                report.index("## SUMMARY") if "## SUMMARY" in report
                else report.index("## Summary")
            )
            next_section = report.index("\n## 1", summary_start)
            summary_text = report[summary_start:next_section].strip()
            if len(summary_text) <= 1500:
                checks.append("✅ SUMMARY 포함, 분량 적정")
            else:
                checks.append(f"⚠ SUMMARY 포함되었으나 분량 초과 ({len(summary_text)}자)")
        except ValueError:
            checks.append("✅ SUMMARY 포함")
    else:
        checks.append("❌ SUMMARY 누락")

    # 2. 섹션 1~4 존재 여부
    required_sections = [
        ("## 1.", "분석 배경"),
        ("## 2.", "기술 현황"),
        ("## 3.", "경쟁사 동향"),
        ("## 4.", "전략적 시사점"),
    ]
    for marker, name in required_sections:
        if marker in report:
            checks.append(f"✅ 섹션 {name} 존재")
        else:
            checks.append(f"❌ 섹션 {name} 누락")

    # 3. REFERENCE 존재 여부
    if "## REFERENCE" in report or "## Reference" in report:
        checks.append("✅ REFERENCE 섹션 존재")
    else:
        checks.append("❌ REFERENCE 섹션 누락")

    # 4. 출처 번호 매칭
    ref_numbers = re.findall(r'\[(\d+)\]', report)
    if ref_numbers:
        checks.append(f"✅ 출처 번호 {len(set(ref_numbers))}개 발견")
    else:
        checks.append("⚠ 본문 내 출처 번호([1], [2]...) 미발견")

    fails = [c for c in checks if c.startswith("❌")]
    if fails:
        return "FAIL: " + "; ".join(checks)
    return "PASS: " + "; ".join(checks)


def _markdown_to_pdf(md_text: str, pdf_path: str) -> None:
    """마크다운 → HTML → PDF 변환 (weasyprint 사용).

    필요 패키지: pip install markdown weasyprint
    한글 폰트: macOS 기본폰트(Apple SD Gothic Neo) 자동 사용
    """
    import markdown
    import weasyprint

    # 마크다운 → HTML
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "nl2br"],
    )

    # 완전한 HTML 문서 조립 (UTF-8 명시)
    full_html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>SK하이닉스 R&D 전략 보고서</title>
</head>
<body>
{html_body}
</body>
</html>"""

    css = weasyprint.CSS(string=_PDF_CSS)
    weasyprint.HTML(string=full_html).write_pdf(pdf_path, stylesheets=[css])


def formatting_node(state: AgentState) -> dict:
    """Formatting Node — 보고서 규격화 및 PDF 최종본 생성"""
    print("\n📋 [Formatting Node] 보고서 규격화 시작...")

    draft = state.get("draft_report", "")
    if not draft:
        return {
            "final_report": "",
            "format_check": "FAIL: 초안이 비어있음",
            "messages": ["[Formatting Node] 초안이 비어있어 포맷 불가"],
        }

    # 보고서 헤더 추가
    today = datetime.now().strftime("%Y-%m-%d")
    techs = ", ".join(state.get("target_technologies", []))

    if not draft.startswith("# "):
        header = "# 차세대 반도체 기술 전략 보고서\n\n"
        header += f"> **작성일**: {today} | **분석 대상**: {techs}\n\n---\n\n"
        final_report = header + draft
    else:
        final_report = draft.replace(
            "# 차세대 반도체 기술 전략 보고서",
            f"# 차세대 반도체 기술 전략 보고서\n\n> **작성일**: {today} | **분석 대상**: {techs}",
            1,
        )

    # 규격 검증
    format_check = _check_format(final_report)
    print(f"   규격 검증 결과: {format_check}")

    # 출력 경로 설정
    output_dir = os.path.join(_PROJECT_ROOT, "outputs")
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, OUTPUT_FILENAME)

    # PDF 변환
    try:
        _markdown_to_pdf(final_report, pdf_path)
        print(f"   ✅ PDF 저장: {pdf_path}")
        saved_path = pdf_path
    except ImportError as e:
        # weasyprint 또는 markdown 미설치 시 마크다운 fallback
        md_path = os.path.join(output_dir, OUTPUT_FILENAME.replace(".pdf", ".md"))
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"   ⚠ PDF 라이브러리 미설치 ({e}) → 마크다운으로 저장: {md_path}")
        print(f"     설치 방법: pip install markdown weasyprint")
        saved_path = md_path
    except Exception as e:
        md_path = os.path.join(output_dir, OUTPUT_FILENAME.replace(".pdf", ".md"))
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"   ⚠ PDF 변환 오류 ({e}) → 마크다운으로 저장: {md_path}")
        saved_path = md_path

    return {
        "final_report": final_report,
        "format_check": format_check,
        "messages": [f"[Formatting Node] 보고서 규격화 완료 → {saved_path}"],
    }
