"""
🌿 감사 도우미 — 투서 분석 · 조사 흐름도 · 문답서 (A4 PDF / 그림 출력)

실행:  streamlit run app.py      (이 폴더에서 실행)
설치:  pip install -r requirements.txt

1. 투서(PDF·사진)를 올리면 사건을 파악하고 주요 쟁점을 정리합니다.
2. 조사 착수부터 결론·처분까지 단계별 조사 흐름도를 만듭니다.
3. 신고자·피신고자·참고인별 문답서를 만듭니다.
모든 결과는 A4 크기 PDF 또는 그림(PNG)으로 내려받을 수 있습니다.
"""
import html
import io
import os
import re
import time
import zipfile
from datetime import date
from typing import List, Tuple

import pymupdf
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(APP_DIR, "fonts")

MODELS = ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
MAX_INLINE_BYTES = 18 * 1024 * 1024  # Gemini 인라인 전송 한도(요청 전체 20MB) 안전 마진
IMAGE_MIME = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp",
              "heic": "image/heic", "heif": "image/heif"}
UPLOAD_TYPES = ["pdf"] + list(IMAGE_MIME)

SYSTEM_PROMPT = """당신은 공공기관 감사실 조사관을 돕는 조사 도우미입니다.
1. 투서는 신고자의 '주장'이지 확정된 사실이 아닙니다. 사실·주장·추정을 구분하세요.
2. 원문에 없는 내용을 만들지 말고, 불명확하면 '원문에 명시 없음'이라고 쓰세요.
3. 무죄추정 원칙, 피신고자의 방어권, 신고자 보호(신원 노출·2차 가해 방지)를 지키세요.
4. 건조하고 객관적인 공공기관 문서체로 쓰고 단정적 표현은 피하세요.
5. 법령 조문 번호는 확실할 때만 쓰세요.
6. 모든 출력은 한국어로 작성하세요."""

STATUS_RULES = {
    "아직 모름": "피신고자 신분이 정해지지 않았으므로, 결론·처분 경로는 공무원인 경우(「국가공무원법」·「공무원 징계령」)와 "
                "우정실무원(공무직·기간제)인 경우(「우정사업본부 공무직 및 기간제근로자 관리규정」·근로계약서·「근로기준법」)를 "
                "구분해 각각 제시하고, 신분 확인을 첫 단계에 포함하세요.",
    "공무원": "피신고자는 공무원입니다(조사관 지정). 「국가공무원법」 제56조(성실 의무)·제57조(복종의 의무)·제58조(직장 이탈 금지)·"
             "제59조(친절·공정의 의무)·제60조(비밀 엄수의 의무)·제61조(청렴의 의무)·제63조(품위 유지의 의무)·제78조(징계 사유)·"
             "제79조(징계의 종류)와 「공무원 징계령」·같은 시행규칙의 징계기준(비위의 정도 × 고의·과실)을 적용합니다. "
             "징계는 파면·해임·강등·정직(중징계), 감봉·견책(경징계)이며, 경미하면 경고·주의 처분을 검토합니다. "
             "공무직 관리규정은 적용하지 않습니다.",
    "우정실무원": "피신고자는 우정실무원(공무직·기간제근로자)입니다(조사관 지정). 「우정사업본부 공무직 및 기간제근로자 관리규정」"
               "(복무의무, 직장 내 괴롭힘, 제69조 징계사유와 <별표 3> 징계 조치기준, 제70조 징계의 종류: 해고·정직·감급·견책, "
               "인사위원회 의결), 「우정실무원 근로계약서」 준수사항, 「근로기준법」(제23조 해고 등의 제한, 제76조의2 직장 내 괴롭힘의 금지)을 "
               "적용합니다. 경미하면 경고·주의 조치를 검토합니다. 「국가공무원법」과 공무원 징계기준은 적용하지 않습니다.",
}

ROLE_FOCUS = {
    "피신고자": "의혹 사실의 인정·부인, 소명 기회 부여, 반박 자료·증인, 업무 권한·결재 구조, 관련 기록(메신저·메일·출입·결재)의 "
            "존재, 관련 복무·징계 규정 인지 여부를 확인한다. 방어권을 보장하고 유도신문을 피한다.",
    "신고자": "주장의 구체성(일시·장소·행위), 직접 경험 여부, 근거 자료, 목격자, 신고 경위·시점, 신고 이후 불이익 여부를 확인한다. "
           "신고자 보호와 2차 피해 방지를 고려한다.",
    "참고인": "직접 목격·청취한 사실과 전해 들은 내용을 구분하고, 관련자와의 관계, 진술의 시점·근거, 이해관계를 확인한다.",
}


# =====================================================================
# AI 응답 형식
# =====================================================================
class Party(BaseModel):
    name: str = Field(description="이름 또는 직위 (원문 표기, 모르면 '미상')")
    role: str = Field(description="신고자 / 피신고자 / 참고인 / 기타 중 하나")
    relation: str = Field(description="사건과의 관계")


class KeyIssue(BaseModel):
    issue: str = Field(description="조사로 판단해야 할 핵심 쟁점 (질문 형태)")
    why: str = Field(description="이 쟁점이 중요한 이유")
    standard: str = Field(description="판단 기준: 위반이 되기 위한 요건")
    to_prove: str = Field(description="사실로 인정하려면 확인·입증되어야 할 사항")
    how_to_check: str = Field(description="확인 방법 (누구에게 묻고 어떤 자료를 볼지)")


class Claim(BaseModel):
    claim: str = Field(description="신고자가 주장하는 개별 사실")
    evidence: str = Field(description="투서에 언급된 증거 (없으면 '언급 없음')")
    verifiability: str = Field(description="검증 가능성: 높음 / 보통 / 낮음")


class Unclear(BaseModel):
    point: str = Field(description="모순·불명확·근거 없는 단정으로 보이는 지점")
    why: str = Field(description="조사에서 중요한 이유")


class EvidenceItem(BaseModel):
    item: str = Field(description="확보할 자료 (CCTV, 메신저, 출입기록, 결재문서 등)")
    where: str = Field(description="확보처")
    priority: str = Field(description="상 / 중 / 하")


class CaseAnalysis(BaseModel):
    title: str = Field(description="사건을 한 줄로 표현한 제목 (실명 제외, 직위·업무 중심)")
    summary: str = Field(description="사건 요약 3~5문장")
    who: str
    when: str
    where: str
    what: str
    how: str
    why: str
    case_types: List[str] = Field(description="사안 유형 (예: 직장 내 괴롭힘, 복무위반, 금품수수, 성비위, 예산·계약 비위)")
    severity: str = Field(description="상 / 중 / 하 (주장이 사실일 경우를 가정)")
    severity_reason: str
    parties: List[Party]
    key_issues: List[KeyIssue] = Field(description="주요 쟁점 3~6개 (가장 중요한 것부터)")
    claims: List[Claim]
    unclear_points: List[Unclear]
    evidence_to_secure: List[EvidenceItem]
    cautions: List[str] = Field(description="조사 시 유의사항 (신고자 보호, 비밀 유지, 증거 보전, 방어권 등)")


class FlowStep(BaseModel):
    action: str = Field(description="해야 할 일 (짧은 제목)")
    detail: str = Field(description="구체적으로 어떻게 하는지")
    target: str = Field(description="대상자 또는 자료")
    output: str = Field(description="이 단계의 산출물 (예: 신고자 문답서, CCTV 사본)")


class FlowPhase(BaseModel):
    title: str = Field(description="단계 이름 (예: 사전 검토, 신고자 면담)")
    goal: str = Field(description="이 단계의 목표 한 문장")
    steps: List[FlowStep] = Field(description="이 단계에서 할 일 2~4개")
    decision: str = Field(description="이 단계 끝에서 판단할 질문 (예/아니오로 답할 수 있게). 분기가 없으면 빈 문자열")
    if_yes: str = Field(description="'예'일 때 다음 조치. decision이 없으면 빈 문자열")
    if_no: str = Field(description="'아니오'일 때 다음 조치. decision이 없으면 빈 문자열")


class IssueJudgment(BaseModel):
    issue: str = Field(description="주요 쟁점")
    how_to_judge: str = Field(description="어떤 사실이 확인되면 인정/불인정으로 판단하는지")


class ConclusionPath(BaseModel):
    condition: str = Field(description="조사 결과 확인된 상황")
    conclusion: str = Field(description="결론 (인정 / 일부 인정 / 불인정 / 확인 불가 등)")
    disposition: str = Field(description="처분·조치 방향 (신분에 맞는 규정 기준)")


class InvestigationFlow(BaseModel):
    overview: str = Field(description="조사 전략 요약 2~3문장")
    phases: List[FlowPhase] = Field(description="조사 착수부터 보고까지 실제 진행 순서의 5~8단계")
    judgments: List[IssueJudgment] = Field(description="주요 쟁점별 판단 방법")
    conclusion_paths: List[ConclusionPath] = Field(description="결론 경로 3~5개")
    checklist: List[str] = Field(description="보고서 작성·종결 전 확인할 사항")
    cautions: List[str] = Field(description="조사 과정 유의사항")


class Question(BaseModel):
    category: str = Field(description="질문 분류 (인적사항·관계 / 사실관계 / 시기·장소·방법 / 증거·물증 / 경위·동기 / 진술 확인)")
    question: str = Field(description="조사관이 묻는 질문 (건조하고 객관적인 어조, 한 번에 하나만 질문)")
    check_point: str = Field(description="조사관이 답변에서 확인할 핵심 한 줄")
    follow_ups: List[str] = Field(description="답변이 인정·부인·모호할 때 이어갈 후속 질문 0~2개")


class Questionnaire(BaseModel):
    target: str
    target_role: str
    purpose: str = Field(description="이 문답의 조사 목적 한 문장")
    opening_notice: str = Field(description="조사 시작 시 고지사항 (조사 목적, 비밀 유지, 진술의 임의성, 의견진술·자료제출 기회, 불이익 금지)")
    questions: List[Question]
    closing_questions: List[str] = Field(description="마무리 질문 2~3개 (추가 진술·자료, 제3자 접촉 자제 안내 등)")


# =====================================================================
# 입력 처리 · Gemini 호출
# =====================================================================
_PII = [
    (re.compile(r"(?<!\d)\d{6}\s?-\s?[1-4]\d{6}(?!\d)"), "[주민등록번호]"),
    (re.compile(r"(?<!\d)01[016789][-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)"), "[휴대전화]"),
    (re.compile(r"(?<!\d)0(?:2|[3-6]\d)[-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)"), "[전화번호]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"), "[이메일]"),
]


def mask_pii(text: str) -> str:
    for pattern, label in _PII:
        text = pattern.sub(label, text)
    return text


def file_mime(name: str) -> str:
    ext = name.rsplit(".", 1)[-1].lower()
    return "application/pdf" if ext == "pdf" else IMAGE_MIME.get(ext, "application/octet-stream")


@st.cache_data(show_spinner=False)
def pdf_text(data: bytes) -> List[str]:
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            return [p.get_text().strip() for p in doc]
    except Exception:  # noqa: BLE001
        return []


def build_source(files: list, mask: bool) -> dict:
    """텍스트가 있는 PDF는 글자로, 스캔본 PDF·사진은 파일 그대로 AI에 전달한다."""
    texts, parts = [], []
    for name, data in files:
        mime = file_mime(name)
        pages = pdf_text(data) if mime == "application/pdf" else []
        if pages and sum(len(p) for p in pages) / len(pages) >= 30:
            texts.append(f"===== 파일: {name} =====")
            texts += [f"[p.{i}]\n{mask_pii(t) if mask else t}" for i, t in enumerate(pages, 1) if t]
        else:
            parts.append((data, mime))
    return {"text": "\n\n".join(texts), "parts": parts}


def source_block(src: dict) -> str:
    note = "\n\n(첨부된 스캔본·사진 파일도 투서 원문입니다. 직접 판독해 함께 분석하세요.)" if src["parts"] else ""
    return (src["text"] or "(텍스트 없음 — 첨부 파일을 판독하세요.)") + note


def contents_for(prompt: str, src: dict):
    parts = [types.Part.from_bytes(data=b, mime_type=m) for b, m in src["parts"]]
    return parts + [prompt] if parts else prompt


class AIError(RuntimeError):
    def __init__(self, message: str, attempts: List[str]):
        super().__init__(message)
        self.attempts = attempts


def ask_ai(api_key: str, models: List[str], prompt: str, src: dict, schema, temperature: float = 0.2):
    client = genai.Client(api_key=api_key.strip(), http_options=types.HttpOptions(timeout=180_000))
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT, temperature=temperature, max_output_tokens=20000,
        response_mime_type="application/json", response_schema=schema)
    attempts = []
    for model in models:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(model=model, contents=contents_for(prompt, src), config=config)
                parsed = getattr(resp, "parsed", None)
                if isinstance(parsed, schema):
                    return parsed
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (resp.text or "").strip())
                if not text:
                    raise ValueError("빈 응답 (안전 필터 또는 출력 한도)")
                return schema.model_validate_json(text)
            except Exception as e:  # noqa: BLE001
                msg, low = str(e), str(e).lower()
                attempts.append(f"[{model}] {msg}")
                if "api key not valid" in low or "api_key_invalid" in low:
                    raise AIError("API 키가 올바르지 않아요. 키를 다시 확인해 주세요.", attempts) from e
                if any(k in low for k in ("404", "not_found", "not found", "no longer available")):
                    break
                if any(k in low for k in ("429", "resource_exhausted", "500", "503", "unavailable", "overloaded",
                                          "deadline", "timeout")) and attempt < 2:
                    time.sleep(3 * (attempt + 1))
                    continue
                break
    raise AIError("AI 호출에 실패했어요. 잠시 후 다시 시도하거나 다른 모델을 골라 주세요.", attempts)


def analysis_prompt(src: dict, status: str) -> str:
    return f"""공공기관 감사실에 접수된 투서(민원)입니다. 조사관이 사건을 빠르게 파악하고 주요 쟁점을 잡을 수 있도록 분석하세요.

- title은 실명 없이 직위·업무 중심으로 쓰세요.
- who~why는 6하원칙이며, 원문에 없으면 '원문에 명시 없음'이라고 쓰세요.
- key_issues가 가장 중요합니다. 조사로 판단해야 할 핵심 쟁점을 중요한 순서로 3~6개 뽑고, 각각 판단 기준(위반 요건),
  입증해야 할 사항, 확인 방법(누구에게 무엇을 묻고 어떤 자료를 볼지)을 구체적으로 쓰세요.
- claims는 신고 주장을 사실 단위로 나누고, unclear_points에는 원문 안의 모순·모호한 부분·근거 없는 단정을 쓰세요.
- evidence_to_secure에는 CCTV, 메신저, 출입·근태 기록, 결재문서 등 확보할 자료와 확보처를 쓰세요.
- 참고: {STATUS_RULES[status]}

[투서 원문]
{source_block(src)}"""


def flow_prompt(src: dict, a: CaseAnalysis, status: str) -> str:
    return f"""아래 사건 분석과 투서를 바탕으로, 조사관이 이 사건을 '어떻게 조사하고 결론을 내야 하는지' 단계별 조사 흐름도를 만드세요.

- phases: 사전 검토(관할·신분·증거보전) → 면담·문답 → 자료 확인 → 사실관계 확정 → 판단 → 보고·처분 요구로 이어지는
  실제 진행 순서로 5~8단계를 쓰세요. 이 사건에 맞게 구체적으로(누구를, 무엇을, 어떻게) 쓰고 일반론은 피하세요.
- 판단이 갈리는 단계에는 decision(예/아니오 질문)과 if_yes, if_no(각각의 다음 조치)를 쓰세요.
  예: '신고자 진술을 뒷받침하는 객관적 자료가 있는가?'
- judgments: 주요 쟁점마다 어떤 사실이 확인되면 인정/불인정으로 판단하는지 쓰세요.
  직장 내 괴롭힘이면 지위·관계 우위 이용, 업무상 적정범위 초과, 신체적·정신적 고통 또는 근무환경 악화를 요건별로 쓰세요.
- conclusion_paths: 조사 결과 상황별 결론과 처분 방향을 쓰세요.
- 적용 신분: {STATUS_RULES[status]}

[사건 분석(JSON)]
{a.model_dump_json(indent=1)}

[투서 원문]
{source_block(src)}"""


def questionnaire_prompt(src: dict, a: CaseAnalysis, target: str, role: str, n: int, status: str) -> str:
    key = "피신고자" if "피신고" in role else "신고자" if "신고" in role else "참고인"
    return f"""사건 분석과 투서를 바탕으로 '{target}'({role})에 대한 문답서 질문을 작성하세요.

- 본 질문은 정확히 {n}개 (마무리 질문은 별도).
- 순서: 인적사항·관계 → 사실관계(개방형) → 시기·장소·방법 → 증거·물증 → 경위·동기 → 진술 확인(모순 지점).
- 대상자 초점: {ROLE_FOCUS[key]}
- 사건 분석의 key_issues와 unclear_points를 질문에 반드시 반영해, 쟁점을 판단하는 데 필요한 사실을 끌어내세요.
- 유도신문, 한 문장에 두 가지를 묻는 질문, 감정적 표현은 쓰지 마세요.
- 피신고자 신분: {status} ({STATUS_RULES[status][:60]}…)

[사건 분석(JSON)]
{a.model_dump_json(indent=1)}

[투서 원문]
{source_block(src)}"""


# =====================================================================
# A4 문서 (HTML → PDF / PNG)
# =====================================================================
def _font_files() -> Tuple[str, dict]:
    """(글꼴 폴더, {굵기: 파일명}). 앱 fonts 폴더 → Windows 맑은 고딕 → 리눅스 나눔고딕 순."""
    options = [
        (FONT_DIR, {"normal": "Pretendard-Regular.ttf", "bold": "Pretendard-Bold.ttf"}),
        ("C:/Windows/Fonts", {"normal": "malgun.ttf", "bold": "malgunbd.ttf"}),
        ("/usr/share/fonts/truetype/nanum", {"normal": "NanumGothic.ttf", "bold": "NanumGothicBold.ttf"}),
    ]
    for folder, files in options:
        if all(os.path.exists(os.path.join(folder, f)) for f in files.values()):
            return folder, files
    raise FileNotFoundError("한글 글꼴을 찾지 못했어요. fonts 폴더의 Pretendard 파일을 확인해 주세요.")


C = {"ink": "#2F2A45", "sub": "#7A7394", "pri": "#6E5BD9", "pri_soft": "#F2EFFF", "line": "#E1DBFF",
     "mint": "#1F9E86", "mint_soft": "#E7F7F2", "peach": "#E0892B", "peach_soft": "#FFF4E6",
     "rose": "#D65A7A", "rose_soft": "#FDEEF2"}


def doc_css() -> str:
    _, f = _font_files()
    return f"""
@font-face {{font-family: K; src: url({f['normal']});}}
@font-face {{font-family: K; src: url({f['bold']}); font-weight: bold;}}
body {{font-family: K; font-size: 9.6pt; color: {C['ink']}; line-height: 1.55;}}
p {{margin: 2pt 0;}}
.band {{background-color: {C['pri_soft']}; padding: 12pt 14pt; margin-bottom: 10pt; border-left: 4pt solid {C['pri']};}}
.kicker {{font-size: 8.5pt; color: {C['pri']}; font-weight: bold; letter-spacing: 1pt;}}
.band h1 {{font-size: 17pt; margin: 3pt 0 4pt 0; color: {C['ink']};}}
.meta {{font-size: 8.5pt; color: {C['sub']};}}
h2 {{font-size: 12pt; color: {C['pri']}; margin: 14pt 0 6pt 0; padding-bottom: 3pt; border-bottom: 1pt solid {C['line']};}}
.card {{background-color: #FBFAFF; border: 0.8pt solid {C['line']}; padding: 8pt 10pt; margin: 5pt 0;}}
.soft {{color: {C['sub']}; font-size: 8.6pt;}}
table {{border-collapse: collapse; width: 100%; margin: 4pt 0;}}
th {{color: {C['pri']}; font-weight: bold; text-align: left; border-bottom: 1.4pt solid {C['pri']};}}
td, th {{border: 0.6pt solid {C['line']}; padding: 4pt 6pt; vertical-align: top;}}
td.k {{color: {C['pri']}; font-weight: bold; white-space: nowrap; padding-right: 10pt; border-right: 1.4pt solid {C['line']};}}
td.nw, th.nw {{white-space: nowrap;}}
.hi {{color: {C['rose']}; font-weight: bold;}}
.mid {{color: {C['peach']}; font-weight: bold;}}
.lo {{color: {C['mint']}; font-weight: bold;}}
.issue-h {{font-size: 10.5pt; font-weight: bold; margin-bottom: 3pt;}}
.num {{color: {C['pri']}; font-weight: bold;}}
.phase-h {{background-color: {C['pri']}; color: white; padding: 5pt 9pt; font-weight: bold; font-size: 10.5pt;}}
.phase {{border: 0.8pt solid {C['line']}; margin: 0;}}
.phase-b {{padding: 6pt 9pt;}}
.goal {{color: {C['sub']}; font-size: 8.6pt; margin-bottom: 3pt;}}
.arrow {{text-align: center; color: #B7ACF5; font-size: 13pt; margin: 1pt 0;}}
.dec {{background-color: {C['peach_soft']}; border: 0.8pt solid #F6C98E; padding: 6pt 9pt; text-align: center;
       font-weight: bold; color: #9A5A12;}}
.yes {{background-color: {C['mint_soft']}; color: #17705F; padding: 5pt 9pt; border-left: 3pt solid {C['mint']}; margin-top: 3pt;}}
.no {{background-color: {C['rose_soft']}; color: #9B3452; padding: 5pt 9pt; border-left: 3pt solid {C['rose']}; margin-top: 3pt;}}
.goal-box {{background-color: {C['mint_soft']}; border-left: 4pt solid {C['mint']}; padding: 8pt 10pt; margin: 6pt 0;}}
.center {{text-align: center;}}
.qtitle {{font-size: 22pt; font-weight: bold; text-align: center; letter-spacing: 10pt; margin: 4pt 0 2pt 0;}}
.qsub {{text-align: center; color: {C['sub']}; font-size: 9pt; margin-bottom: 10pt;}}
.notice {{background-color: {C['pri_soft']}; padding: 8pt 10pt; margin: 8pt 0; font-size: 9pt;}}
.cat {{font-size: 8.5pt; color: {C['pri']}; font-weight: bold; margin-top: 9pt;}}
.q {{font-weight: bold; margin: 3pt 0 2pt 0; font-size: 10pt;}}
.cp {{font-size: 8.4pt; color: {C['mint']}; margin: 0 0 2pt 12pt;}}
.fu {{font-size: 8.4pt; color: {C['sub']}; margin: 0 0 2pt 12pt;}}
.a {{font-weight: bold; color: {C['pri']}; margin-top: 3pt;}}
.line {{border-bottom: 0.7pt dotted #B3A8EC; height: 17pt; margin: 0;}}
.sign td {{padding: 10pt 6pt;}}
"""


def e(x) -> str:
    return html.escape(str(x or ""))


def _sev_cls(v: str) -> str:
    return "hi" if "상" in v else "mid" if "중" in v else "lo"


def html_analysis(a: CaseAnalysis, status: str) -> List[str]:
    """사건 분석 문서. 쪽 나눔 단위(덩어리) 목록을 반환한다. 제목(h2)은 다음 덩어리와 붙여 둔다."""
    six = "".join(f'<tr><td class="k">{k}</td><td>{e(v)}</td></tr>' for k, v in
                  [("누가", a.who), ("언제", a.when), ("어디서", a.where), ("무엇을", a.what), ("어떻게", a.how), ("왜", a.why)])
    b = [f"""<div class="band"><div class="kicker">CASE ANALYSIS · 사건 분석</div><h1>{e(a.title)}</h1>
<div class="meta">작성일 {date.today():%Y. %m. %d.} · 사안 유형: {e(', '.join(a.case_types))} ·
중대성(잠정) <span class="{_sev_cls(a.severity)}">{e(a.severity)}</span> · 피신고자 신분: {e(status)}</div></div>""",
         f'<h2>1. 한눈에 보기</h2><div class="card">{e(a.summary)}</div>',
         f'<table>{six}</table><p class="soft">중대성 판단 근거: {e(a.severity_reason)}</p>']
    for i, k in enumerate(a.key_issues, 1):
        b.append(("<h2>2. 주요 쟁점</h2>" if i == 1 else "") + f"""<div class="card">
<div class="issue-h"><span class="num">쟁점 {i}</span> &nbsp;{e(k.issue)}</div>
<table><tr><td class="k">왜 중요한가</td><td>{e(k.why)}</td></tr>
<tr><td class="k">판단 기준</td><td>{e(k.standard)}</td></tr>
<tr><td class="k">입증할 사항</td><td>{e(k.to_prove)}</td></tr>
<tr><td class="k">확인 방법</td><td>{e(k.how_to_check)}</td></tr></table></div>""")
    b.append('<h2>3. 관련자</h2><table><tr><th>성명·직위</th><th>구분</th><th>사건과의 관계</th></tr>'
             + "".join(f"<tr><td>{e(p.name)}</td><td class='nw'>{e(p.role)}</td><td>{e(p.relation)}</td></tr>"
                       for p in a.parties) + "</table>")
    b.append('<h2>4. 신고 주장 정리</h2><table><tr><th>주장</th><th>언급된 증거</th><th class="nw">검증 가능성</th></tr>'
             + "".join(f'<tr><td>{e(c.claim)}</td><td>{e(c.evidence)}</td><td class="nw">{e(c.verifiability)}</td></tr>'
                       for c in a.claims) + "</table>")
    if a.unclear_points:
        b.append('<h2>5. 모순·불명확한 부분</h2><table><tr><th>지점</th><th>조사상 중요성</th></tr>'
                 + "".join(f"<tr><td>{e(u.point)}</td><td>{e(u.why)}</td></tr>" for u in a.unclear_points) + "</table>")
    b.append('<h2>6. 확보할 증거</h2><table><tr><th>자료</th><th>확보처</th><th class="nw">우선순위</th></tr>'
             + "".join(f'<tr><td>{e(x.item)}</td><td>{e(x.where)}</td>'
                       f'<td class="nw {_sev_cls(x.priority)}">{e(x.priority)}</td></tr>' for x in a.evidence_to_secure)
             + "</table>")
    b.append('<h2>7. 조사 시 유의사항</h2>' + "".join(f"<p>• {e(c)}</p>" for c in a.cautions))
    return b


def html_flow(f: InvestigationFlow, a: CaseAnalysis, status: str) -> List[str]:
    b = [f"""<div class="band"><div class="kicker">INVESTIGATION FLOW · 조사 흐름도</div><h1>{e(a.title)}</h1>
<div class="meta">작성일 {date.today():%Y. %m. %d.} · 피신고자 신분: {e(status)}</div></div>
<div class="goal-box"><b>조사 전략</b><br>{e(f.overview)}</div>"""]
    for i, p in enumerate(f.phases, 1):
        rows = "".join(f'<tr><td><b>{e(s.action)}</b><br><span class="soft">{e(s.detail)}</span></td>'
                       f'<td>{e(s.target)}</td><td>{e(s.output)}</td></tr>' for s in p.steps)
        head = "<h2>1. 조사 진행 흐름</h2>" if i == 1 else '<p class="arrow">▼</p>'
        b.append(f"""{head}<div class="phase"><div class="phase-h">STEP {i} · {e(p.title)}</div><div class="phase-b">
<div class="goal">목표 · {e(p.goal)}</div>
<table><tr><th>할 일</th><th class="nw">대상·자료</th><th class="nw">산출물</th></tr>{rows}</table></div></div>""")
        if p.decision.strip():
            b.append(f'<p class="arrow">▼</p><div class="dec">◆ 판단 · {e(p.decision)}</div>'
                     f'<div class="yes"><b>예 →</b> {e(p.if_yes)}</div>'
                     f'<div class="no"><b>아니오 →</b> {e(p.if_no)}</div>')
    b.append('<p class="arrow">▼</p><div class="dec">★ 결론 도출</div>')
    b.append('<h2>2. 쟁점별 판단 방법</h2><table><tr><th>쟁점</th><th>이렇게 판단합니다</th></tr>'
             + "".join(f"<tr><td><b>{e(j.issue)}</b></td><td>{e(j.how_to_judge)}</td></tr>" for j in f.judgments)
             + "</table>")
    b.append('<h2>3. 결론 경로</h2><table><tr><th>조사 결과 이런 경우</th><th class="nw">결론</th>'
             '<th>처분·조치 방향</th></tr>'
             + "".join(f'<tr><td>{e(c.condition)}</td><td class="nw"><b>{e(c.conclusion)}</b></td>'
                       f'<td>{e(c.disposition)}</td></tr>' for c in f.conclusion_paths) + "</table>")
    b.append('<h2>4. 종결 전 체크리스트</h2>' + "".join(f"<p>☐ {e(c)}</p>" for c in f.checklist))
    if f.cautions:
        b.append('<h2>5. 유의사항</h2>' + "".join(f"<p>• {e(c)}</p>" for c in f.cautions))
    return b


def html_questionnaire(q: Questionnaire, investigator: bool, answer_lines: int) -> List[str]:
    lines = '<div class="line">&nbsp;</div>' * answer_lines
    blank_dt = "20&nbsp;&nbsp;&nbsp;.&nbsp;&nbsp;&nbsp;&nbsp;.&nbsp;&nbsp;&nbsp;&nbsp;.&nbsp;&nbsp;&nbsp;&nbsp;:"
    b = [f'<div class="qtitle">문 답 서</div><div class="qsub">{e(q.target_role)} 대상 · {e(q.purpose)}</div>'
         + '<table>' + "".join(f'<tr><td class="k">{k}</td><td>{v}</td></tr>' for k, v in [
             ("진 술 인", f"{e(q.target)} ({e(q.target_role)})"), ("소속·직위", ""), ("조사 일시", blank_dt),
             ("조사 장소", ""), ("조 사 관", "")]) + "</table>",
         f'<div class="notice"><b>조사에 앞서 안내드립니다</b><br>{e(q.opening_notice)}</div>']
    cat = None
    for i, item in enumerate(q.questions, 1):
        head = ""
        if item.category != cat:
            cat = item.category
            head = f'<div class="cat">■ {e(cat)}</div>'
        extra = ""
        if investigator:
            extra = f'<div class="cp">✔ 확인 포인트: {e(item.check_point)}</div>' + "".join(
                f'<div class="fu">↳ 후속 질문: {e(x)}</div>' for x in item.follow_ups)
        b.append(f'{head}<div class="q">문 {i}. {e(item.question)}</div>{extra}<div class="a">답 :</div>{lines}')
    for j, cq in enumerate(q.closing_questions, len(q.questions) + 1):
        head = '<div class="cat">■ 마무리</div>' if j == len(q.questions) + 1 else ""
        b.append(f'{head}<div class="q">문 {j}. {e(cq)}</div><div class="a">답 :</div>{lines}')
    b.append("""<p style="margin-top:14pt">위 진술 내용을 열람(낭독)하였는바, 진술한 대로 기재되었음을 확인합니다.</p>
<p class="center" style="margin:8pt 0">20&nbsp;&nbsp;&nbsp;&nbsp;.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;.</p>
<table class="sign"><tr><td class="k">진 술 인</td><td>(서명 또는 인)</td><td class="k">조 사 관</td><td>(서명 또는 인)</td></tr></table>""")
    return b


@st.cache_data(show_spinner=False)
def render_pdf(docs: Tuple[Tuple[str, ...], ...], footer: str) -> bytes:
    """문서(덩어리 목록) 여러 개를 A4 PDF로 만든다. 각 문서는 새 쪽에서 시작한다.
    덩어리가 남은 공간에 다 들어가지 않으면 통째로 다음 쪽으로 넘겨, 쟁점·단계·질문이 쪽 사이에서 끊기지 않게 한다.
    (한 쪽보다 긴 덩어리만 나눠 싣는다.)"""
    folder, fonts = _font_files()
    archive, css = pymupdf.Archive(folder), doc_css()
    a4 = pymupdf.paper_rect("a4")
    body = a4 + (48, 46, -48, -50)  # 좌우 약 17mm, 위 16mm, 아래 18mm
    buf = io.BytesIO()
    writer = pymupdf.DocumentWriter(buf)
    for blocks in docs:
        dev, y = writer.begin_page(a4), body.y0
        for blk in blocks:
            story = pymupdf.Story(html=blk, user_css=css, archive=archive)
            more, filled = story.place(pymupdf.Rect(body.x0, y, body.x1, body.y1))
            if more and y > body.y0 + 1:  # 남은 공간 부족 → 다음 쪽에서 통째로
                writer.end_page()
                dev, y = writer.begin_page(a4), body.y0
                story = pymupdf.Story(html=blk, user_css=css, archive=archive)
                more, filled = story.place(body)
            story.draw(dev)
            while more:  # 한 쪽보다 긴 덩어리
                writer.end_page()
                dev = writer.begin_page(a4)
                more, filled = story.place(body)
                story.draw(dev)
            y = pymupdf.Rect(filled).y1 + 2
        writer.end_page()
    writer.close()
    doc = pymupdf.open(stream=buf.getvalue(), filetype="pdf")
    font_path = os.path.join(folder, fonts["normal"])
    font = pymupdf.Font(fontfile=font_path)
    for n, page in enumerate(doc, 1):
        page.insert_font(fontname="kf", fontfile=font_path)
        right = f"{n} / {doc.page_count}"
        page.draw_line((48, a4.height - 34), (a4.width - 48, a4.height - 34), color=(0.88, 0.86, 1), width=0.6)
        if footer:
            page.insert_text((48, a4.height - 22), footer, fontname="kf", fontsize=7.5, color=(0.48, 0.45, 0.58))
        page.insert_text((a4.width - 48 - font.text_length(right, 7.5), a4.height - 22), right,
                         fontname="kf", fontsize=7.5, color=(0.48, 0.45, 0.58))
    try:
        doc.subset_fonts()  # 쓴 글자만 담아 파일 크기를 줄임
    except Exception:  # noqa: BLE001
        pass
    out = doc.tobytes(garbage=3, deflate=True)
    doc.close()
    return out


@st.cache_data(show_spinner=False)
def pdf_to_png(pdf: bytes, dpi: int) -> List[bytes]:
    with pymupdf.open(stream=pdf, filetype="pdf") as doc:
        return [p.get_pixmap(dpi=dpi).tobytes("png") for p in doc]


# =====================================================================
# 화면
# =====================================================================
APP_CSS = """
<style>
@import url("https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css");
html, body, .stApp, .stMarkdown, button, input, textarea, select, label, p, li, h1, h2, h3, h4 {
  font-family: 'Pretendard', -apple-system, 'Malgun Gothic', sans-serif !important; }
.stApp { background: linear-gradient(180deg, #F4F0FF 0%, #FBFAFF 38%, #FFFFFF 100%); }
.block-container { max-width: 1100px; padding-top: 2.2rem; }
.hero { background: linear-gradient(120deg, #EAE4FF 0%, #FCEEF4 55%, #E4F6F0 100%);
  border-radius: 28px; padding: 30px 34px; margin-bottom: 18px; box-shadow: 0 8px 30px rgba(110, 91, 217, .10); }
.hero .t { font-size: 2.05rem; font-weight: 800; color: #3A2F7A; letter-spacing: -0.5px; }
.hero .s { font-size: 1.02rem; color: #5F5880; margin-top: 6px; }
.steps { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 18px; }
.step { background: rgba(255,255,255,.75); border-radius: 999px; padding: 7px 16px; font-size: .92rem;
  color: #6E6790; border: 1px solid #E6E0FF; }
.step.done { background: #6E5BD9; color: white; border-color: #6E5BD9; }
.cardbox { background: white; border-radius: 22px; padding: 22px 24px; border: 1px solid #EEEAFF;
  box-shadow: 0 6px 22px rgba(110, 91, 217, .07); margin-bottom: 14px; }
.stat { background: white; border-radius: 20px; padding: 16px 18px; border: 1px solid #EEEAFF;
  box-shadow: 0 4px 16px rgba(110, 91, 217, .06); }
.stat .l { font-size: .85rem; color: #8A83A8; } .stat .v { font-size: 1.55rem; font-weight: 800; color: #3A2F7A; }
.issue { background: #FBFAFF; border: 1px solid #ECE7FF; border-radius: 16px; padding: 12px 16px; margin: 8px 0; }
.issue b { color: #6E5BD9; }
.hint { color: #8A83A8; font-size: .9rem; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 22px !important; border-color: #EEEAFF !important;
  background: rgba(255,255,255,.9); box-shadow: 0 6px 22px rgba(110, 91, 217, .06); }
.stButton > button, .stDownloadButton > button { border-radius: 999px !important; padding: .55rem 1.25rem;
  font-weight: 600; border: 1px solid #DCD5FF; transition: all .15s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(110, 91, 217, .18); border-color: #B9ACFF; }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, #7C6CF2, #A98BF7) !important;
  border: none !important; color: white !important; }
.stTabs [role="tablist"], .stTabs [data-baseweb="tab-list"] { gap: 10px; border: none; }
.stTabs [role="tab"] { border-radius: 999px !important; background: #F1EEFF; padding: 9px 22px !important;
  height: auto; border: 1px solid #E6E0FF; transition: all .15s ease; }
.stTabs [role="tab"]:hover { background: #E6E0FF; }
.stTabs [role="tab"] p { font-size: 1rem; font-weight: 600; margin: 0; }
.stTabs [role="tab"][aria-selected="true"] { background: #6E5BD9 !important; border-color: #6E5BD9; }
.stTabs [role="tab"][aria-selected="true"] p { color: white !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
[data-testid="stFileUploaderDropzone"] { border-radius: 20px; border: 2px dashed #CFC5FF; background: #FAF8FF; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #F5F2FF, #FDFBFF); }
[data-testid="stImage"] img { border-radius: 10px; box-shadow: 0 6px 24px rgba(47, 42, 69, .12); }
</style>
"""


def default_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "")
    if key:
        return key
    try:
        return st.secrets.get("GEMINI_API_KEY", "")
    except Exception:  # noqa: BLE001
        return ""


def sidebar() -> dict:
    sb = st.sidebar
    sb.markdown("### 🌿 설정")
    api_key = sb.text_input("🔑 Gemini API 키", type="password", value=default_api_key(),
                            help="환경변수 GEMINI_API_KEY 또는 .streamlit/secrets.toml에 넣어 두면 자동으로 채워져요.")
    model = sb.selectbox("🤖 AI 모델", MODELS, help="고른 모델이 안 되면 다음 모델로 자동으로 바꿔 시도해요.")
    sb.markdown("#### 👤 피신고자 신분")
    status = sb.radio("피신고자 신분", list(STATUS_RULES), horizontal=True, label_visibility="collapsed",
                      help="조사 흐름도의 결론·처분 경로와 문답서 질문에 반영돼요.")
    n_q = sb.slider("📝 문답서 질문 수", 5, 20, 10)
    mask = sb.toggle("🔒 전화번호·주민번호·이메일 가리기", value=True,
                     help="글자로 된 PDF에만 적용돼요. 사진·스캔본은 AI가 이미지를 직접 읽어요.")
    sb.markdown('<p class="hint">무료 API는 입력 내용이 Google 서비스 개선에 쓰일 수 있어요. '
                '민감한 사건은 기관 보안 지침을 먼저 확인해 주세요.</p>', unsafe_allow_html=True)
    return {"api_key": api_key, "models": [model] + [m for m in MODELS if m != model], "status": status,
            "n_q": n_q, "mask": mask}


def hero(ss):
    done = [ss.analysis is not None, ss.flow is not None, bool(ss.qs)]
    labels = ["① 투서 분석 · 주요 쟁점", "② 조사 흐름도", "③ 문답서"]
    steps = "".join(f'<span class="step {"done" if d else ""}">{"✓ " if d else ""}{l}</span>' for l, d in zip(labels, done))
    st.markdown(f"""<div class="hero"><div class="t">🌿 감사 도우미</div>
<div class="s">투서를 올려 주시면 사건 분석부터 조사 흐름, 문답서까지 차근차근 함께 준비해 드릴게요.</div>
<div class="steps">{steps}</div></div>""", unsafe_allow_html=True)


def show_error(err: Exception):
    st.error(str(err))
    if getattr(err, "attempts", None):
        with st.expander("자세한 오류 보기"):
            for line in err.attempts:
                st.code(line, language="text")


def output_panel(pdf: bytes, name: str, key: str):
    """내려받기 버튼 + A4 미리보기."""
    pngs = pdf_to_png(pdf, 150)
    c1, c2 = st.columns(2)
    c1.download_button("📄 PDF 받기 (A4)", pdf, file_name=f"{name}.pdf", mime="application/pdf",
                       key=f"pdf_{key}", width="stretch")
    if len(pngs) == 1:
        c2.download_button("🖼️ 그림 받기 (PNG)", pngs[0], file_name=f"{name}.png", mime="image/png",
                           key=f"png_{key}", width="stretch")
    else:
        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            for i, p in enumerate(pngs, 1):
                z.writestr(f"{name}_{i:02d}.png", p)
        c2.download_button(f"🖼️ 그림 받기 ({len(pngs)}장)", zbuf.getvalue(), file_name=f"{name}_그림.zip",
                           mime="application/zip", key=f"png_{key}", width="stretch")
    st.markdown(f'<p class="hint">A4 {len(pngs)}쪽 · 그림은 쪽마다 150dpi(1240×1754) PNG'
                + (", 여러 쪽이면 ZIP으로 묶어 드려요." if len(pngs) > 1 else "예요.") + "</p>", unsafe_allow_html=True)
    previews = pdf_to_png(pdf, 80)
    cols = st.columns(2)
    for i, img in enumerate(previews):
        cols[i % 2].image(img, caption=f"{i + 1}쪽", width="stretch")


def stat(col, label, value):
    col.markdown(f'<div class="stat"><div class="l">{label}</div><div class="v">{value}</div></div>',
                 unsafe_allow_html=True)


def sample_case():
    """API 키 없이 화면을 둘러볼 수 있는 가상의 예시 사건."""
    a = CaseAnalysis(
        title="○○우편집중국 소포팀장의 팀원 대상 폭언 및 업무 배제 의혹",
        summary="신고자는 소포팀장이 2026년 8월부터 주간 회의에서 반복적으로 모욕적인 발언을 하고, 신고자를 주요 업무에서 "
                "배제했다고 주장함. 동료 2명이 목격했다고 기재하였으나 녹취 등 객관적 자료 보유 여부는 원문에 명시 없음. "
                "주장이 사실일 경우 직장 내 괴롭힘에 해당할 소지가 있음.",
        who="소포팀장(피신고자)이 소포팀 실무원(신고자)에게", when="2026. 8.~9. (8. 12. 회의 등 일부 일자 명시)",
        where="소포팀 사무실, 주간 회의", what="모욕적 발언, 주요 업무 배제", how="여러 팀원이 있는 회의 자리에서 반복 발언",
        why="원문에 명시 없음", case_types=["직장 내 괴롭힘", "복무위반"], severity="중",
        severity_reason="반복성과 목격자가 주장되나 객관적 자료는 아직 확인되지 않음",
        parties=[Party(name="김○○", role="신고자", relation="소포팀 우정실무원"),
                 Party(name="이○○", role="피신고자", relation="소포팀장"),
                 Party(name="박○○", role="참고인", relation="동료, 회의 참석·목격 주장")],
        key_issues=[
            KeyIssue(issue="피신고자가 회의 중 신고자에게 모욕적 발언을 하였는가?", why="괴롭힘 성립의 핵심 행위",
                     standard="지위 우위 이용, 업무상 적정범위 초과, 정신적 고통 또는 근무환경 악화",
                     to_prove="발언 일시·내용·반복성, 목격자 진술의 일치 여부",
                     how_to_check="참고인 2명 개별 면담, 회의록·업무 메신저 확인"),
            KeyIssue(issue="업무 배제가 정당한 업무 조정이었는가?", why="업무상 적정범위 판단의 기준",
                     standard="업무상 필요성과 상당성, 다른 팀원과의 형평", to_prove="배제 경위와 사유, 결재 여부",
                     how_to_check="업무분장표 변경 이력, 근무표 비교, 팀장 소명")],
        claims=[Claim(claim="8. 12. 회의에서 '일도 못하면서'라고 발언", evidence="동료 2명 목격 주장", verifiability="보통"),
                Claim(claim="8월 말부터 소포 구분 주업무에서 제외", evidence="언급 없음", verifiability="높음")],
        unclear_points=[Unclear(point="폭언이 시작된 시점이 7월과 8월로 다르게 적혀 있음", why="반복성·기간 판단에 영향")],
        evidence_to_secure=[EvidenceItem(item="주간 회의록", where="소포팀", priority="상"),
                            EvidenceItem(item="업무분장표 변경 이력", where="물류총괄과", priority="상"),
                            EvidenceItem(item="업무 메신저 대화", where="신고자·피신고자", priority="중")],
        cautions=["신고자 신원이 드러나지 않게 조사 순서를 조정", "신고자와 피신고자 분리 필요성 검토",
                  "참고인 간 진술 맞추기 방지를 위해 개별 면담"])
    f = InvestigationFlow(
        overview="목격자 진술과 업무분장 자료로 발언·업무 배제 사실을 먼저 확정하고, 괴롭힘 3요건을 차례로 검토함.",
        phases=[
            FlowPhase(title="사전 검토", goal="관할·신분 확인과 증거 보전",
                      steps=[FlowStep(action="신분 확인", detail="피신고자 신분(공무원/우정실무원)과 직위 확인",
                                      target="인사기록", output="신분 확인 결과"),
                             FlowStep(action="증거 보전", detail="회의록·메신저 보존 요청", target="소포팀", output="보존 요청 공문")],
                      decision="", if_yes="", if_no=""),
            FlowPhase(title="신고자 면담", goal="주장을 일시·장소·발언 단위로 구체화",
                      steps=[FlowStep(action="사실확인 면담", detail="발언 내용·일시·목격자 특정", target="신고자",
                                      output="신고자 문답서")],
                      decision="신고자 진술이 구체적이고 목격자가 특정되는가?", if_yes="참고인 면담 진행",
                      if_no="추가 자료 요청 후 종결 여부 검토"),
            FlowPhase(title="참고인 면담 · 자료 확인", goal="진술을 뒷받침하는 사실 확보",
                      steps=[FlowStep(action="참고인 개별 면담", detail="직접 목격 여부와 전해 들은 내용 구분",
                                      target="박○○ 등", output="참고인 문답서"),
                             FlowStep(action="업무분장 대조", detail="배제 전후 업무 비교", target="업무분장표",
                                      output="비교표")],
                      decision="", if_yes="", if_no=""),
            FlowPhase(title="피신고자 조사", goal="소명 기회 부여와 반박 자료 확인",
                      steps=[FlowStep(action="피신고자 문답", detail="확인된 사실 제시 후 인정·부인과 사유 청취",
                                      target="이○○", output="피신고자 문답서")],
                      decision="3요건(우위·적정범위 초과·고통)이 모두 인정되는가?", if_yes="괴롭힘 인정, 처분 수준 검토",
                      if_no="부적절 행위 여부와 주의 조치 검토")],
        judgments=[IssueJudgment(issue="모욕적 발언 여부",
                                 how_to_judge="참고인 중 1명 이상이 일시·내용을 구체적으로 일치 진술하거나 기록이 있으면 인정"),
                   IssueJudgment(issue="업무 배제의 정당성",
                                 how_to_judge="결재된 업무 조정 사유가 없고 신고자에게만 불리하게 적용되었으면 적정범위 초과로 판단")],
        conclusion_paths=[ConclusionPath(condition="발언·업무 배제 모두 확인", conclusion="인정",
                                         disposition="관리규정에 따라 인사위원회 회부(견책 이상 의견), 분리조치"),
                          ConclusionPath(condition="발언만 일부 확인", conclusion="일부 인정",
                                         disposition="경고 또는 주의, 재발 방지 교육"),
                          ConclusionPath(condition="진술만 대립, 자료 없음", conclusion="확인 불가",
                                         disposition="종결하되 모니터링과 신고자 보호 조치")],
        checklist=["피신고자에게 소명 기회를 주었는가", "신고자 보호조치를 확인했는가", "모든 판단에 근거 자료를 달았는가"],
        cautions=["2차 가해 방지", "조사 내용 비밀 유지"])
    qs = {}
    for name, role, items in [
        ("김○○", "신고자", [("인적사항·관계", "현재 소속과 담당 업무, 피신고자와의 관계를 말씀해 주십시오.", "지위 관계 확인", []),
                           ("사실관계", "2026. 8. 12. 회의에서 있었던 일을 순서대로 말씀해 주십시오.", "발언의 구체성",
                            ["그 자리에 누가 있었습니까?"]),
                           ("증거·물증", "당시 상황을 뒷받침할 기록이나 자료가 있습니까?", "객관적 자료 유무", [])]),
        ("이○○", "피신고자", [("인적사항·관계", "현재 소속과 담당 업무를 말씀해 주십시오.", "지위 관계 확인", []),
                            ("사실관계", "2026. 8. 12. 주간 회의에서 있었던 일을 말씀해 주십시오.", "발언 인정 여부",
                             ["신고자에게 한 말을 기억나는 대로 말씀해 주십시오."]),
                            ("경위·동기", "8월 말 업무분장을 바꾼 이유는 무엇입니까?", "업무상 필요성", ["결재를 받았습니까?"])]),
    ]:
        qs[f"{name} ({role})"] = Questionnaire(
            target=name, target_role=role, purpose="민원 내용의 사실관계 확인" + (" 및 소명 기회 부여" if role == "피신고자" else ""),
            opening_notice="이 조사는 접수된 민원의 사실관계를 확인하기 위한 것입니다. 진술은 자유로운 의사에 따라 하시면 되고, "
                           "조사 내용은 비밀로 유지됩니다. 의견을 말하거나 자료를 낼 기회가 보장되며, 진술로 인한 불이익은 없습니다.",
            questions=[Question(category=c, question=q, check_point=cp, follow_ups=fu) for c, q, cp, fu in items],
            closing_questions=["추가로 진술하거나 제출할 자료가 있습니까?", "오늘 조사 내용은 다른 직원에게 말하지 말아 주시기 바랍니다. 이해하셨습니까?"])
    return a, f, qs


def main():
    st.set_page_config(page_title="감사 도우미", page_icon="🌿", layout="wide")
    st.markdown(APP_CSS, unsafe_allow_html=True)
    ss = st.session_state
    for k, v in {"analysis": None, "flow": None, "qs": {}, "src": None, "sig": None, "status_used": None,
                 "sample": False}.items():
        ss.setdefault(k, v)
    cfg = sidebar()
    hero(ss)

    # ---------------- 투서 올리기 ----------------
    with st.container(border=True):
        st.markdown("#### 📮 투서 올리기")
        files = st.file_uploader("투서·민원 PDF나 사진(JPG·PNG 등)을 끌어다 놓아 주세요. 여러 개도 괜찮아요.",
                                 type=UPLOAD_TYPES, accept_multiple_files=True)
        loaded = [(f.name, f.getvalue()) for f in files or []]
        sig = tuple((n, len(d)) for n, d in loaded)
        if sig != ss.sig:
            ss.update(sig=sig, analysis=None, flow=None, qs={}, src=None, sample=False)
        size = sum(len(d) for _, d in loaded)
        go = st.button("✨ 사건 분석하기", type="primary", disabled=not loaded)
        st.markdown(f'<p class="hint">'
                    + (f"{len(loaded)}개 파일 · {size / 1024 / 1024:.1f}MB — 글자 PDF는 텍스트로, 스캔본·사진은 AI가 직접 읽어요."
                       if loaded else "파일을 올리면 분석을 시작할 수 있어요.") + "</p>", unsafe_allow_html=True)
        if go:
            if not cfg["api_key"]:
                st.warning("왼쪽 설정에 Gemini API 키를 먼저 넣어 주세요.")
            else:
                src = build_source(loaded, cfg["mask"])
                if sum(len(b) for b, _ in src["parts"]) > MAX_INLINE_BYTES:
                    st.error("사진·스캔본 합계가 18MB를 넘어요. 사진 크기를 줄이거나 나눠서 올려 주세요.")
                else:
                    with st.spinner("투서를 꼼꼼히 읽고 있어요… (보통 20~60초)"):
                        try:
                            ss.analysis = ask_ai(cfg["api_key"], cfg["models"], analysis_prompt(src, cfg["status"]),
                                                 src, CaseAnalysis)
                            ss.update(src=src, flow=None, qs={}, status_used=cfg["status"])
                            st.rerun()
                        except Exception as err:  # noqa: BLE001
                            show_error(err)

    a: CaseAnalysis = ss.analysis
    if a is None:
        st.markdown('<div class="cardbox"><b>이렇게 도와드려요</b><br><br>'
                    '① <b>투서 분석</b> — 6하원칙 요약, 관련자, <b>주요 쟁점과 판단 기준</b>, 확보할 증거를 정리해요.<br>'
                    '② <b>조사 흐름도</b> — 무엇부터 조사하고 어디서 판단이 갈리는지, 결론·처분까지의 길을 그려요.<br>'
                    '③ <b>문답서</b> — 신고자·피신고자·참고인별로 질문과 답변란이 있는 문답서를 만들어요.<br><br>'
                    '<span class="hint">모든 결과는 A4 PDF와 그림(PNG)으로 내려받을 수 있어요.</span></div>',
                    unsafe_allow_html=True)
        if st.button("🌱 예시 사건으로 먼저 둘러보기", help="API 키 없이 가상의 예시 사건으로 결과 화면과 A4 문서를 미리 볼 수 있어요."):
            sa, sf, sq = sample_case()
            ss.update(analysis=sa, flow=(sf, "우정실무원"), qs=sq, status_used=cfg["status"], sample=True,
                      src={"text": "(예시 사건 — 가상의 투서)", "parts": []})
            st.rerun()
        return

    if ss.get("sample"):
        st.info("🌱 가상의 예시 사건을 보고 있어요. 투서 파일을 올리면 예시는 사라지고 실제 분석을 시작할 수 있어요.")

    if ss.status_used != cfg["status"]:
        st.info(f"피신고자 신분을 '{cfg['status']}'(으)로 바꾸셨어요. 흐름도와 문답서는 새로 만들 때 반영돼요.")

    tab1, tab2, tab3 = st.tabs(["🔎 사건 분석", "🧭 조사 흐름도", "📝 문답서"])

    # ---------------- ① 사건 분석 ----------------
    with tab1:
        c = st.columns(4)
        stat(c[0], "주요 쟁점", f"{len(a.key_issues)}개")
        stat(c[1], "관련자", f"{len(a.parties)}명")
        stat(c[2], "확보할 증거", f"{len(a.evidence_to_secure)}건")
        stat(c[3], "중대성(잠정)", a.severity)
        st.markdown(f"#### {a.title}")
        st.markdown(f'<p class="hint">{html.escape(a.summary)}</p>', unsafe_allow_html=True)
        for i, k in enumerate(a.key_issues, 1):
            st.markdown(f'<div class="issue"><b>쟁점 {i}</b> &nbsp;{html.escape(k.issue)}<br>'
                        f'<span class="hint">판단 기준 · {html.escape(k.standard)}</span></div>', unsafe_allow_html=True)
        st.markdown("##### 📄 A4 문서")
        pdf = render_pdf((tuple(html_analysis(a, cfg["status"])),), "감사 도우미 · AI가 작성한 초안이므로 조사관의 검토가 필요합니다.")
        output_panel(pdf, "사건분석", "analysis")

    # ---------------- ② 조사 흐름도 ----------------
    with tab2:
        st.markdown('<p class="hint">조사 착수부터 결론·처분까지, 단계마다 할 일과 판단이 갈리는 지점을 그려 드려요. '
                    f'피신고자 신분(<b>{html.escape(cfg["status"])}</b>)에 맞는 처분 경로로 만들어요.</p>',
                    unsafe_allow_html=True)
        label = "🔄 흐름도 다시 만들기" if ss.flow else "🧭 조사 흐름도 만들기"
        if st.button(label, type="secondary" if ss.flow else "primary", key="mk_flow"):
            if not cfg["api_key"]:
                st.warning("API 키를 넣어 주세요.")
            else:
                with st.spinner("조사 순서와 판단 지점을 정리하고 있어요…"):
                    try:
                        ss.flow = (ask_ai(cfg["api_key"], cfg["models"], flow_prompt(ss.src, a, cfg["status"]),
                                          ss.src, InvestigationFlow), cfg["status"])
                        st.rerun()
                    except Exception as err:  # noqa: BLE001
                        show_error(err)
        if ss.flow:
            flow, used = ss.flow
            pdf = render_pdf((tuple(html_flow(flow, a, used)),), "감사 도우미 · AI가 작성한 초안이므로 조사관의 검토가 필요합니다.")
            output_panel(pdf, "조사흐름도", "flow")

    # ---------------- ③ 문답서 ----------------
    with tab3:
        labels = [f"{p.name} ({p.role})" for p in a.parties]
        by_label = {f"{p.name} ({p.role})": p for p in a.parties}
        default = [l for l in labels if "신고자" in l or "피신고자" in l][:2] or labels[:1]
        c1, c2 = st.columns([3, 2])
        chosen = c1.multiselect("누구의 문답서를 만들까요?", labels, default=default)
        extra_name = c2.text_input("목록에 없는 사람 추가", placeholder="예: 인사팀장")
        extra_role = c2.selectbox("추가한 사람의 구분", ["참고인", "신고자", "피신고자"]) if extra_name.strip() else None
        c3, c4 = st.columns([3, 2])
        version = c3.radio("문답서 종류", ["🗂️ 조사 진행용 (확인 포인트·후속 질문 포함)", "✍️ 출력·작성용 (질문과 답변란만)"],
                           horizontal=False)
        lines = c4.slider("답변란 줄 수", 2, 8, 4)
        if st.button("📝 문답서 만들기", type="primary", disabled=not (chosen or extra_name.strip())):
            if not cfg["api_key"]:
                st.warning("API 키를 넣어 주세요.")
            else:
                targets = [(by_label[l].name, by_label[l].role) for l in chosen]
                if extra_name.strip():
                    targets.append((extra_name.strip(), extra_role))
                bar = st.progress(0.0, text="문답서를 준비하고 있어요…")
                for i, (name, role) in enumerate(targets):
                    bar.progress(i / len(targets), text=f"{name} ({role}) 문답서 작성 중…")
                    try:
                        ss.qs[f"{name} ({role})"] = ask_ai(
                            cfg["api_key"], cfg["models"],
                            questionnaire_prompt(ss.src, a, name, role, cfg["n_q"], cfg["status"]), ss.src,
                            Questionnaire)
                    except Exception as err:  # noqa: BLE001
                        show_error(err)
                bar.progress(1.0, text="완료했어요!")
        if ss.qs:
            investigator = version.startswith("🗂️")
            st.markdown("##### 📄 A4 문서")
            pick = st.radio("볼 문답서", ["모두 합쳐서"] + list(ss.qs), horizontal=True)
            keys = list(ss.qs) if pick == "모두 합쳐서" else [pick]
            footer = "감사 도우미 · 조사 진행용 (확인 포인트 포함)" if investigator else ""
            pdf = render_pdf(tuple(tuple(html_questionnaire(ss.qs[k], investigator, lines)) for k in keys), footer)
            name = "문답서_전체" if pick == "모두 합쳐서" else f"문답서_{re.sub(r'[^0-9A-Za-z가-힣]+', '_', pick).strip('_')}"
            output_panel(pdf, name + ("_진행용" if investigator else "_작성용"), f"q_{pick}_{investigator}_{lines}")


if __name__ == "__main__":
    main()
