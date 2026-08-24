import re
import time

import google.generativeai as genai
import requests
import streamlit as st
from google.api_core import exceptions as gexc

NAVER_CLIENT_ID = st.secrets["NAVER_CLIENT_ID"]
NAVER_CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
MODEL_PREFERENCES = [
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
    "gemini-2.5-flash",
]

MAX_DESC_CHARS = 2000
MAX_TOTAL_ITEMS = 120      # 중복 제거 후 최종 상한 (토큰 폭주 방지)
MAX_OUTPUT_TOKENS = 16384  # thinking 모델은 사고 토큰도 여기 포함됨
MAX_RETRIES = 2
RETRY_WAIT = 65            # TPM 윈도우가 1분이라 그보다 길게
NAVER_DELAY = 0.15         # 연속 호출 간 간격 (초)

# 정렬별 수집량: 최신순에 비중을 둠 (상업적 최적화가 덜 된 글이 많음)
SORT_PLAN = [("sim", 20), ("date", 40)]

MODES = [
    "🍽️ 맛집/핫플 탐색",
    "💻 IT/기술 동향 분석",
    "✈️ 여행/데이트 코스",
    "✏️ 내 맘대로 직접 지시",
]

# 모드별 검색어 확장 접미사. 부정·후기성 단어가 협찬글을 자연스럽게 걸러냄.
QUERY_SUFFIXES = {
    MODES[0]: ["", "후기", "웨이팅", "솔직"],
    MODES[1]: ["", "후기", "단점", "문제"],
    MODES[2]: ["", "후기", "코스", "주차"],
    MODES[3]: ["", "후기"],
}


# ---------------------------------------------------------------------------
# 모델 선택
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def list_text_models() -> list[str]:
    return [
        m.name.removeprefix("models/")
        for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def resolve_model() -> str:
    available = list_text_models()

    for name in MODEL_PREFERENCES:
        if name in available:
            return name

    def version_of(name: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)", name)
        return float(m.group(1)) if m else 0.0

    flash = [n for n in available if "flash" in n.lower() and "lite" not in n.lower()]
    if flash:
        return max(flash, key=version_of)

    raise RuntimeError(f"사용 가능한 flash 계열 모델이 없습니다. 접근 가능: {available}")


# ---------------------------------------------------------------------------
# 네이버 검색
# ---------------------------------------------------------------------------
def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def _fetch_once(query: str, sort: str, display: int) -> list[dict]:
    url = "https://naverapihub.apigw.ntruss.com/search/v1/blog"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
    }
    params = {"query": query, "display": display, "sort": sort}

    res = requests.get(url, headers=headers, params=params, timeout=10)
    res.raise_for_status()
    return res.json().get("items", [])


def expand_queries(base: str, mode: str) -> list[str]:
    """모드에 맞춰 검색어를 여러 각도로 확장."""
    base = base.strip()
    seen, out = set(), []
    for suffix in QUERY_SUFFIXES.get(mode, ["", "후기"]):
        q = f"{base} {suffix}".strip()
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


@st.cache_data(ttl=600, show_spinner=False)
def collect_blogs(base_query: str, mode: str) -> tuple[list[dict], list[str]]:
    """확장 검색어 × 정렬 조합으로 수집 후 링크 기준 중복 제거."""
    queries = expand_queries(base_query, mode)

    seen: set[str] = set()
    result: list[dict] = []

    for q in queries:
        for sort, display in SORT_PLAN:
            try:
                items = _fetch_once(q, sort, display)
            except requests.RequestException:
                continue  # 일부 쿼리 실패는 무시하고 계속

            for it in items:
                link = it.get("link", "")
                if not link or link in seen:
                    continue
                seen.add(link)
                result.append({
                    "title": _strip_tags(it.get("title", "")),
                    "desc": _strip_tags(it.get("description", ""))[:MAX_DESC_CHARS],
                    "date": it.get("postdate", ""),
                    "via": q,
                })
            time.sleep(NAVER_DELAY)

    # 최신순 우선으로 정렬한 뒤 상한 적용
    result.sort(key=lambda b: b["date"], reverse=True)
    return result[:MAX_TOTAL_ITEMS], queries


def format_blogs(blogs: list[dict]) -> str:
    lines = []
    for i, b in enumerate(blogs, 1):
        d = b["date"]
        date_str = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else "날짜미상"
        lines.append(f"[{i}] ({date_str}) {b['title']}\n{b['desc']}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------
COMMON_RULES = """
[공통 원칙]
- 아래 데이터는 블로그 '본문'이 아니라 검색 결과의 짧은 요약문입니다. 정보가 부족한 항목이 많은 것이 정상입니다.
- 데이터에 없는 내용은 절대 추측하거나 지어내지 마세요. 근거가 없으면 "정보 부족"이라고 명시하세요.
- 협찬·체험단으로 의심되는 글도 버리지 말고 활용하되, 용도를 구분하세요.
  · 사실 정보(메뉴, 가격, 위치, 영업시간, 주차)는 그대로 인용해도 됩니다.
  · 평가·감상(맛있다, 분위기 좋다)은 신뢰도를 낮춰 취급하고, 필요하면 협찬 의심을 짧게 표시하세요.
- 여러 항목에서 반복 언급된 내용을 우선하고, 몇 건에서 나왔는지 밝히세요. 단일 출처면 그렇다고 쓰세요.
- 마지막에 "⚠️ 데이터 한계" 항목을 두고, 이번 분석에서 확인하지 못한 부분을 2~3줄로 정리하세요.
"""

SYSTEM_PROMPTS = {
    MODES[0]: """
[분석 지침 — 맛집/핫플]
1. 작성 날짜를 확인하세요. 최근 12개월 내 언급이 없는 곳은 폐업 가능성을 함께 표시하세요.
2. 추천 장소 3~5곳을 선정하고 각각 다음을 정리하세요.
   · 언급 건수 · 인기 메뉴와 가격대 · 불만족 포인트(웨이팅, 주차, 서비스)
3. 불만족 언급이 데이터에 없으면 "언급 없음"이라고 쓰고, 없는 단점을 만들어내지 마세요.
""",
    MODES[1]: """
[분석 지침 — IT/기술]
1. 해당 기술·제품의 최신 동향과 장단점을 요약하세요.
2. 실무자·개발자 관점의 문제점(이슈, 한계, 호환성)을 중심으로 정리하세요.
3. 시점에 따라 평가가 달라진 부분이 있으면 날짜를 근거로 짚어주세요.
4. 마케팅 문구는 배제하고 검증 가능한 팩트 위주로 쓰세요.
""",
    MODES[2]: """
[분석 지침 — 여행/데이트]
1. 가볼 만한 코스, 명소, 숙소를 추천하고 반나절/하루 단위로 묶어 제안하세요.
2. 휴장 여부, 운영 시간, 주차 팁 등 실질적 방문 정보를 포함하세요.
3. 계절·시기별 주의사항이 언급되었다면 반영하세요.
""",
}


def build_prompt(query: str, mode: str, raw_data: str, custom: str, queries: list[str]) -> str:
    guide = SYSTEM_PROMPTS.get(mode, f"[사용자 특별 지침]\n{custom}")
    used = ", ".join(f"'{q}'" for q in queries)

    return f"""'{query}'에 대해 네이버 블로그를 검색한 결과입니다.
실제 사용된 검색어: {used}
각 항목은 제목과 요약문이며, 괄호 안은 작성 날짜입니다.
{COMMON_RULES}
{guide}

[수집 데이터]
{raw_data}
"""


# ---------------------------------------------------------------------------
# Gemini 호출
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def count_tokens(prompt: str, model_name: str) -> int:
    return genai.GenerativeModel(model_name).count_tokens(prompt).total_tokens


@st.cache_data(ttl=3600, show_spinner=False)
def generate(prompt: str, model_name: str) -> str:
    model = genai.GenerativeModel(
        model_name,
        generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS},
    )

    for attempt in range(MAX_RETRIES):
        try:
            resp = model.generate_content(prompt)

            if not resp.candidates:
                fb = getattr(resp, "prompt_feedback", None)
                return f"⚠️ 응답이 생성되지 않았습니다 (프롬프트 차단 가능성)\n\n```\n{fb}\n```"

            cand = resp.candidates[0]
            text = "".join(p.text for p in cand.content.parts if hasattr(p, "text"))

            if not text.strip():
                return (
                    "⚠️ 응답 본문이 비어 있습니다.\n\n"
                    f"- finish_reason: `{cand.finish_reason}`\n"
                    f"- usage: `{resp.usage_metadata}`\n\n"
                    "`MAX_TOKENS`라면 상단의 `MAX_OUTPUT_TOKENS`를 올리세요."
                )
            return text

        except gexc.ResourceExhausted:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_WAIT)

    raise RuntimeError("unreachable")


def analyze(query: str, mode: str, custom: str = ""):
    meta: dict = {}

    try:
        blogs, queries = collect_blogs(query, mode)
    except Exception as e:
        return None, f"❌ 네이버 검색 실패: {type(e).__name__}: {e}", meta

    if not blogs:
        return None, "⚠️ 검색 결과가 없습니다.", meta

    meta["count"] = len(blogs)
    meta["queries"] = queries

    try:
        model_name = resolve_model()
    except Exception as e:
        return None, f"❌ 모델 확인 실패: {e}", meta

    prompt = build_prompt(query, mode, format_blogs(blogs), custom, queries)

    try:
        meta["tokens"] = count_tokens(prompt, model_name)
    except Exception:
        meta["tokens"] = None

    try:
        return model_name, generate(prompt, model_name), meta
    except gexc.ResourceExhausted as e:
        detail = "\n".join(str(d) for d in (getattr(e, "details", []) or []))
        return model_name, f"❌ 쿼터/결제 오류\n\n```\n{e.message}\n\n{detail}\n```", meta
    except Exception as e:
        return model_name, f"❌ 분석 중 오류: {type(e).__name__}: {e}", meta


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="네이버 AI 분석기", page_icon="🔍")
st.title("🔍 네이버 다목적 AI 분석기")
st.caption("검색어를 여러 각도로 확장해 수집하고 Gemini로 분석합니다.")

with st.sidebar:
    st.subheader("⚙️ 상태")
    try:
        st.success(f"모델: `{resolve_model()}`")
    except Exception as e:
        st.error(str(e))

    with st.expander("사용 가능한 모델"):
        try:
            st.code("\n".join(list_text_models()))
        except Exception as e:
            st.error(f"조회 실패: {e}")

    if st.button("🗑️ 캐시 비우기"):
        st.cache_data.clear()
        st.success("캐시를 비웠습니다.")

mode = st.radio("어떤 목적으로 검색하시나요?", MODES)

custom_instruction = ""
if mode == MODES[3]:
    custom_instruction = st.text_area(
        "제미나이에게 내릴 분석 지시사항을 적어주세요.",
        placeholder="예: 가장 많이 언급되는 장점과 단점 3가지만 표로 정리해 줘.",
    )

query = st.text_input("검색어를 입력하세요", placeholder="예: 여의도 맛집 추천")

if query.strip():
    st.caption("확장 검색어: " + " · ".join(expand_queries(query, mode)))

if st.button("분석 시작하기", type="primary"):
    if not query.strip():
        st.warning("검색어를 입력해주세요.")
    else:
        with st.spinner(f"[{mode}] 모드로 수집·분석 중입니다... ⏳"):
            used_model, result, meta = analyze(query, mode, custom_instruction)

        st.markdown("### 📊 분석 결과")

        bits = []
        if used_model:
            bits.append(f"모델 `{used_model}`")
        if meta.get("count"):
            bits.append(f"수집 {meta['count']}건")
        if meta.get("tokens"):
            bits.append(f"입력 {meta['tokens']:,} 토큰")
        if bits:
            st.caption(" · ".join(bits))

        st.markdown(result)
