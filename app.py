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
# 위에서부터 순서대로 시도하고, 서버에 실제로 존재하는 첫 번째 모델을 사용합니다.
MODEL_PREFERENCES = [
    "gemini-3.7-flash",
    "gemini-2.5-flash",
]

MAX_DESC_CHARS = 2000    # 블로그 1건당 본문 상한 (네이버 원본이 짧아 사실상 미절단)
DISPLAY_PER_SORT = 50    # 정렬 방식별 수집 건수 (관련도 30 + 최신 30)
MAX_OUTPUT_TOKENS = 16384  # thinking 모델은 사고 토큰도 여기 포함되므로 넉넉히
MAX_RETRIES = 2          # 429 발생 시 재시도 횟수
RETRY_WAIT = 65          # TPM 윈도우(1분)가 지나야 의미가 있음


# ---------------------------------------------------------------------------
# 모델 선택
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def list_text_models() -> list[str]:
    """generateContent를 지원하는 모델 이름 목록. 1시간 캐싱."""
    return [
        m.name.removeprefix("models/")
        for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    ]


@st.cache_data(ttl=3600, show_spinner=False)
def resolve_model() -> str:
    """선호 목록과 실제 사용 가능 목록을 대조해 쓸 모델을 결정."""
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

    raise RuntimeError(
        f"사용 가능한 flash 계열 모델이 없습니다. 접근 가능 모델: {available}"
    )


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


@st.cache_data(ttl=600, show_spinner=False)
def fetch_naver_blogs(query: str) -> list[dict]:
    """관련도순 + 최신순을 함께 수집하고 링크 기준으로 중복 제거."""
    items: list[dict] = []
    for sort in ("sim", "date"):
        items.extend(_fetch_once(query, sort, DISPLAY_PER_SORT))

    seen: set[str] = set()
    result: list[dict] = []
    for it in items:
        link = it.get("link", "")
        if link in seen:
            continue
        seen.add(link)

        result.append({
            "title": _strip_tags(it.get("title", "")),
            "desc": _strip_tags(it.get("description", ""))[:MAX_DESC_CHARS],
            "date": it.get("postdate", ""),
            "blogger": _strip_tags(it.get("bloggername", "")),
        })
    return result


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
SYSTEM_PROMPTS = {
    "🍽️ 맛집/핫플 탐색": """
[필수 지침]
1. 각 항목의 작성 날짜를 반드시 참고하세요. 최근 6개월 내 언급이 없는 곳은 폐업 가능성이 있으므로 추천에서 제외하거나 주의 표시를 하세요.
2. '협찬', '소정의 원고료', '체험단' 등 마케팅 문구가 섞인 칭찬 일색의 글은 신뢰도를 낮춰 취급하세요.
3. 추천 장소 3~5곳을 선정하고, 각각 인기 메뉴와 실제 불만족 포인트(웨이팅, 주차, 서비스 등)를 함께 정리하세요.
4. 여러 글에서 반복적으로 언급된 곳을 우선하고, 몇 건에서 언급되었는지 밝히세요.
""",
    "💻 IT/기술 동향 분석": """
[필수 지침]
1. 해당 기술/제품의 최신 동향과 장단점을 심층적으로 요약하세요.
2. 실무자·개발자 관점에서 언급된 문제점(이슈, 한계점, 호환성)을 중심으로 분석하세요.
3. 마케팅적 요소는 배제하고 객관적 팩트 위주로 정리하세요.
4. 시점에 따라 평가가 달라진 부분이 있다면 날짜를 근거로 짚어주세요.
""",
    "✈️ 여행/데이트 코스": """
[필수 지침]
1. 최신 후기를 바탕으로 가볼 만한 코스, 명소, 숙소를 추천하세요.
2. 휴장 여부, 운영 시간 변동, 주차 팁 등 실질적 방문 정보를 포함하세요.
3. 동선을 고려해 반나절/하루 코스로 묶어 제안하세요.
4. 계절이나 시기에 따른 주의사항이 언급되었다면 반영하세요.
""",
}


def build_prompt(query: str, mode: str, raw_data: str, custom: str) -> str:
    system_prompt = SYSTEM_PROMPTS.get(mode, f"[사용자 특별 지침]\n{custom}")
    return f"""다음은 '{query}'에 대해 네이버 블로그에서 수집한 검색 결과입니다.
각 항목은 제목과 요약문이며, 괄호 안은 작성 날짜입니다.

{system_prompt}

[수집 데이터]
{raw_data}
"""


# ---------------------------------------------------------------------------
# Gemini 호출
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def count_tokens(prompt: str, model_name: str) -> int:
    """무료 호출이며 쿼터를 소모하지 않음."""
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
            text = "".join(
                p.text for p in cand.content.parts if hasattr(p, "text")
            )

            if not text.strip():
                return (
                    "⚠️ 응답 본문이 비어 있습니다.\n\n"
                    f"- finish_reason: `{cand.finish_reason}`\n"
                    f"- usage: `{resp.usage_metadata}`\n\n"
                    "`MAX_TOKENS`라면 코드 상단의 `MAX_OUTPUT_TOKENS`를 올리세요."
                )
            return text

        except gexc.ResourceExhausted:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_WAIT)

    raise RuntimeError("unreachable")


def analyze_naver_trend(query: str, mode: str, custom_instruction: str = ""):
    """returns (model_name, result_text, meta)"""
    meta: dict = {}

    try:
        blogs = fetch_naver_blogs(query)
    except requests.HTTPError as e:
        return None, f"❌ 네이버 API 오류 {e.response.status_code}\n\n{e.response.text}", meta
    except requests.RequestException as e:
        return None, f"❌ 네이버 API 요청 실패: {e}", meta

    if not blogs:
        return None, "⚠️ 검색 결과가 없습니다.", meta

    meta["count"] = len(blogs)

    try:
        model_name = resolve_model()
    except Exception as e:
        return None, f"❌ 모델 확인 실패: {e}", meta

    prompt = build_prompt(query, mode, format_blogs(blogs), custom_instruction)

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
st.caption("네이버 블로그를 관련도순·최신순으로 함께 수집해 Gemini로 분석합니다.")

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

mode = st.radio(
    "어떤 목적으로 검색하시나요?",
    ["🍽️ 맛집/핫플 탐색", "💻 IT/기술 동향 분석", "✈️ 여행/데이트 코스", "✏️ 내 맘대로 직접 지시"],
)

custom_instruction = ""
if mode == "✏️ 내 맘대로 직접 지시":
    custom_instruction = st.text_area(
        "제미나이에게 내릴 분석 지시사항을 적어주세요.",
        placeholder="예: 가장 많이 언급되는 장점과 단점 3가지만 표로 정리해 줘.",
    )

query = st.text_input("검색어를 입력하세요", placeholder="예: 여의도 맛집 추천")

if st.button("분석 시작하기", type="primary"):
    if not query.strip():
        st.warning("검색어를 입력해주세요.")
    else:
        with st.spinner(f"[{mode}] 모드로 분석 중입니다... ⏳"):
            used_model, result, meta = analyze_naver_trend(query, mode, custom_instruction)

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
