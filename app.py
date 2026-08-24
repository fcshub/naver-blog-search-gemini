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
# 모델 설정
# ---------------------------------------------------------------------------
# 위에서부터 순서대로 시도하고, 서버에 실제로 존재하는 첫 번째 모델을 사용합니다.
# 새 버전이 나오면 이 리스트 맨 위에 이름만 추가하면 됩니다.
MODEL_PREFERENCES = [
    "gemini-3.7-flash",   # 사용자가 지정한 모델 (존재 여부 미확인)
    "gemini-2.5-flash",   # 2026-05 기준 무료 등급의 안정적인 기본값
]

MAX_DESC_CHARS = 500   # 블로그 1건당 본문 상한 (TPM 초과 방지)
MAX_RETRIES = 1         # 429 발생 시 재시도 횟수


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

    # 선호 목록이 전부 없으면 flash 계열 중 버전이 가장 높은 것으로 대체
    def version_of(name: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)", name)
        return float(m.group(1)) if m else 0.0

    flash = [n for n in available if "flash" in n.lower() and "lite" not in n.lower()]
    if flash:
        return max(flash, key=version_of)

    raise RuntimeError(
        "사용 가능한 flash 계열 모델을 찾지 못했습니다. "
        f"현재 API 키로 접근 가능한 모델: {available}"
    )


# ---------------------------------------------------------------------------
# 네이버 검색
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def fetch_naver_blogs(query: str) -> list[str]:
    url = "https://naverapihub.apigw.ntruss.com/search/v1/blog"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
    }
    params = {"query": query, "display": 20, "sort": "sim"}

    res = requests.get(url, headers=headers, params=params, timeout=10)
    res.raise_for_status()

    texts = []
    for item in res.json().get("items", []):
        title = re.sub(r"<[^>]+>", "", item["title"])
        desc = re.sub(r"<[^>]+>", "", item["description"])[:MAX_DESC_CHARS]
        texts.append(f"- {title} : {desc}")
    return texts


# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------
SYSTEM_PROMPTS = {
    "🍽️ 맛집/핫플 탐색": """
    [필수 지침]
    1. 과거에는 유명했으나 최근 리뷰가 끊겨 폐업이 의심되는 곳은 절대 추천하지 마세요. (현재 영업 중인 곳만)
    2. '협찬', '소정의 원고료' 등 마케팅 문구가 섞인 칭찬 일색의 내용은 철저히 배제하세요.
    3. 추천 장소 2~3곳의 인기 메뉴와 방문자들의 실제 불만족 포인트(웨이팅, 주차, 서비스 등)를 객관적으로 요약하세요.
    """,
    "💻 IT/기술 동향 분석": """
    [필수 지침]
    1. 검색된 리뷰와 문서들을 바탕으로 해당 기술/제품의 최신 동향과 장단점을 심층적으로 요약하세요.
    2. 실무자나 개발자 관점에서 언급된 문제점(이슈, 한계점, 호환성)을 중심으로 분석하세요.
    3. 마케팅적인 요소는 배제하고 객관적이고 논리적인 팩트 위주로 정리해 주세요.
    """,
    "✈️ 여행/데이트 코스": """
    [필수 지침]
    1. 최신 후기를 바탕으로 가볼 만한 여행 코스, 명소, 숙소 등을 추천해 주세요.
    2. 휴장 여부, 운영 시간 변동, 주차 팁 등 실질적인 방문 정보를 꼭 포함하세요.
    3. 동선을 고려하여 장단점을 요약해 주세요.
    """,
}


def build_prompt(query: str, mode: str, raw_data: str, custom: str) -> str:
    system_prompt = SYSTEM_PROMPTS.get(mode, f"[사용자 특별 지침]\n{custom}")
    return f"""다음은 '{query}'에 대해 네이버 블로그에서 수집한 최신 원문 데이터입니다.
이 데이터를 바탕으로 아래 지침에 따라 심층 분석을 수행해 주세요.

{system_prompt}

[수집된 네이버 원문 데이터]
{raw_data}
"""


# ---------------------------------------------------------------------------
# Gemini 호출 (429 백오프 + 결과 캐싱)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def generate(prompt: str, model_name: str) -> str:
    model = genai.GenerativeModel(
        model_name,
        generation_config={"max_output_tokens": 2048},
    )

    for attempt in range(MAX_RETRIES):
        try:
            return model.generate_content(prompt).text
        except gexc.ResourceExhausted:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(5 * (2 ** attempt))   # 5s → 10s → 20s
    raise RuntimeError("unreachable")


def analyze_naver_trend(query: str, mode: str, custom_instruction: str = ""):
    try:
        blogs = fetch_naver_blogs(query)
    except requests.HTTPError as e:
        return None, f"❌ 네이버 API 연결 오류: {e.response.status_code}\n\n{e.response.text}"
    except requests.RequestException as e:
        return None, f"❌ 네이버 API 요청 실패: {e}"

    if not blogs:
        return None, "⚠️ 검색 결과가 없습니다."

    try:
        model_name = resolve_model()
    except Exception as e:
        return None, f"❌ 모델 확인 실패: {e}"

    prompt = build_prompt(query, mode, "\n".join(blogs), custom_instruction)

    try:
        return model_name, generate(prompt, model_name)
    except gexc.ResourceExhausted as e:
        return model_name, (
            "⏳ **쿼터 한도(429)에 걸렸습니다.** 잠시 후 다시 시도해 주세요.\n\n"
            f"```\n{e.message}\n```"
        )
    except Exception as e:
        return model_name, f"❌ 제미나이 분석 중 오류: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="네이버 AI 분석기", page_icon="🔍")
st.title("🔍 네이버 다목적 AI 분석기")
st.caption("네이버 블로그 최신 글을 수집해 Gemini로 심층 분석합니다.")

with st.sidebar:
    st.subheader("⚙️ 모델 상태")
    try:
        st.success(f"사용 모델: `{resolve_model()}`")
    except Exception as e:
        st.error(str(e))

    with st.expander("사용 가능한 모델 전체 보기"):
        try:
            st.code("\n".join(list_text_models()))
        except Exception as e:
            st.error(f"모델 목록 조회 실패: {e}")

mode = st.radio(
    "어떤 목적으로 검색하시나요?",
    ["🍽️ 맛집/핫플 탐색", "💻 IT/기술 동향 분석", "✈️ 여행/데이트 코스", "✏️ 내 맘대로 직접 지시"],
)

custom_instruction = ""
if mode == "✏️ 내 맘대로 직접 지시":
    custom_instruction = st.text_area(
        "제미나이에게 내릴 분석 지시사항을 적어주세요.",
        placeholder="예: 최신 글들을 읽고, 가장 많이 언급되는 장점과 단점 3가지만 표로 정리해 줘.",
    )

query = st.text_input("검색어를 입력하세요", placeholder="예: 여의도 맛집 추천")

if st.button("분석 시작하기", type="primary"):
    if not query.strip():
        st.warning("검색어를 입력해주세요.")
    else:
        with st.spinner(f"[{mode}] 모드로 분석 중입니다... ⏳"):
            used_model, result = analyze_naver_trend(query, mode, custom_instruction)

        st.markdown("### 📊 분석 결과")
        if used_model:
            st.caption(f"사용된 모델: `{used_model}`")
        st.markdown(result)
