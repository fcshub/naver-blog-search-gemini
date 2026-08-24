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
MAX_TOTAL_ITEMS = 80
MAX_OUTPUT_TOKENS = 16384
MAX_RETRIES = 2
RETRY_WAIT = 65
NAVER_DELAY = 0.15

SORT_PLAN = [("sim", 20), ("date", 40)]

MODES = [
    "🍽️ 맛집/핫플 탐색",
    "💻 IT/기술 동향 분석",
    "✈️ 여행/데이트 코스",
    "✏️ 내 맘대로 직접 지시",
]

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


def resolve_model() -> str:
    """캐시하지 않음 — MODEL_PREFERENCES 수정이 즉시 반영되도록."""
    available = list_text_models()

    for name in MODEL_PREFERENCES:
        if name in available:
            return name

    def version_of(name: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)", name)
        return float(m.group(1)) if m else 0.0

    flash = [n for n in available if "flash" in n.lower()]
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
    queries = expand_queries(base_query, mode)

    seen: set[str] = set()
    result: list[dict] = []

    for q in queries:
        for sort, display in SORT_PLAN:
            try:
                items = _fetch_once(q, sort, display)
            except requests.RequestException:
                continue

            for it in items:
                link = it.get("link", "")
                if not link or link in seen:
                    continue
                seen.add(link)
                result.append({
                    "title": _strip_tags(it.get("title", "")),
                    "desc": _strip_tags(it.get("description", ""))[:MAX_DESC_CHARS],
                    "date": it.get("postdate", ""),
                })
            time.sleep(NAVER_DELAY)

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
  · 평가·감상(맛있다, 분위기 좋다)은 신뢰도를 낮춰 취급하세요.
- 여러 항목에서 반복 언급된 내용을 우선하고, 몇 건에서 나왔는지 밝히세요.
- 마지막에 "⚠️ 데이터 한계" 항목을 두고 확인하지 못한 부분을 2~3줄로 정리하세요.
"""

SYSTEM_PROMPTS = {
    MODES[0]: """
[분석 지침 — 맛집/핫플]
1. 작성 날짜를 확인하세요. 최근 12개월 내 언급이 없는 곳은 폐업 가능성을 표시하세요.
2. 추천 장소 3~5곳을 선정하고 각각 정리하세요.
   · 언급 건수 · 인기 메뉴와 가격대 · 불만족 포인트(웨이팅, 주차, 서비스)
3. 불만족 언급이 없으면 "언급 없음"이라고 쓰고, 없는 단점을 만들지 마세요.
""",
    MODES[1]: """
[분석 지침 — IT/기술]
1. 해당 기술·제품의 최신 동향과 장단점을 요약하세요.
2. 실무자·개발자 관점의 문제점(이슈, 한계, 호환성)을 중심으로 정리하세요.
3. 시점에 따라 평가가 달라진 부분이 있으면 날짜를 근거로 짚어주세요.
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


def build_light_context(query: str, mode: str, first_result: str) -> str:
    """후속 질문용 경량 컨텍스트 — 원본 데이터 제외."""
    return f"""'{query}'에 대해 네이버 블로그를 검색하고 [{mode}] 관점으로 분석한 결과입니다.
아래는 그 분석 결과 전문입니다. 이어지는 질문에는 이 내용을 근거로 답하세요.
여기 없는 내용을 물으면 추측하지 말고 "원본 데이터에 없어 답변할 수 없습니다"라고 하세요.

[1차 분석 결과]
{first_result}
"""


# ---------------------------------------------------------------------------
# Gemini 호출
# ---------------------------------------------------------------------------
def _extract(resp) -> str:
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


def make_model(model_name: str):
    return genai.GenerativeModel(
        model_name,
        generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS},
    )


def generate(prompt: str, model_name: str) -> str:
    model = make_model(model_name)
    for attempt in range(MAX_RETRIES):
        try:
            return _extract(model.generate_content(prompt))
        except gexc.ResourceExhausted:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_WAIT)
    raise RuntimeError("unreachable")


def continue_chat(history: list[dict], question: str, model_name: str) -> str:
    """history: [{'role': 'user'|'model', 'parts': [str]}, ...]"""
    model = make_model(model_name)
    chat = model.start_chat(history=history)
    for attempt in range(MAX_RETRIES):
        try:
            return _extract(chat.send_message(question))
        except gexc.ResourceExhausted:
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(RETRY_WAIT)
    raise RuntimeError("unreachable")


def run_analysis(query: str, mode: str, custom: str = ""):
    meta: dict = {}

    try:
        blogs, queries = collect_blogs(query, mode)
    except Exception as e:
        return None, f"❌ 네이버 검색 실패: {type(e).__name__}: {e}", meta, None

    if not blogs:
        return None, "⚠️ 검색 결과가 없습니다.", meta, None

    meta["count"] = len(blogs)

    try:
        model_name = resolve_model()
    except Exception as e:
        return None, f"❌ 모델 확인 실패: {e}", meta, None

    prompt = build_prompt(query, mode, format_blogs(blogs), custom, queries)
    meta["chars"] = len(prompt)

    try:
        return model_name, generate(prompt, model_name), meta, prompt
    except gexc.ResourceExhausted as e:
        detail = "\n".join(str(d) for d in (getattr(e, "details", []) or []))
        return model_name, f"❌ 쿼터/결제 오류\n\n```\n{e.message}\n\n{detail}\n```", meta, None
    except Exception as e:
        return model_name, f"❌ 분석 중 오류: {type(e).__name__}: {e}", meta, None


# ---------------------------------------------------------------------------
# 세션 상태
# ---------------------------------------------------------------------------
if "ctx" not in st.session_state:
    st.session_state.ctx = None   # 분석 컨텍스트
if "turns" not in st.session_state:
    st.session_state.turns = []   # [(question, answer), ...]


def build_history(ctx: dict, carry_raw: bool) -> list[dict]:
    """1차 분석 + 이후 대화를 Gemini history 형식으로 조립."""
    if carry_raw and ctx.get("prompt"):
        opener = ctx["prompt"]
    else:
        opener = build_light_context(ctx["query"], ctx["mode"], ctx["result"])

    history = [
        {"role": "user", "parts": [opener]},
        {"role": "model", "parts": [ctx["result"]]},
    ]
    for q, a in st.session_state.turns:
        history.append({"role": "user", "parts": [q]})
        history.append({"role": "model", "parts": [a]})
    return history


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="네이버 AI 분석기", page_icon="🔍")
st.title("🔍 네이버 다목적 AI 분석기")
st.caption("검색어를 여러 각도로 확장해 수집하고, 결과에 대해 이어서 질문할 수 있습니다.")

with st.sidebar:
    st.subheader("⚙️ 상태")
    try:
        st.success(f"모델: `{resolve_model()}`")
    except Exception as e:
        st.error(str(e))

    with st.expander("사용 가능한 모델"):
        if st.button("목록 불러오기"):
            try:
                st.code("\n".join(list_text_models()))
            except Exception as e:
                st.error(f"조회 실패: {e}")

    st.divider()
    carry_raw = st.checkbox(
        "후속 질문에 원본 데이터 포함",
        value=False,
        help="켜면 정확하지만 매 질문마다 블로그 전체가 재전송되어 토큰 소모가 큽니다.",
    )

    if st.button("🗑️ 캐시 비우기"):
        st.cache_data.clear()
        st.success("캐시를 비웠습니다.")

    if st.button("🔄 대화 초기화"):
        st.session_state.ctx = None
        st.session_state.turns = []
        st.rerun()

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
            model_name, result, meta, prompt = run_analysis(query, mode, custom_instruction)

        st.session_state.turns = []
        st.session_state.ctx = {
            "query": query,
            "mode": mode,
            "model": model_name,
            "result": result,
            "prompt": prompt,
            "meta": meta,
        }

# --- 결과 및 대화 표시 (세션 상태 기반이라 리런에도 유지됨) ---
ctx = st.session_state.ctx

if ctx:
    st.divider()
    st.markdown("### 📊 분석 결과")

    bits = []
    if ctx["model"]:
        bits.append(f"모델 `{ctx['model']}`")
    if ctx["meta"].get("count"):
        bits.append(f"수집 {ctx['meta']['count']}건")
    if ctx["meta"].get("chars"):
        bits.append(f"프롬프트 {ctx['meta']['chars']:,}자")
    if bits:
        st.caption(" · ".join(bits))

    st.markdown(ctx["result"])

    for q, a in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    if ctx.get("prompt"):   # 분석이 성공한 경우에만 후속 질문 허용
        follow = st.chat_input("결과에 대해 이어서 질문해보세요")
        if follow:
            with st.chat_message("user"):
                st.markdown(follow)
            with st.chat_message("assistant"):
                with st.spinner("생각 중..."):
                    try:
                        history = build_history(ctx, carry_raw)
                        answer = continue_chat(history, follow, ctx["model"])
                    except gexc.ResourceExhausted as e:
                        answer = f"❌ 쿼터/결제 오류\n\n```\n{e.message}\n```"
                    except Exception as e:
                        answer = f"❌ 오류: {type(e).__name__}: {e}"
                st.markdown(answer)

            st.session_state.turns.append((follow, answer))
