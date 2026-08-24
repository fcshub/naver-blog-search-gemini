"""
네이버 블로그 AI 분석기 v5 — 본문 크롤링 버전

추가 설치 필요:
    pip install beautifulsoup4 lxml
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import google.generativeai as genai
import requests
import streamlit as st
from bs4 import BeautifulSoup
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

MAX_OUTPUT_TOKENS = 16384
MAX_RETRIES = 2
RETRY_WAIT = 65

# 수집
SORT_PLAN = [("sim", 20), ("date", 30)]
MAX_TOTAL_ITEMS = 60       # 검색 결과 상한
NAVER_DELAY = 0.15

# 크롤링
CRAWL_TOP_N = 15           # 본문을 실제로 긁을 글 수
MAX_BODY_CHARS = 2500      # 글 1건당 본문 상한
CRAWL_WORKERS = 5          # 동시 요청 수 (너무 올리면 차단 위험)
CRAWL_TIMEOUT = 8

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# 협찬 판별용 키워드 (본문 전체를 볼 수 있게 되어 실제 탐지가 가능해짐)
AD_KEYWORDS = [
    "소정의", "원고료", "제공받아", "제공 받아", "협찬", "체험단",
    "무상으로", "지원받아", "지원 받아", "광고", "유료광고",
    "대가성", "서포터즈", "앰배서더",
]

MODES = [
    "🍽️ 맛집/핫플 탐색",
    "💻 IT/기술 동향 분석",
    "✈️ 여행/데이트 코스",
    "✏️ 내 맘대로 직접 지시",
]

QUERY_SUFFIXES = {
    MODES[0]: ["", "후기", "웨이팅", "가격"],
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
    available = list_text_models()
    for name in MODEL_PREFERENCES:
        if name in available:
            return name

    def version_of(n: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)", n)
        return float(m.group(1)) if m else 0.0

    flash = [n for n in available if "flash" in n.lower()]
    if flash:
        return max(flash, key=version_of)
    raise RuntimeError(f"flash 계열 모델 없음. 접근 가능: {available}")


# ---------------------------------------------------------------------------
# 네이버 검색
# ---------------------------------------------------------------------------
def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def _search_once(query: str, sort: str, display: int) -> list[dict]:
    url = "https://naverapihub.apigw.ntruss.com/search/v1/blog"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
    }
    res = requests.get(
        url, headers=headers,
        params={"query": query, "display": display, "sort": sort},
        timeout=10,
    )
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


# ---------------------------------------------------------------------------
# 본문 크롤링
# ---------------------------------------------------------------------------
def to_mobile_url(link: str) -> str | None:
    """blog.naver.com/{id}/{logNo} → m.blog.naver.com/{id}/{logNo}"""
    m = re.search(r"blog\.naver\.com/([^/?#]+)/(\d+)", link)
    if m:
        return f"https://m.blog.naver.com/{m.group(1)}/{m.group(2)}"

    # PostView.naver?blogId=xxx&logNo=123 형태
    m = re.search(r"blogId=([^&]+).*?logNo=(\d+)", link)
    if m:
        return f"https://m.blog.naver.com/{m.group(1)}/{m.group(2)}"
    return None


def clean_text(s: str) -> str:
    s = re.sub(r"[ \t\u00a0\u200b]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def fetch_body(link: str) -> str | None:
    """네이버 블로그 본문 텍스트 추출. 실패 시 None."""
    url = to_mobile_url(link)
    if not url:
        return None

    try:
        res = requests.get(
            url,
            headers={"User-Agent": UA, "Accept-Language": "ko-KR,ko;q=0.9"},
            timeout=CRAWL_TIMEOUT,
        )
        if res.status_code != 200:
            return None

        soup = BeautifulSoup(res.text, "lxml")

        # 에디터 버전별 컨테이너 후보
        node = (
            soup.select_one("div.se-main-container")      # SmartEditor ONE
            or soup.select_one("div#postViewArea")         # 구 에디터
            or soup.select_one("div.post_ct")
            or soup.select_one("div#viewTypeSelector")
        )
        if node is None:
            return None

        for tag in node.select("script, style"):
            tag.decompose()

        text = clean_text(node.get_text("\n"))
        return text if len(text) > 50 else None

    except requests.RequestException:
        return None
    except Exception:
        return None


def detect_ad(text: str) -> bool:
    head = text[:300]
    tail = text[-800:]
    zone = head + "\n" + tail
    return any(k in zone for k in AD_KEYWORDS)


# ---------------------------------------------------------------------------
# 수집 파이프라인
# ---------------------------------------------------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def collect(base_query: str, mode: str) -> tuple[list[dict], list[str], dict]:
    queries = expand_queries(base_query, mode)
    seen: set[str] = set()
    items: list[dict] = []

    for q in queries:
        for sort, display in SORT_PLAN:
            try:
                raw = _search_once(q, sort, display)
            except requests.RequestException:
                continue
            for it in raw:
                link = it.get("link", "")
                if not link or link in seen:
                    continue
                seen.add(link)
                items.append({
                    "title": _strip_tags(it.get("title", "")),
                    "desc": _strip_tags(it.get("description", "")),
                    "date": it.get("postdate", ""),
                    "link": link,
                    "body": None,
                    "is_ad": False,
                })
            time.sleep(NAVER_DELAY)

    items.sort(key=lambda b: b["date"], reverse=True)
    items = items[:MAX_TOTAL_ITEMS]

    # 상위 N건만 본문 크롤링 (병렬)
    targets = items[:CRAWL_TOP_N]
    with ThreadPoolExecutor(max_workers=CRAWL_WORKERS) as ex:
        futures = {ex.submit(fetch_body, it["link"]): it for it in targets}
        for fut in as_completed(futures):
            it = futures[fut]
            body = fut.result()
            if body:
                it["is_ad"] = detect_ad(body)
                it["body"] = body[:MAX_BODY_CHARS]

    stats = {
        "total": len(items),
        "crawled": sum(1 for i in items if i["body"]),
        "attempted": len(targets),
        "ads": sum(1 for i in items if i["is_ad"]),
    }
    return items, queries, stats


def format_items(items: list[dict]) -> str:
    blocks = []
    for i, b in enumerate(items, 1):
        d = b["date"]
        date_str = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else "날짜미상"

        if b["body"]:
            tag = " [협찬의심]" if b["is_ad"] else ""
            blocks.append(
                f"[{i}] ({date_str}){tag} {b['title']}\n"
                f"<본문>\n{b['body']}\n</본문>"
            )
        else:
            blocks.append(
                f"[{i}] ({date_str}) {b['title']}\n"
                f"<요약문만>\n{b['desc']}\n</요약문만>"
            )
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------
COMMON_RULES = """
[데이터 읽는 법]
- <본문> 태그가 있는 항목은 블로그 전문입니다. 가장 신뢰도 높은 근거로 삼으세요.
- <요약문만> 태그가 있는 항목은 검색 미리보기 두어 줄뿐입니다. 보조 근거로만 쓰고, 여기서 세부 정보를 끌어내려 하지 마세요.
- [협찬의심] 표시는 본문에서 대가성 문구가 발견된 글입니다. 버리지 말고 이렇게 쓰세요.
  · 사실 정보(메뉴, 가격, 위치, 영업시간, 주차)는 그대로 인용 가능
  · 평가·감상은 신뢰도를 낮춰 취급하고, 인용 시 협찬 여부를 밝힐 것

[공통 원칙]
- 데이터에 없는 내용은 절대 추측하거나 지어내지 마세요. 근거가 없으면 "정보 없음"이라고 쓰세요.
- 구체적 수치(가격, 대기시간, 운영시간)를 쓸 때는 몇 번 항목에서 나왔는지 [3] 형식으로 표기하세요.
- 여러 글에서 반복 언급된 내용을 우선하고, 몇 건에서 나왔는지 밝히세요.
- 마지막에 "⚠️ 데이터 한계"를 두고 확인하지 못한 부분을 2~3줄로 정리하세요.
"""

SYSTEM_PROMPTS = {
    MODES[0]: """
[분석 지침 — 맛집/핫플]
1. 작성 날짜를 확인하세요. 최근 12개월 내 언급이 없으면 폐업 가능성을 표시하세요.
2. 추천 장소 3~5곳을 선정하고 각각 정리하세요.
   · 언급 건수
   · 대표 메뉴와 실제 가격 (본문에 나온 숫자를 그대로, 출처 번호와 함께)
   · 웨이팅 실태 (대기 시간, 예약 가능 여부)
   · 주차 정보
   · 불만족 포인트 — 없으면 "언급 없음"이라고 쓰고 지어내지 말 것
3. 협찬 글에만 근거한 추천이라면 그 사실을 명시하세요.
""",
    MODES[1]: """
[분석 지침 — IT/기술]
1. 최신 동향과 장단점을 요약하세요.
2. 실무자·개발자 관점의 문제점(이슈, 한계, 호환성)을 구체적 사례와 함께 정리하세요.
3. 버전·시점에 따라 평가가 달라진 부분이 있으면 날짜를 근거로 짚으세요.
4. 벤치마크 수치나 설정값이 본문에 있으면 출처 번호와 함께 인용하세요.
""",
    MODES[2]: """
[분석 지침 — 여행/데이트]
1. 코스를 반나절/하루 단위로 묶어 제안하고 이동 동선을 설명하세요.
2. 입장료, 운영시간, 휴무일, 주차 요금을 본문에서 찾아 출처 번호와 함께 쓰세요.
3. 계절·시기별 주의사항, 혼잡도 언급이 있으면 반영하세요.
""",
}


def build_prompt(query: str, mode: str, data: str, custom: str,
                 queries: list[str], stats: dict) -> str:
    guide = SYSTEM_PROMPTS.get(mode, f"[사용자 특별 지침]\n{custom}")
    used = ", ".join(f"'{q}'" for q in queries)

    return f"""'{query}'에 대한 네이버 블로그 수집 결과입니다.
검색어: {used}
총 {stats['total']}건 중 {stats['crawled']}건은 본문 전문, 나머지는 요약문만 포함되어 있습니다.
{COMMON_RULES}
{guide}

[수집 데이터]
{data}
"""


def build_light_context(query: str, mode: str, result: str) -> str:
    return f"""'{query}'에 대해 네이버 블로그를 수집하고 [{mode}] 관점으로 분석한 결과입니다.
이어지는 질문에는 아래 내용을 근거로 답하세요.
여기 없는 내용은 추측하지 말고 "원본 데이터에 없어 답변할 수 없습니다"라고 하세요.

[1차 분석 결과]
{result}
"""


# ---------------------------------------------------------------------------
# Gemini 호출
# ---------------------------------------------------------------------------
def _extract(resp) -> str:
    if not resp.candidates:
        fb = getattr(resp, "prompt_feedback", None)
        return f"⚠️ 응답이 생성되지 않았습니다 (차단 가능성)\n\n```\n{fb}\n```"

    cand = resp.candidates[0]
    text = "".join(p.text for p in cand.content.parts if hasattr(p, "text"))

    if not text.strip():
        return (
            "⚠️ 응답 본문이 비어 있습니다.\n\n"
            f"- finish_reason: `{cand.finish_reason}`\n"
            f"- usage: `{resp.usage_metadata}`\n\n"
            "`MAX_TOKENS`라면 `MAX_OUTPUT_TOKENS`를 올리세요."
        )
    return text


def make_model(name: str):
    return genai.GenerativeModel(
        name, generation_config={"max_output_tokens": MAX_OUTPUT_TOKENS}
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
    chat = make_model(model_name).start_chat(history=history)
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
        items, queries, stats = collect(query, mode)
    except Exception as e:
        return None, f"❌ 수집 실패: {type(e).__name__}: {e}", meta, None

    if not items:
        return None, "⚠️ 검색 결과가 없습니다.", meta, None

    meta.update(stats)

    try:
        model_name = resolve_model()
    except Exception as e:
        return None, f"❌ 모델 확인 실패: {e}", meta, None

    prompt = build_prompt(query, mode, format_items(items), custom, queries, stats)
    meta["chars"] = len(prompt)

    try:
        return model_name, generate(prompt, model_name), meta, prompt
    except gexc.ResourceExhausted as e:
        detail = "\n".join(str(d) for d in (getattr(e, "details", []) or []))
        return model_name, f"❌ 쿼터/결제 오류\n\n```\n{e.message}\n\n{detail}\n```", meta, None
    except Exception as e:
        return model_name, f"❌ 분석 오류: {type(e).__name__}: {e}", meta, None


# ---------------------------------------------------------------------------
# 세션 상태
# ---------------------------------------------------------------------------
st.session_state.setdefault("ctx", None)
st.session_state.setdefault("turns", [])


def build_history(ctx: dict, carry_raw: bool) -> list[dict]:
    opener = (
        ctx["prompt"] if (carry_raw and ctx.get("prompt"))
        else build_light_context(ctx["query"], ctx["mode"], ctx["result"])
    )
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
st.caption("블로그 본문을 직접 수집해 분석하고, 결과에 대해 이어서 질문할 수 있습니다.")

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
        "후속 질문에 원본 포함", value=False,
        help="켜면 정확하지만 매 질문마다 본문 전체가 재전송되어 토큰 소모가 큽니다.",
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
        with st.spinner("블로그 본문을 수집하고 분석 중입니다... (30초~1분) ⏳"):
            model_name, result, meta, prompt = run_analysis(query, mode, custom_instruction)

        st.session_state.turns = []
        st.session_state.ctx = {
            "query": query, "mode": mode, "model": model_name,
            "result": result, "prompt": prompt, "meta": meta,
        }

ctx = st.session_state.ctx

if ctx:
    st.divider()
    st.markdown("### 📊 분석 결과")

    m = ctx["meta"]
    bits = []
    if ctx["model"]:
        bits.append(f"모델 `{ctx['model']}`")
    if m.get("total"):
        bits.append(f"검색 {m['total']}건")
    if m.get("attempted"):
        bits.append(f"본문 {m.get('crawled', 0)}/{m['attempted']}건")
    if m.get("ads"):
        bits.append(f"협찬의심 {m['ads']}건")
    if m.get("chars"):
        bits.append(f"{m['chars']:,}자")
    if bits:
        st.caption(" · ".join(bits))

    st.markdown(ctx["result"])

    for q, a in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    if ctx.get("prompt"):
        follow = st.chat_input("결과에 대해 이어서 질문해보세요")
        if follow:
            with st.chat_message("user"):
                st.markdown(follow)
            with st.chat_message("assistant"):
                with st.spinner("생각 중..."):
                    try:
                        answer = continue_chat(
                            build_history(ctx, carry_raw), follow, ctx["model"]
                        )
                    except gexc.ResourceExhausted as e:
                        answer = f"❌ 쿼터/결제 오류\n\n```\n{e.message}\n```"
                    except Exception as e:
                        answer = f"❌ 오류: {type(e).__name__}: {e}"
                st.markdown(answer)
            st.session_state.turns.append((follow, answer))
