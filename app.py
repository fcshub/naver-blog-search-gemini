"""
네이버 블로그 AI 분석기 v6 — 대량 수집 + 2단계 분석

추가 설치:
    pip install beautifulsoup4 lxml

동작 개요
  1) 확장 검색어로 최대 100건 수집
  2) 100건 전부 본문 크롤링 (병렬)
  3) 15건씩 묶어 각각 사실 추출  ← 1단계 (호출 여러 번)
  4) 추출 결과를 합쳐 최종 분석  ← 2단계 (호출 1번)
"""

import math
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
SORT_PLAN = [("sim", 40), ("date", 60)]
MAX_TOTAL_ITEMS = 100
NAVER_DELAY = 0.15

# 크롤링
CRAWL_TOP_N = 100
MAX_BODY_CHARS = 4000
CRAWL_WORKERS = 4
CRAWL_TIMEOUT = 8

# 2단계 분석
CHUNK_SIZE = 15          # 1단계에서 한 번에 처리할 글 수
STAGE1_MAX_TOKENS = 8192  # 사실 추출은 짧아도 됨

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

AD_KEYWORDS = [
    "소정의", "원고료", "제공받아", "제공 받아", "협찬", "체험단",
    "무상으로", "지원받아", "지원 받아", "유료광고", "대가성",
    "서포터즈", "앰배서더", "쿠팡 파트너스",
]

MODES = [
    "🍽️ 맛집/핫플 탐색",
    "💻 IT/기술 동향 분석",
    "✈️ 여행/데이트 코스",
    "✏️ 내 맘대로 직접 지시",
]

QUERY_SUFFIXES = {
    MODES[0]: ["", "후기", "웨이팅", "가격", "메뉴", "주차"],
    MODES[1]: ["", "후기", "단점", "문제", "비교", "설정"],
    MODES[2]: ["", "후기", "코스", "주차", "숙소", "입장료"],
    MODES[3]: ["", "후기", "정리"],
}


# ---------------------------------------------------------------------------
# 모델
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

    def ver(n: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)", n)
        return float(m.group(1)) if m else 0.0

    flash = [n for n in available if "flash" in n.lower()]
    if flash:
        return max(flash, key=ver)
    raise RuntimeError(f"flash 계열 모델 없음. 접근 가능: {available}")


def make_model(name: str, max_tokens: int = MAX_OUTPUT_TOKENS):
    return genai.GenerativeModel(
        name, generation_config={"max_output_tokens": max_tokens}
    )


# ---------------------------------------------------------------------------
# 네이버 검색
# ---------------------------------------------------------------------------
def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def _search_once(query: str, sort: str, display: int) -> list[dict]:
    res = requests.get(
        "https://naverapihub.apigw.ntruss.com/search/v1/blog",
        headers={
            "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
            "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET,
        },
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
# 크롤링
# ---------------------------------------------------------------------------
def to_mobile_url(link: str) -> str | None:
    m = re.search(r"blog\.naver\.com/([^/?#]+)/(\d+)", link)
    if m:
        return f"https://m.blog.naver.com/{m.group(1)}/{m.group(2)}"
    m = re.search(r"blogId=([^&]+).*?logNo=(\d+)", link)
    if m:
        return f"https://m.blog.naver.com/{m.group(1)}/{m.group(2)}"
    return None


def clean_text(s: str) -> str:
    s = re.sub(r"[ \t\u00a0\u200b]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def fetch_body(link: str) -> str | None:
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
        node = (
            soup.select_one("div.se-main-container")
            or soup.select_one("div#postViewArea")
            or soup.select_one("div.post_ct")
            or soup.select_one("div#viewTypeSelector")
        )
        if node is None:
            return None
        for tag in node.select("script, style"):
            tag.decompose()

        text = clean_text(node.get_text("\n"))
        return text if len(text) > 50 else None
    except Exception:
        return None


def detect_ad(text: str) -> bool:
    zone = text[:400] + "\n" + text[-1000:]
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
        "attempted": len(targets),
        "crawled": sum(1 for i in items if i["body"]),
        "ads": sum(1 for i in items if i["is_ad"]),
        "chars": sum(len(i["body"] or i["desc"]) for i in items),
    }
    return items, queries, stats


def format_chunk(items: list[dict], offset: int) -> str:
    blocks = []
    for i, b in enumerate(items, offset + 1):
        d = b["date"]
        date_str = f"{d[:4]}-{d[4:6]}-{d[6:]}" if len(d) == 8 else "날짜미상"
        if b["body"]:
            tag = " [협찬의심]" if b["is_ad"] else ""
            blocks.append(f"[{i}] ({date_str}){tag} {b['title']}\n<본문>\n{b['body']}\n</본문>")
        else:
            blocks.append(f"[{i}] ({date_str}) {b['title']}\n<요약문만>\n{b['desc']}\n</요약문만>")
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 프롬프트
# ---------------------------------------------------------------------------
STAGE1_INSTRUCTION = """당신은 자료 정리 담당입니다. 아직 최종 분석을 하지 마세요.
아래 블로그 글들에서 **사실 정보만** 추출해 구조화하세요.

[추출 규칙]
- 언급된 장소·제품·항목마다 다음을 정리하세요.
  · 이름
  · 구체적 수치 (가격, 대기시간, 운영시간, 입장료 등) — 본문에 나온 숫자를 그대로
  · 위치·접근 정보 (주소, 역, 주차)
  · 긍정 평가 요지
  · 부정 평가·불만 사항
  · 출처 번호와 작성 날짜
- [협찬의심] 표시가 있는 글에서 나온 평가는 반드시 "(협찬)"이라고 표기하세요.
- 본문에 없는 내용은 절대 만들지 마세요. 해당 항목이 없으면 생략하세요.
- 요약·의견·추천을 쓰지 말고, 사실만 나열하세요. 최종 판단은 다음 단계에서 합니다.
- 간결한 불릿 형식으로 쓰세요.
"""

STAGE2_RULES = """
[데이터 성격]
- 아래는 블로그 원문에서 1차로 추출한 사실 목록입니다. 여러 묶음으로 나뉘어 있으니 전체를 종합하세요.
- 대괄호 숫자는 원본 글 번호입니다. 수치를 인용할 때 함께 표기하세요.
- "(협찬)" 표시된 평가는 신뢰도를 낮춰 취급하되, 사실 정보는 그대로 써도 됩니다.

[공통 원칙]
- 여러 묶음에 걸쳐 반복 등장한 항목을 우선하고, 몇 건에서 나왔는지 밝히세요.
- 자료에 없는 내용은 추측하지 말고 "정보 없음"이라고 쓰세요.
- 수치가 서로 다르게 나오면 둘 다 제시하고 날짜를 근거로 최신 쪽에 무게를 두세요.
- 마지막에 "⚠️ 데이터 한계"를 두고 확인하지 못한 부분을 정리하세요.
"""

SYSTEM_PROMPTS = {
    MODES[0]: """
[최종 분석 — 맛집/핫플]
1. 추천 장소 5~7곳을 선정하고 각각 정리하세요.
   · 언급 건수 · 대표 메뉴와 실제 가격 · 웨이팅 실태 · 주차 · 불만족 포인트
2. 최근 12개월 내 언급이 없으면 폐업 가능성을 표시하세요.
3. 협찬 글에만 근거한 추천이면 그 사실을 명시하세요.
4. 마지막에 목적별 추천(가성비 / 분위기 / 웨이팅 없는 곳)을 짧게 덧붙이세요.
""",
    MODES[1]: """
[최종 분석 — IT/기술]
1. 최신 동향과 장단점을 종합하세요.
2. 실무자 관점의 문제점(이슈, 한계, 호환성)을 구체적 사례·수치와 함께 정리하세요.
3. 버전·시점에 따라 평가가 갈린 지점을 날짜 근거로 짚으세요.
4. 도입을 검토한다면 어떤 조건에서 적합하고 어떤 조건에서 부적합한지 정리하세요.
""",
    MODES[2]: """
[최종 분석 — 여행/데이트]
1. 코스를 반나절/하루 단위로 묶어 2~3안 제안하고 이동 동선을 설명하세요.
2. 입장료, 운영시간, 휴무일, 주차 요금을 출처 번호와 함께 명시하세요.
3. 계절·시기별 주의사항, 혼잡도를 반영하세요.
4. 예상 총 비용을 대략 계산해 제시하세요.
""",
}


def build_stage1_prompt(query: str, chunk_text: str, idx: int, total: int) -> str:
    return f"""'{query}' 관련 네이버 블로그 자료입니다. ({idx}/{total} 묶음)

{STAGE1_INSTRUCTION}

[원문]
{chunk_text}
"""


def build_stage2_prompt(query: str, mode: str, extracts: list[str],
                        custom: str, queries: list[str], stats: dict) -> str:
    guide = SYSTEM_PROMPTS.get(mode, f"[사용자 특별 지침]\n{custom}")
    used = ", ".join(f"'{q}'" for q in queries)
    joined = "\n\n".join(
        f"===== 묶음 {i} =====\n{t}" for i, t in enumerate(extracts, 1)
    )
    return f"""'{query}'에 대한 네이버 블로그 {stats['total']}건을 수집해
1차 추출한 사실 목록입니다. (본문 확보 {stats['crawled']}건)
검색어: {used}
{STAGE2_RULES}
{guide}

[1차 추출 자료]
{joined}
"""


def build_light_context(query: str, mode: str, result: str) -> str:
    return f"""'{query}'에 대해 네이버 블로그를 대량 수집하고 [{mode}] 관점으로 분석한 결과입니다.
이어지는 질문에는 아래 내용을 근거로 답하세요.
여기 없는 내용은 추측하지 말고 "원본 자료에 없어 답변할 수 없습니다"라고 하세요.

[분석 결과]
{result}
"""


# ---------------------------------------------------------------------------
# Gemini 호출
# ---------------------------------------------------------------------------
def _extract(resp) -> str:
    if not resp.candidates:
        fb = getattr(resp, "prompt_feedback", None)
        return f"⚠️ 응답 생성 실패 (차단 가능성)\n\n```\n{fb}\n```"
    cand = resp.candidates[0]
    text = "".join(p.text for p in cand.content.parts if hasattr(p, "text"))
    if not text.strip():
        return (
            "⚠️ 응답 본문이 비어 있습니다.\n"
            f"- finish_reason: `{cand.finish_reason}`\n"
            f"- usage: `{resp.usage_metadata}`"
        )
    return text


def call_model(prompt: str, model_name: str, max_tokens: int = MAX_OUTPUT_TOKENS) -> str:
    model = make_model(model_name, max_tokens)
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


# ---------------------------------------------------------------------------
# 2단계 분석 실행
# ---------------------------------------------------------------------------
def run_pipeline(query: str, mode: str, custom: str, progress, status):
    meta: dict = {}

    status.write("① 네이버 검색 및 본문 수집 중...")
    try:
        items, queries, stats = collect(query, mode)
    except Exception as e:
        return None, f"❌ 수집 실패: {type(e).__name__}: {e}", meta, None
    if not items:
        return None, "⚠️ 검색 결과가 없습니다.", meta, None

    meta.update(stats)
    progress.progress(0.25)

    try:
        model_name = resolve_model()
    except Exception as e:
        return None, f"❌ 모델 확인 실패: {e}", meta, None

    # --- 1단계: 묶음별 사실 추출 ---
    chunks = [items[i:i + CHUNK_SIZE] for i in range(0, len(items), CHUNK_SIZE)]
    n = len(chunks)
    extracts: list[str] = []
    failed = 0

    for i, chunk in enumerate(chunks, 1):
        status.write(f"② 자료 추출 중... ({i}/{n} 묶음)")
        offset = (i - 1) * CHUNK_SIZE
        p = build_stage1_prompt(query, format_chunk(chunk, offset), i, n)
        try:
            extracts.append(call_model(p, model_name, STAGE1_MAX_TOKENS))
        except gexc.ResourceExhausted as e:
            failed += 1
            if not extracts:   # 첫 묶음부터 실패하면 중단
                return model_name, f"❌ 쿼터/결제 오류\n\n```\n{e.message}\n```", meta, None
        except Exception:
            failed += 1
        progress.progress(0.25 + 0.55 * i / n)

    if not extracts:
        return model_name, "❌ 자료 추출에 모두 실패했습니다.", meta, None

    meta["chunks"] = n
    meta["chunk_failed"] = failed
    meta["api_calls"] = len(extracts) + 1

    # --- 2단계: 종합 분석 ---
    status.write("③ 종합 분석 중...")
    p2 = build_stage2_prompt(query, mode, extracts, custom, queries, stats)
    meta["stage2_chars"] = len(p2)

    try:
        result = call_model(p2, model_name)
    except gexc.ResourceExhausted as e:
        detail = "\n".join(str(d) for d in (getattr(e, "details", []) or []))
        return model_name, f"❌ 쿼터/결제 오류\n\n```\n{e.message}\n\n{detail}\n```", meta, None
    except Exception as e:
        return model_name, f"❌ 분석 오류: {type(e).__name__}: {e}", meta, None

    progress.progress(1.0)
    return model_name, result, meta, extracts


# ---------------------------------------------------------------------------
# 세션 상태
# ---------------------------------------------------------------------------
st.session_state.setdefault("ctx", None)
st.session_state.setdefault("turns", [])


def build_history(ctx: dict, carry_raw: bool) -> list[dict]:
    if carry_raw and ctx.get("extracts"):
        joined = "\n\n".join(
            f"===== 묶음 {i} =====\n{t}" for i, t in enumerate(ctx["extracts"], 1)
        )
        opener = (
            f"'{ctx['query']}'에 대한 블로그 수집 자료의 1차 추출 결과입니다.\n"
            f"이어지는 질문에 이 자료를 근거로 답하세요.\n\n{joined}"
        )
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
st.set_page_config(page_title="네이버 AI 분석기", page_icon="🔍", layout="centered")
st.title("🔍 네이버 다목적 AI 분석기")
st.caption("블로그 본문을 대량 수집해 2단계로 분석하고, 결과에 이어서 질문할 수 있습니다.")

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
    st.caption(
        f"수집 {MAX_TOTAL_ITEMS}건 · 본문 {MAX_BODY_CHARS:,}자\n\n"
        f"분석당 API 호출 약 {math.ceil(MAX_TOTAL_ITEMS / CHUNK_SIZE) + 1}회"
    )
    carry_raw = st.checkbox(
        "후속 질문에 추출 자료 포함", value=False,
        help="켜면 정확하지만 매 질문마다 토큰 소모가 큽니다.",
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
        progress = st.progress(0.0)
        status = st.empty()
        model_name, result, meta, extracts = run_pipeline(
            query, mode, custom_instruction, progress, status
        )
        progress.empty()
        status.empty()

        st.session_state.turns = []
        st.session_state.ctx = {
            "query": query, "mode": mode, "model": model_name,
            "result": result, "extracts": extracts, "meta": meta,
        }

ctx = st.session_state.ctx

if ctx:
    st.divider()
    st.markdown("### 📊 분석 결과")

    m = ctx["meta"]
    bits = []
    if ctx["model"]:
        bits.append(f"`{ctx['model']}`")
    if m.get("total"):
        bits.append(f"수집 {m['total']}건")
    if m.get("attempted"):
        bits.append(f"본문 {m.get('crawled', 0)}/{m['attempted']}")
    if m.get("ads"):
        bits.append(f"협찬의심 {m['ads']}")
    if m.get("api_calls"):
        bits.append(f"호출 {m['api_calls']}회")
    if bits:
        st.caption(" · ".join(bits))

    if m.get("chunk_failed"):
        st.warning(f"{m['chunk_failed']}개 묶음의 추출이 실패해 일부 자료가 빠졌습니다.")

    st.markdown(ctx["result"])

    if ctx.get("extracts"):
        with st.expander("🔎 1차 추출 자료 보기"):
            for i, t in enumerate(ctx["extracts"], 1):
                st.markdown(f"**묶음 {i}**")
                st.markdown(t)
                st.divider()

    for q, a in st.session_state.turns:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    if ctx.get("extracts"):
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
