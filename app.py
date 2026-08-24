"""
네이버 블로그 AI 분석기 v8 (최종)

추가 설치:
    pip install streamlit requests google-generativeai beautifulsoup4 lxml

동작 개요
  1) 확장 검색어로 후보 수집
  2) 기간별 층화 샘플링 (최근 40% / 3개월~1년 40% / 1년 이상 20%)
  3) 선정된 글 전부 본문 크롤링 (병렬)
  4) CHUNK_SIZE씩 묶어 사실 추출   ← 1단계
  5) 추출 결과를 종합해 최종 분석  ← 2단계
"""

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

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
# 사이드바 기본 선택값 우선순위. 목록에 있는 첫 번째가 기본으로 잡힙니다.
MODEL_PREFERENCES = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.7-flash",
    "gemini-2.5-flash",
]

MAX_OUTPUT_TOKENS = 16384
STAGE1_MAX_TOKENS = 8192
MAX_RETRIES = 2
RETRY_WAIT = 65

SORT_PLAN = [("sim", 70), ("date", 30)]
MAX_TOTAL_ITEMS = 100
NAVER_DELAY = 0.15

STRATA_RATIO = {"recent": 0.4, "mid": 0.4, "old": 0.2}
RECENT_DAYS = 90
MID_DAYS = 365

MAX_BODY_CHARS = 4000
CRAWL_WORKERS = 4
CRAWL_TIMEOUT = 8

CHUNK_SIZE = 15

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
    names = [
        m.name.removeprefix("models/")
        for m in genai.list_models()
        if "generateContent" in m.supported_generation_methods
    ]
    # 실사용 가능성이 높은 순으로: flash → pro → 나머지
    def rank(n: str) -> tuple:
        low = n.lower()
        tier = 0 if "flash" in low else (1 if "pro" in low else 2)
        m = re.search(r"(\d+(?:\.\d+)?)", n)
        return (tier, -(float(m.group(1)) if m else 0.0), n)

    return sorted(names, key=rank)


def default_model_index(available: list[str]) -> int:
    for name in MODEL_PREFERENCES:
        if name in available:
            return available.index(name)
    for i, n in enumerate(available):
        if "flash" in n.lower():
            return i
    return 0


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
# 기간별 층화 샘플링
# ---------------------------------------------------------------------------
def age_days(item: dict, today: datetime) -> int:
    d = item.get("date", "")
    if len(d) != 8:
        return 99999
    try:
        return (today - datetime.strptime(d, "%Y%m%d")).days
    except ValueError:
        return 99999


def stratify(items: list[dict], total: int) -> tuple[list[dict], dict]:
    today = datetime.now()
    buckets: dict[str, list[dict]] = {"recent": [], "mid": [], "old": []}

    for b in items:
        a = age_days(b, today)
        if a <= RECENT_DAYS:
            buckets["recent"].append(b)
        elif a <= MID_DAYS:
            buckets["mid"].append(b)
        else:
            buckets["old"].append(b)

    quota = {k: int(total * r) for k, r in STRATA_RATIO.items()}
    picked: list[dict] = []
    chosen: set[str] = set()

    for k in ("recent", "mid", "old"):
        buckets[k].sort(key=lambda b: b["date"], reverse=True)
        for b in buckets[k][:quota[k]]:
            picked.append(b)
            chosen.add(b["link"])

    if len(picked) < total:
        rest = [b for b in items if b["link"] not in chosen]
        rest.sort(key=lambda b: b["date"], reverse=True)
        for b in rest[:total - len(picked)]:
            picked.append(b)
            chosen.add(b["link"])

    picked.sort(key=lambda b: b["date"], reverse=True)

    dist = {
        "recent": sum(1 for b in picked if age_days(b, today) <= RECENT_DAYS),
        "mid": sum(1 for b in picked if RECENT_DAYS < age_days(b, today) <= MID_DAYS),
        "old": sum(1 for b in picked if age_days(b, today) > MID_DAYS),
    }
    return picked, dist


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
    pool: list[dict] = []

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
                pool.append({
                    "title": _strip_tags(it.get("title", "")),
                    "desc": _strip_tags(it.get("description", "")),
                    "date": it.get("postdate", ""),
                    "link": link,
                    "body": None,
                    "is_ad": False,
                })
            time.sleep(NAVER_DELAY)

    pool_size = len(pool)
    items, dist = stratify(pool, MAX_TOTAL_ITEMS)

    with ThreadPoolExecutor(max_workers=CRAWL_WORKERS) as ex:
        futures = {ex.submit(fetch_body, it["link"]): it for it in items}
        for fut in as_completed(futures):
            it = futures[fut]
            body = fut.result()
            if body:
                it["is_ad"] = detect_ad(body)
                it["body"] = body[:MAX_BODY_CHARS]

    stats = {
        "pool": pool_size,
        "total": len(items),
        "crawled": sum(1 for i in items if i["body"]),
        "ads": sum(1 for i in items if i["is_ad"]),
        "dist": dist,
        "span": (
            f"{items[-1]['date'][:6]}~{items[0]['date'][:6]}"
            if items and len(items[0]["date"]) == 8 else "-"
        ),
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
STAGE1_INSTRUCTION = """당신은 자료 정리 담당입니다. 아직 최종 분석이나 추천을 하지 마세요.
아래 블로그 글들에서 **사실 정보만** 추출해 구조화하세요.

[추출 규칙]
- 언급된 장소·제품·항목마다 다음을 정리하세요.
  · 이름
  · 구체적 수치 (가격, 대기시간, 운영시간, 입장료 등) — 본문에 나온 숫자를 그대로
  · 위치·접근 정보 (주소, 역, 주차)
  · 긍정 평가 요지
  · 부정 평가·불만 사항
  · 출처 번호와 작성 날짜 — 반드시 함께 적을 것
- [협찬의심] 표시가 있는 글에서 나온 평가에는 "(협찬)"이라고 표기하세요.
- 본문에 없는 내용은 절대 만들지 마세요. 해당 항목이 없으면 생략하세요.
- 요약·의견·추천을 쓰지 말고 사실만 나열하세요. 판단은 다음 단계에서 합니다.
- 간결한 불릿 형식으로 쓰세요.
"""

STAGE2_RULES = """
[데이터 성격]
- 아래는 블로그 원문에서 1차로 추출한 사실 목록입니다. 여러 묶음으로 나뉘어 있으니 전체를 종합하세요.
- 대괄호 숫자는 원본 글 번호입니다. 수치를 인용할 때 함께 표기하세요.
- "(협찬)" 표시된 평가는 신뢰도를 낮춰 취급하되, 사실 정보는 그대로 써도 됩니다.

[시기 해석]
- 자료는 최근 글과 오래된 글이 의도적으로 섞여 있습니다. 각 항목의 날짜를 반드시 확인하세요.
- 같은 대상에 대한 평가나 가격이 시기별로 달라졌다면 그 변화를 짚어주세요.
- 오래된 정보와 최근 정보가 충돌하면 최근 쪽에 무게를 두되, 둘 다 제시하고 날짜를 밝히세요.
- 특정 계절·시기에만 해당하는 내용(계절 메뉴, 성수기 혼잡)은 그 조건을 명시하세요.
- 최근 12개월 내 언급이 전혀 없는 대상은 폐업·단종 가능성을 표시하세요.

[공통 원칙]
- 여러 묶음에 걸쳐 반복 등장한 항목을 우선하고, 몇 건에서 나왔는지 밝히세요.
- 자료에 없는 내용은 추측하지 말고 "정보 없음"이라고 쓰세요.
- 마지막에 "⚠️ 데이터 한계"를 두고 확인하지 못한 부분을 정리하세요.
"""

SYSTEM_PROMPTS = {
    MODES[0]: """
[최종 분석 — 맛집/핫플]
1. 추천 장소 5~7곳을 선정하고 각각 정리하세요.
   · 언급 건수와 언급 시기 범위
   · 대표 메뉴와 실제 가격 (가격이 시기별로 다르면 변동을 표시)
   · 웨이팅 실태 · 주차 · 불만족 포인트
2. 협찬 글에만 근거한 추천이면 그 사실을 명시하세요.
3. 마지막에 목적별 추천(가성비 / 분위기 / 웨이팅 없는 곳)을 짧게 덧붙이세요.
""",
    MODES[1]: """
[최종 분석 — IT/기술]
1. 최신 동향과 장단점을 종합하세요.
2. 실무자 관점의 문제점(이슈, 한계, 호환성)을 구체적 사례·수치와 함께 정리하세요.
3. 버전·시점에 따라 평가가 갈린 지점을 날짜 근거로 짚으세요.
4. 어떤 조건에서 적합하고 어떤 조건에서 부적합한지 정리하세요.
""",
    MODES[2]: """
[최종 분석 — 여행/데이트]
1. 코스를 반나절/하루 단위로 묶어 2~3안 제안하고 이동 동선을 설명하세요.
2. 입장료, 운영시간, 휴무일, 주차 요금을 출처 번호와 함께 명시하세요.
3. 계절·시기별 주의사항과 혼잡도를 반영하고, 어느 시기 자료인지 밝히세요.
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
    d = stats["dist"]
    joined = "\n\n".join(
        f"===== 묶음 {i} =====\n{t}" for i, t in enumerate(extracts, 1)
    )
    return f"""'{query}'에 대한 네이버 블로그 {stats['total']}건의 1차 추출 자료입니다.
(본문 확보 {stats['crawled']}건 · 수집 시기 {stats['span']})
시기 분포: 최근 3개월 {d['recent']}건 / 3개월~1년 {d['mid']}건 / 1년 이상 {d['old']}건
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
# 파이프라인
# ---------------------------------------------------------------------------
def run_pipeline(query: str, mode: str, custom: str, model_name: str, progress, status):
    meta: dict = {}

    status.write("① 네이버 검색 및 본문 수집 중...")
    try:
        items, queries, stats = collect(query, mode)
    except Exception as e:
        return f"❌ 수집 실패: {type(e).__name__}: {e}", meta, None
    if not items:
        return "⚠️ 검색 결과가 없습니다.", meta, None

    meta.update(stats)
    progress.progress(0.25)

    # 1단계
    chunks = [items[i:i + CHUNK_SIZE] for i in range(0, len(items), CHUNK_SIZE)]
    n = len(chunks)
    extracts: list[str] = []
    failed = 0

    for i, chunk in enumerate(chunks, 1):
        status.write(f"② 자료 추출 중... ({i}/{n} 묶음)")
        p = build_stage1_prompt(query, format_chunk(chunk, (i - 1) * CHUNK_SIZE), i, n)
        try:
            extracts.append(call_model(p, model_name, STAGE1_MAX_TOKENS))
        except gexc.ResourceExhausted as e:
            failed += 1
            if not extracts:
                return f"❌ 쿼터/결제 오류\n\n```\n{e.message}\n```", meta, None
        except Exception:
            failed += 1
        progress.progress(0.25 + 0.55 * i / n)

    if not extracts:
        return "❌ 자료 추출에 모두 실패했습니다.", meta, None

    meta["chunks"] = n
    meta["chunk_failed"] = failed
    meta["api_calls"] = len(extracts) + 1

    # 2단계
    status.write("③ 종합 분석 중...")
    p2 = build_stage2_prompt(query, mode, extracts, custom, queries, stats)
    meta["stage2_chars"] = len(p2)

    try:
        result = call_model(p2, model_name)
    except gexc.ResourceExhausted as e:
        detail = "\n".join(str(d) for d in (getattr(e, "details", []) or []))
        return f"❌ 쿼터/결제 오류\n\n```\n{e.message}\n\n{detail}\n```", meta, None
    except Exception as e:
        return f"❌ 분석 오류: {type(e).__name__}: {e}", meta, None

    progress.progress(1.0)
    return result, meta, extracts


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
st.set_page_config(page_title="네이버 AI 분석기", page_icon="🔍")
st.title("🔍 네이버 다목적 AI 분석기")
st.caption("여러 시기의 블로그 본문을 균형 있게 수집해 2단계로 분석합니다.")

with st.sidebar:
    st.subheader("🤖 모델")

    try:
        available = list_text_models()
    except Exception as e:
        available = []
        st.error(f"모델 목록 조회 실패: {e}")

    if available:
        selected_model = st.selectbox(
            "사용할 모델",
            options=available,
            index=default_model_index(available),
            help="Lite 계열은 RPD 여유가 크고, 상위 모델은 정확도가 높습니다.",
        )
        st.caption(f"전체 {len(available)}개 모델 사용 가능")
    else:
        selected_model = st.text_input("모델 이름 직접 입력", value=MODEL_PREFERENCES[0])

    if st.button("🔁 모델 목록 새로고침"):
        list_text_models.clear()
        st.rerun()

    st.divider()
    st.subheader("⚙️ 설정")
    st.caption(
        f"수집 {MAX_TOTAL_ITEMS}건 · 본문 {MAX_BODY_CHARS:,}자\n\n"
        f"시기 배분 최근 {int(STRATA_RATIO['recent']*100)}% / "
        f"중기 {int(STRATA_RATIO['mid']*100)}% / "
        f"장기 {int(STRATA_RATIO['old']*100)}%\n\n"
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
    elif not selected_model:
        st.warning("사용할 모델을 선택해주세요.")
    else:
        progress = st.progress(0.0)
        status = st.empty()
        result, meta, extracts = run_pipeline(
            query, mode, custom_instruction, selected_model, progress, status
        )
        progress.empty()
        status.empty()

        st.session_state.turns = []
        st.session_state.ctx = {
            "query": query, "mode": mode, "model": selected_model,
            "result": result, "extracts": extracts, "meta": meta,
        }

ctx = st.session_state.ctx

if ctx:
    st.divider()
    st.markdown("### 📊 분석 결과")

    m = ctx["meta"]
    bits = [f"`{ctx['model']}`"]
    if m.get("total"):
        bits.append(f"수집 {m['total']}/{m.get('pool', '?')}건")
    if m.get("crawled") is not None:
        bits.append(f"본문 {m['crawled']}건")
    if m.get("ads"):
        bits.append(f"협찬의심 {m['ads']}")
    if m.get("api_calls"):
        bits.append(f"호출 {m['api_calls']}회")
    st.caption(" · ".join(bits))

    if m.get("dist"):
        d = m["dist"]
        st.caption(
            f"시기 분포 — 최근 3개월 {d['recent']}건 · "
            f"3개월~1년 {d['mid']}건 · 1년 이상 {d['old']}건 "
            f"({m.get('span', '-')})"
        )

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
