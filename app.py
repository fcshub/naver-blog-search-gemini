import streamlit as st
import requests
import google.generativeai as genai
import re

NAVER_CLIENT_ID = st.secrets["NAVER_CLIENT_ID"]
NAVER_CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

def analyze_naver_trend(query, mode, custom_instruction=""):
    # 1. 네이버 블로그 검색 API 호출 (API HUB 신규 규격)
    url = f"https://naverapihub.apigw.ntruss.com/search/v1/blog?query={query}&display=30&sort=sim"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return f"네이버 API 연결 오류: {response.status_code}\n(응답 내용: {response.text})"
        
    data = response.json()
    blog_texts = []
    for item in data.get('items', []):
        clean_title = re.sub(r'<[^>]+>', '', item['title'])
        clean_desc = re.sub(r'<[^>]+>', '', item['description'])
        blog_texts.append(f"- {clean_title} : {clean_desc}")
    
    raw_data = "\n".join(blog_texts)
    
    # 2. 제미나이 연결 및 자동 모델 탐색 (404 에러 원천 차단)
    genai.configure(api_key=GEMINI_API_KEY)
    
    # 내 키로 사용 가능한 모델 목록을 불러와서 자동으로 알맞은 모델 선택
    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    target_model_name = "gemini-1.5-flash" # 만약을 위한 기본값
    
    for m_name in available_models:
        if "1.5-flash" in m_name: # flash 모델이 있으면 최우선으로 사용
            target_model_name = m_name
            break
        elif "gemini-pro" in m_name: # 없으면 pro 모델 사용
            target_model_name = m_name
            
    model = genai.GenerativeModel(target_model_name)
    
    # 3. 선택한 모드에 따라 프롬프트 분기
    if mode == "🍽️ 맛집/핫플 탐색":
        system_prompt = """
        [필수 지침]
        1. 과거에는 유명했으나 최근 리뷰가 끊겨 폐업이 의심되는 곳은 절대 추천하지 마세요. (현재 영업 중인 곳만)
        2. '협찬', '소정의 원고료' 등 마케팅 문구가 섞인 칭찬 일색의 내용은 배제하세요.
        3. 추천 장소 2~3곳의 인기 메뉴와 방문자들의 실제 단점(웨이팅, 주차 등)을 솔직하게 요약하세요.
        """
    elif mode == "💻 IT/기술 동향 분석":
        system_prompt = """
        [필수 지침]
        1. 검색된 리뷰와 문서들을 바탕으로 해당 기술/제품의 최신 동향과 장단점을 요약하세요.
        2. 실무자나 개발자 관점에서 언급된 문제점(이슈, 한계점, 호환성)을 중심으로 분석하세요.
        3. 마케팅적인 요소는 배제하고 객관적이고 기술적인 팩트 위주로 정리해 주세요.
        """
    elif mode == "✈️ 여행/데이트 코스":
        system_prompt = """
        [필수 지침]
        1. 최신 후기를 바탕으로 가볼 만한 여행 코스, 명소, 숙소 등을 추천해 주세요.
        2. 휴장 여부, 운영 시간 변동, 해변/관광지 규정, 주차 팁 등 실질적인 방문 정보를 꼭 포함하세요.
        3. 커플 여행이나 데이트 관점에서 동선을 고려하여 장단점을 요약해 주세요.
        """
    else:
        system_prompt = f"[사용자 특별 지침]\n{custom_instruction}"
        
    prompt = f"""
    다음은 '{query}'에 대해 네이버 블로그에서 수집한 30개의 최신 원문 데이터입니다.
    이 데이터를 바탕으로 아래 지침에 따라 분석을 수행해 주세요.
    
    {system_prompt}
    
    [수집된 네이버 원문 데이터]
    {raw_data}
    """
    
    result = model.generate_content(prompt)
    return result.text

# --- 앱 화면(UI) 구성 ---
st.set_page_config(page_title="네이버 다목적 분석기", page_icon="🔍")
st.title("🔍 네이버 다목적 AI 분석기")
st.write("검색 목적에 맞춰 네이버 최신 글 30개를 똑똑하게 요약합니다.")

mode = st.radio(
    "어떤 목적으로 검색하시나요?", 
    ["🍽️ 맛집/핫플 탐색", "💻 IT/기술 동향 분석", "✈️ 여행/데이트 코스", "✏️ 내 맘대로 직접 지시"]
)

custom_instruction = ""
if mode == "✏️ 내 맘대로 직접 지시":
    custom_instruction = st.text_area(
        "제미나이에게 내릴 분석 지시사항을 적어주세요.", 
        "예: 최신 글들을 읽고, 가장 많이 언급되는 장점과 단점 3가지만 표로 정리해 줘."
    )

query = st.text_input("검색어를 입력하세요")

if st.button("분석 시작하기"):
    if query:
        with st.spinner(f"[{mode}] 모드로 네이버 데이터를 긁어와 분석 중입니다... ⏳ (약 10초)"):
            result = analyze_naver_trend(query, mode, custom_instruction)
            st.markdown("### 📊 분석 결과")
            st.info(result)
    else:
        st.warning("검색어를 입력해주세요.")
