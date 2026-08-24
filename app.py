import streamlit as st
import requests
import google.generativeai as genai
import re

# 클라우드 환경에서 비밀 열쇠를 안전하게 불러오는 코드
NAVER_CLIENT_ID = st.secrets["NAVER_CLIENT_ID"]
NAVER_CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

def analyze_naver_trend(query):
    # 최근 1~2년 누적 정확도순(sim)으로 30개의 글 수집
    url = f"https://openapi.naver.com/v1/search/blog.json?query={query}&display=30&sort=sim"
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return f"네이버 API 연결 오류: {response.status_code}"
        
    data = response.json()
    
    blog_texts = []
    for item in data.get('items', []):
        clean_title = re.sub(r'<[^>]+>', '', item['title'])
        clean_desc = re.sub(r'<[^>]+>', '', item['description'])
        blog_texts.append(f"- {clean_title} : {clean_desc}")
    
    raw_data = "\n".join(blog_texts)
    
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # 1~2년 누적 데이터 분석 및 영업 여부 교차 검증을 강력하게 지시하는 프롬프트
    prompt = f"""
    다음은 '{query}'에 대해 네이버 블로그에서 정확도순으로 수집한 30개의 원문 데이터입니다.
    이 데이터를 바탕으로 최근 1~2년간 꾸준히 사랑받은 지역 트렌드와 핫플레이스를 분석해 주세요.
    
    [필수 지침]
    1. 과거에는 유명했으나 최근 리뷰가 완전히 끊겨 영업 중단(폐업)이 의심되는 곳은 절대 추천하지 마세요. 모처럼 시간을 내어 방문했을 때 헛걸음하는 일이 없도록, 현재 확실히 운영 중인 곳만 추려야 합니다.
    2. '소정의 원고료', '협찬' 등의 마케팅 문구가 들어간 칭찬 일색의 내용은 배제하세요.
    3. 오랜 기간 검증된 진짜 핫플레이스 2~3곳을 선정하고, 인기 메뉴와 방문자들의 불만/주의사항(웨이팅, 주차, 동선 등)을 솔직하게 요약해 주세요.
    
    [수집된 네이버 원문 데이터]
    {raw_data}
    """
    
    result = model.generate_content(prompt)
    return result.text

# --- 앱 화면 디자인 부분 ---
st.set_page_config(page_title="진짜 맛집/트렌드 분석기", page_icon="🕵️‍♂️")
st.title("🕵️‍♂️ 네이버 찐 핫플 탐지기")
st.write("광고를 거르고 최근 1~2년간 꾸준히 검증된 진짜 트렌드만 분석합니다.")

query = st.text_input("검색어를 입력하세요 (예: 강원도 해변 근처 맛집)")

if st.button("분석 시작하기"):
    if query:
        with st.spinner("네이버 블로그 30개를 긁어와 꼼꼼히 분석 중입니다... ⏳ (약 10초)"):
            result = analyze_naver_trend(query)
            st.markdown("### 📊 분석 결과")
            st.info(result)
    else:
        st.warning("검색어를 입력해주세요.")
