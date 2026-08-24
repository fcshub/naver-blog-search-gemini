import streamlit as st
import requests
import google.generativeai as genai
import re

NAVER_CLIENT_ID = st.secrets["NAVER_CLIENT_ID"]
NAVER_CLIENT_SECRET = st.secrets["NAVER_CLIENT_SECRET"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

def analyze_naver_trend(query, mode, custom_instruction=""):
    # 1. 네이버 블로그 검색 API
    url = f"https://naverapihub.apigw.ntruss.com/search/v1/blog?query={query}&display=20&sort=sim"
    headers = {
        "X-NCP-APIGW-API-KEY-ID": NAVER_CLIENT_ID,
        "X-NCP-APIGW-API-KEY": NAVER_CLIENT_SECRET
    }
    
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return f"❌ 네이버 API 연결 오류: {response.status_code}\n(응답 내용: {response.text})"
        
    data = response.json()
    blog_texts = []
    for item in data.get('items', []):
        clean_title = re.sub(r'<[^>]+>', '', item['title'])
        clean_desc = re.sub(r'<[^>]+>', '', item['description'])
        blog_texts.append(f"- {clean_title} : {clean_desc}")
    
    raw_data = "\n".join(blog_texts)
    
    if not raw_data:
        return "⚠️ 검색 결과가 없습니다."

    # 2. 제미나이 연결 및 최상위 모델 자동 탐색 로직 적용
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        # 구글 서버에서 현재 사용 가능한 전체 텍스트 모델 리스트 가져오기
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # 'pro'가 포함된 모델만 필터링
        pro_models = [name for name in available_models if 'pro' in name.lower()]
        
        # 정규식을 이용해 버전 숫자(예: 3.1) 추출 후 내림차순 정렬 (가장 높은 숫자가 1등)
        def extract_version(model_name):
            match = re.search(r'(\d+\.\d+)', model_name)
            return float(match.group(1)) if match else 0.0
            
        pro_models.sort(key=extract_version, reverse=True)
        
        # 가장 버전 숫자가 높은 모델을 자동으로 선택 (만약 못 찾으면 기본값 gemini-pro 할당)
        best_pro_model = pro_models[0] if pro_models else 'models/gemini-pro'
        
        model = genai.GenerativeModel(best_pro_model)
        
        if mode == "🍽️ 맛집/핫플 탐색":
            system_prompt = """
            [필수 지침]
            1. 과거에는 유명했으나 최근 리뷰가 끊겨 폐업이 의심되는 곳은 절대 추천하지 마세요. (현재 영업 중인 곳만)
            2. '협찬', '소정의 원고료' 등 마케팅 문구가 섞인 칭찬 일색의 내용은 철저히 배제하세요.
            3. 추천 장소 2~3곳의 인기 메뉴와 방문자들의 실제 불만족 포인트(웨이팅, 주차, 서비스 등)를 객관적으로 요약하세요.
            """
        elif mode == "💻 IT/기술 동향 분석":
            system_prompt = """
            [필수 지침]
            1. 검색된 리뷰와 문서들을 바탕으로 해당 기술/제품의 최신 동향과 장단점을 심층적으로 요약하세요.
            2. 실무자나 개발자 관점에서 언급된 문제점(이슈, 한계점, 호환성)을 중심으로 분석하세요.
            3. 마케팅적인 요소는 배제하고 객관적이고 논리적인 팩트 위주로 정리해 주세요.
            """
        elif mode == "✈️ 여행/데이트 코스":
            system_prompt = """
            [필수 지침]
            1. 최신 후기를 바탕으로 가볼 만한 여행 코스, 명소, 숙소 등을 추천해 주세요.
            2. 휴장 여부, 운영 시간 변동, 주차 팁 등 실질적인 방문 정보를 꼭 포함하세요.
            3. 동선을 고려하여 장단점을 요약해 주세요.
            """
        else:
            system_prompt = f"[사용자 특별 지침]\n{custom_instruction}"
            
        prompt = f"""
        다음은 '{query}'에 대해 네이버 블로그에서 수집한 최신 원문 데이터입니다.
        이 데이터를 바탕으로 아래 지침에 따라 심층 분석을 수행해 주세요.
        
        {system_prompt}
        
        [수집된 네이버 원문 데이터]
        {raw_data}
        """
        
        result = model.generate_content(prompt)
        # 화면에 어떤 모델이 자동 선택되었는지 함께 표시
        return f"*(사용된 AI 모델: {best_pro_model})*\n\n" + result.text

    except Exception as e:
        if "ResourceExhausted" in str(e) or "429" in str(e):
            return "⏳ **최상위 모델의 1분당 사용량(2회)을 초과했습니다.** 딱 60초만 숨을 고른 뒤 다시 버튼을 눌러주시면 정상 작동합니다."
        else:
            return f"❌ 제미나이 분석 중 오류가 발생했습니다: {str(e)}"

# --- 앱 화면(UI) 구성 ---
st.set_page_config(page_title="네이버 분석기 (Auto-Update)", page_icon="🔍")
st.title("🔍 네이버 다목적 AI 분석기 (최고 모델 자동 탐색)")
st.write("구글 서버를 스캔하여 현재 출시된 가장 똑똑한 최상위 버전 모델을 자동으로 잡아내어 분석합니다.")

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

query = st.text_input("검색어를 입력하세요 (예: 여의도 맛집 추천)")

if st.button("최고 품질 분석 시작하기"):
    if query:
        with st.spinner(f"[{mode}] 모드로 최상위 모델을 탐색 후 심층 분석 중입니다... ⏳"):
            result = analyze_naver_trend(query, mode, custom_instruction)
            st.markdown("### 📊 분석 결과")
            st.info(result)
    else:
        st.warning("검색어를 입력해주세요.")
