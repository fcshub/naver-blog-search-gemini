import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="제미나이 진단기", page_icon="🛠️")
st.title("🛠️ 제미나이 API 연결 진단기")

# Secrets에서 키 불러오기
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    st.info(f"🔑 입력된 제미나이 키 앞자리: {GEMINI_API_KEY[:10]}...")
except Exception as e:
    st.error("❌ Secrets에서 GEMINI_API_KEY를 찾을 수 없습니다.")
    st.stop()

# 제미나이 서버에 연결 테스트
if st.button("진단 시작하기"):
    with st.spinner("제미나이 서버에 사용 가능한 모델 목록을 요청 중입니다..."):
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            
            # 내 키로 쓸 수 있는 모델 리스트 가져오기
            models = genai.list_models()
            available_models = [m.name for m in models if 'generateContent' in m.supported_generation_methods]
            
            if available_models:
                st.success("✅ 제미나이 API 인증 성공! 통신이 정상적입니다.")
                st.markdown("### 🤖 현재 내 키로 사용 가능한 모델 목록:")
                for name in available_models:
                    st.write(f"- `{name}`")
                
                st.info("👆 위 목록에 나오는 이름 중 하나를 복사해서 분석기 코드에 넣으면 완벽하게 작동합니다!")
            else:
                st.warning("⚠️ 인증은 되었으나, 이 키로 사용할 수 있는 텍스트 모델이 하나도 없습니다.")
                
        except Exception as e:
            st.error("❌ 제미나이 서버 통신 실패!")
            st.code(str(e))
