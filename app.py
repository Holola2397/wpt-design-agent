import streamlit as st
import pandas as pd

# --- [System Setup] 페이지 및 상태 초기화 ---
st.set_page_config(page_title="Intelligent WPT Platform", layout="wide")

# 세션 상태(Session State) 초기화: 현재 스텝, 사용자 모드, 입력 데이터 저장용
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'project_data' not in st.session_state:
    st.session_state.project_data = {}

# 스텝 이동 함수
def go_to_step(step_num):
    st.session_state.step = step_num

def set_mode_and_next(mode):
    st.session_state.mode = mode
    st.session_state.step = 1

# --- [UI Layout] 프로그레스 바 (진행 상태 표시) ---
if st.session_state.step > 0:
    st.progress(st.session_state.step / 5.0, text=f"Step {st.session_state.step} / 5 진행 중...")
    st.divider()

# ==========================================
# [Phase 0] 진입 화면: 사용자 모드 선택
# ==========================================
if st.session_state.step == 0:
    st.title("⚡ 지능형 WPT 모듈 통합 설계 플랫폼")
    st.markdown("본 시스템은 무선전력전송 모듈의 요구사항 분석부터 파라미터 도출, 효율 시뮬레이션까지 원스톱으로 제공합니다.")
    
    st.write("### 설계 모드를 선택해 주십시오.")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("💡 **초심자 / 기획자용 (Auto Mode)**")
        st.write("어플리케이션과 물리적 제약조건만 입력하면, AI가 최적의 토폴로지와 초기 파라미터를 역산하여 추천합니다.")
        st.button("Auto Mode 시작하기", use_container_width=True, on_click=set_mode_and_next, args=('Auto',))
        
    with col2:
        st.warning("⚙️ **고급 설계자용 (Manual Mode)**")
        st.write("AI 추천값을 바탕으로 엔지니어가 직접 코일 스펙과 보상 소자를 미세 조정(Tuning)할 수 있습니다.")
        st.button("Manual Mode 시작하기", use_container_width=True, on_click=set_mode_and_next, args=('Manual',))

# ==========================================
# [Phase 1] 프로젝트 정의 및 제약 조건 입력
# ==========================================
elif st.session_state.step == 1:
    st.header("Step 1. 시스템 요구사항 및 제약 조건 입력")
    st.markdown("설계하고자 하는 어플리케이션의 물리적, 전기적 한계를 정의합니다.")
    
    with st.form("constraints_form"):
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🎯 어플리케이션 정보")
            app_type = st.selectbox("적용 분야", ["드론 (UAV)", "사족보행 로봇", "AGV/AMR", "전기차 (EV)", "모바일/가전"])
            battery_type = st.selectbox("배터리 종류", ["Li-ion (리튬이온)", "LiPo (리튬폴리머)", "LFP (리튬인산철)"])
            battery_vol = st.number_input("배터리 팩 공칭 전압 (V)", value=48.0, step=1.0)
            target_power = st.number_input("목표 충전 전력 (W)", value=300.0, step=50.0)
            
        with c2:
            st.subheader("📦 물리적 제약 조건")
            rx_weight_limit = st.number_input("수신부(Rx) 최대 허용 무게 (g)", value=400, step=50, help="초경량화가 필요한 경우 타이트하게 설정하세요.")
            tx_size = st.text_input("송신 패드(Tx) 가용 면적 (예: 200x200mm)", "200x200")
            rx_size = st.text_input("수신 패드(Rx) 가용 면적 (예: 100x100mm)", "100x100")
            air_gap = st.number_input("예상 이격 거리 (Air Gap, mm)", value=50, step=5)
            
        submitted = st.form_submit_button("입력 완료 및 다음 단계로 ➔")
        if submitted:
            # 입력 데이터 세션에 저장
            st.session_state.project_data = {
                "app_type": app_type, "battery_type": battery_type, "battery_vol": battery_vol,
                "target_power": target_power, "rx_weight_limit": rx_weight_limit,
                "tx_size": tx_size, "rx_size": rx_size, "air_gap": air_gap
            }
            go_to_step(2)
    
    st.button("⬅️ 처음으로 돌아가기", on_click=go_to_step, args=(0,))

# ==========================================
# [Phase 2] 지능형 추천 및 토폴로지 선정 (LLM 연동 예정 위치)
# ==========================================
elif st.session_state.step == 2:
    st.header("Step 2. AI 기반 토폴로지 및 스펙 추천")
    st.info("⏳ 앞서 입력하신 제약 조건을 바탕으로 최적의 설계를 분석 중입니다... (향후 LLM API 연동됨)")
    
    # 임시 목업(Mock-up) 결과 화면
    st.subheader(f"✅ 추천 토폴로지: **LCC-S (수신부 초경량화 구조)**")
    st.markdown(f"> **추천 사유 (AI 분석):**\n> 입력하신 **{st.session_state.project_data.get('app_type', '')}** 어플리케이션은 수신부 최대 허용 무게가 **{st.session_state.project_data.get('rx_weight_limit', '')}g**으로 매우 제한적입니다. 따라서 수신측 보상 인덕터가 생략되어 극단적인 경량화가 가능한 LCC-S 구조가 적합합니다.")
    
    st.divider()
    st.write("### AI 역산 제안 파라미터")
    col1, col2, col3 = st.columns(3)
    col1.metric("권장 입력 전압 (Vin)", "100 V")
    col2.metric("권장 스위칭 주파수 (f0)", "85 kHz")
    col3.metric("요구 상호 인덕턴스 (M)", "15.68 μH")
    
    st.write("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.button("⬅️ 이전 단계로", on_click=go_to_step, args=(1,), use_container_width=True)
    with c2:
        if st.session_state.mode == 'Auto':
            st.button("바로 시뮬레이션 결과 보기 ➔", on_click=go_to_step, args=(4,), type="primary", use_container_width=True)
        else:
            st.button("파라미터 상세 튜닝하기 (Manual) ➔", on_click=go_to_step, args=(3,), type="primary", use_container_width=True)

# ==========================================
# [Phase 3] 파라미터 상세 설계 (Manual Mode 전용)
# ==========================================
elif st.session_state.step == 3:
    st.header("Step 3. 코일 및 파라미터 상세 튜닝 (Expert)")
    st.markdown("초기 추천값을 바탕으로 코일 인덕턴스와 결합 계수를 미세 조정합니다.")
    
    # 기존 코드에 있던 좌측 슬라이더들을 이곳에 배치할 예정
    st.warning("🔧 (이곳에 기존의 Ltx, Lrx, k 슬라이더와 실시간 코일 전류 게이지가 배치됩니다.)")
    
    c1, c2 = st.columns(2)
    with c1:
        st.button("⬅️ 이전 단계로", on_click=go_to_step, args=(2,), use_container_width=True)
    with c2:
        st.button("시뮬레이션 및 최종 리포트 생성 ➔", on_click=go_to_step, args=(4,), type="primary", use_container_width=True)

# ==========================================
# [Phase 4 & 5] 최종 시뮬레이션 리포트 및 Export
# ==========================================
elif st.session_state.step == 4:
    st.header("Step 4 & 5. 최종 설계 리포트 및 데이터 내보내기")
    st.success("🎉 시스템 설계가 완료되었습니다.")
    
    # 탭으로 결과 화면 분리
    tab1, tab2, tab3 = st.tabs(["📊 효율 및 발열 분석", "⚡ 파라미터 및 내압", "📈 주파수 응답 시뮬레이션"])
    
    with tab1:
        st.write("(이곳에 기존의 효율 분석 막대 그래프와 발열량 수치가 렌더링됩니다.)")
    with tab2:
        st.write("(이곳에 Ltx, Lrx, Cp, Cs 등의 소자값 및 내압 가이드가 렌더링됩니다.)")
    with tab3:
        st.write("(이곳에 75kHz~95kHz 주파수 응답 Altair 동적 그래프가 렌더링됩니다.)")
        
    st.divider()
    
    st.subheader("📥 프로젝트 데이터 Export")
    st.markdown("입력하신 제약 조건과 최종 계산된 파라미터를 하나의 CSV 파일로 다운로드합니다.")
    
    # 다운로드 버튼 (임시)
    st.button("📊 사업계획서용 전체 설계 데이터 다운로드 (.csv)", type="primary")
    
    st.write("<br>", unsafe_allow_html=True)
    st.button("🔄 새로운 프로젝트 설계하기 (초기화)", on_click=go_to_step, args=(0,))
