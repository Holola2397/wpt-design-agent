import streamlit as st
import pandas as pd

# --- [System Setup] 페이지 설정 ---
st.set_page_config(page_title="WPT Design Platform", layout="wide", initial_sidebar_state="collapsed")

# --- [Custom CSS] 적응형(Adaptive) 스타일 적용 ---
# 색상을 강제하지 않고 Streamlit의 CSS 변수(var(--text-color) 등)를 활용하여 테마 전환에 완벽히 대응합니다.
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* 전체 폰트 적용 */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* 폼(Form) 카드 UI: 테마에 맞춰 배경색 자동 변경 */
    div[data-testid="stForm"] {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color, rgba(128, 128, 128, 0.2));
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    /* 입력창 모서리 둥글게 */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        border-radius: 8px !important;
    }
    
    /* 버튼 디자인 (애플 스타일 Primary Blue 고정) */
    .stButton>button {
        background-color: #0A84FF !important;
        color: white !important;
        border-radius: 10px !important;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0071E3 !important;
        transform: scale(1.02);
    }
    
    /* 프로그레스 바 색상 고정 */
    .stProgress > div > div > div > div {
        background-color: #0A84FF !important;
    }
    
    /* 헤더 텍스트 디자인 */
    h1, h2, h3, h4 {
        font-weight: 600 !important;
        letter-spacing: -0.5px;
    }
    </style>
""", unsafe_allow_html=True)

# --- [Session State] 상태 초기화 ---
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'project_data' not in st.session_state:
    st.session_state.project_data = {}

def go_to_step(step_num):
    st.session_state.step = step_num

def set_mode_and_next(mode):
    st.session_state.mode = mode
    st.session_state.step = 1

# 상단 진행바
if st.session_state.step > 0:
    st.progress(st.session_state.step / 5.0, text=f"Step {st.session_state.step} / 5 진행 중...")
    st.write("<br>", unsafe_allow_html=True)

# ==========================================
# [Phase 0] 진입 화면: 사용자 모드 선택
# ==========================================
if st.session_state.step == 0:
    st.markdown("<h1 style='text-align: center; font-size: 3rem;'>Intelligent WPT Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: var(--text-color); opacity: 0.7; font-size: 1.2rem; margin-bottom: 3rem;'>무선전력전송 모듈의 요구사항 분석부터 파라미터 도출까지, 원스톱 통합 설계 솔루션.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col2:
        st.info("💡 **Auto Mode (초심자 / 기획자용)**")
        st.write("제약조건만 입력하면, AI가 최적의 토폴로지와 초기 파라미터를 추천합니다.")
        st.button("Auto Mode 시작하기", use_container_width=True, on_click=set_mode_and_next, args=('Auto',))
        
    with col3:
        st.warning("⚙️ **Manual Mode (고급 설계자용)**")
        st.write("AI 추천값을 바탕으로 엔지니어가 직접 코일 스펙과 소자값을 미세 조정합니다.")
        st.button("Manual Mode 시작하기", use_container_width=True, on_click=set_mode_and_next, args=('Manual',))

# ==========================================
# [Phase 1] 프로젝트 정의 및 제약 조건 입력
# ==========================================
elif st.session_state.step == 1:
    st.header("Step 1. 시스템 요구사항 및 제약 조건 입력")
    st.caption("설계하고자 하는 어플리케이션의 물리적, 전기적 한계를 정의합니다.")
    
    with st.form("constraints_form"):
        st.subheader("🔋 어플리케이션 및 배터리 시스템")
        c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
        with c1:
            app_type = st.selectbox("어플리케이션 분야", ["드론 (UAV)", "사족보행 로봇", "AGV/AMR", "전기차 (EV)", "모바일/가전"])
            target_power = st.number_input("목표 충전 전력 (W)", value=300.0, step=50.0)
        with c2:
            battery_type = st.selectbox("배터리 셀 화학 조성", ["Li-ion (3.7V)", "LiPo (3.7V)", "LFP (3.2V)"])
        with c3:
            battery_cells = st.number_input("직렬 셀 구성 (S)", min_value=1, value=13, step=1)
        with c4:
            unit_v = 3.2 if "LFP" in battery_type else 3.7
            battery_vol = unit_v * battery_cells
            st.metric("팩 공칭 전압", f"{battery_vol:.1f} V")
        
        st.divider()
        
        st.subheader("📐 가용 공간 제약 (Dimensions)")
        st.caption("단위: mm (가로 x 세로 x 두께)")
        
        st.markdown("**1. 마그네틱 코일 패드부**")
        pad_c1, pad_c2 = st.columns(2)
        with pad_c1:
            st.caption("송신 패드 (Tx)")
            tx_cw, tx_cl, tx_ch = st.columns(3)
            tx_coil_w = tx_cw.number_input("가로 (W)", key='tcw', value=200, step=10)
            tx_coil_l = tx_cl.number_input("세로 (L)", key='tcl', value=200, step=10)
            tx_coil_h = tx_ch.number_input("두께 (H)", key='tch', value=10, step=1)
        with pad_c2:
            st.caption("수신 패드 (Rx)")
            rx_cw, rx_cl, rx_ch = st.columns(3)
            rx_coil_w = rx_cw.number_input("가로 (W)", key='rcw', value=100, step=10)
            rx_coil_l = rx_cl.number_input("세로 (L)", key='rcl', value=100, step=10)
            rx_coil_h = rx_ch.number_input("두께 (H)", key='rch', value=5, step=1)

        st.write("<br>", unsafe_allow_html=True)

        st.markdown("**2. 전력 회로부 (인버터/정류기 및 보상회로)**")
        pwr_c1, pwr_c2 = st.columns(2)
        with pwr_c1:
            st.caption("송신 회로보드 (Tx)")
            tx_pw, tx_pl, tx_ph = st.columns(3)
            tx_pwr_w = tx_pw.number_input("가로 (W)", key='tpw', value=150, step=10)
            tx_pwr_l = tx_pl.number_input("세로 (L)", key='tpl', value=100, step=10)
            tx_pwr_h = tx_ph.number_input("두께 (H)", key='tph', value=30, step=1)
        with pwr_c2:
            st.caption("수신 회로보드 (Rx)")
            rx_pw, rx_pl, rx_ph = st.columns(3)
            rx_pwr_w = rx_pw.number_input("가로 (W)", key='rpw', value=80, step=10)
            rx_pwr_l = rx_pl.number_input("세로 (L)", key='rpl', value=60, step=10)
            rx_pwr_h = rx_ph.number_input("두께 (H)", key='rph', value=15, step=1)

        st.divider()

        st.subheader("⚖️ 무게 및 환경 제약")
        w_c1, w_c2, w_c3 = st.columns(3)
        with w_c1:
            rx_weight_limit = st.number_input("수신부(Rx) 허용 총 무게 (g)", value=400, step=50, help="코일과 회로를 합친 모빌리티 탑재측 최대 무게입니다.")
        with w_c2:
            tx_weight_limit = st.number_input("송신부(Tx) 허용 총 무게 (kg)", value=5.0, step=1.0)
        with w_c3:
            air_gap = st.number_input("목표 이격 거리 (Air Gap, mm)", value=50, step=5)
            
        st.write("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("입력 완료 및 다음 단계로 ➔")
        if submitted:
            st.session_state.project_data = {
                "app_type": app_type,
                "battery": f"{battery_cells}S {battery_type} ({battery_vol:.1f}V)",
                "target_power": target_power,
                "rx_weight": rx_weight_limit,
                "air_gap": air_gap,
                "tx_coil_size": f"{tx_coil_w}x{tx_coil_l}x{tx_coil_h}",
                "rx_coil_size": f"{rx_coil_w}x{rx_coil_l}x{rx_coil_h}",
                "tx_pwr_size": f"{tx_pwr_w}x{tx_pwr_l}x{tx_pwr_h}",
                "rx_pwr_size": f"{rx_pwr_w}x{rx_pwr_l}x{rx_pwr_h}",
            }
            go_to_step(2)
            
    st.button("⬅️ 메인으로 돌아가기", on_click=go_to_step, args=(0,))

# ==========================================
# [Phase 2 이후] 임시 처리 (UI 흐름용)
# ==========================================
elif st.session_state.step == 2:
    st.header("Step 2. AI 기반 토폴로지 및 스펙 추천")
    st.info("⏳ 앞서 입력하신 제약 조건을 바탕으로 최적의 설계를 분석 중입니다... (LLM API 연동 대기 중)")
    
    saved_data = st.session_state.project_data
    
    st.subheader(f"✅ 추천 토폴로지: **LCC-S (수신부 초경량화 구조)**")
    st.markdown(f"> **AI 분석 코멘트:**\n> 입력하신 **{saved_data.get('app_type')}** 어플리케이션은 수신부 무게가 **{saved_data.get('rx_weight')}g** 이하로 제한되어 있으며, **{saved_data.get('tx_coil_size')}** 크기의 코일 공간 제약을 갖습니다. 배터리 공칭 전압이 높은 점을 감안할 때 수신부 보상 인덕터 생략이 가능한 LCC-S 구조가 절대적으로 유리합니다.")
    
    st.write("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.button("⬅️ 제약 조건 다시 입력하기", on_click=go_to_step, args=(1,), use_container_width=True)
    with c2:
        if st.session_state.mode == 'Auto':
            st.button("시뮬레이션 결과 보기 ➔", on_click=go_to_step, args=(4,), use_container_width=True)
        else:
            st.button("파라미터 상세 튜닝 (Manual) ➔", on_click=go_to_step, args=(3,), use_container_width=True)

elif st.session_state.step == 3:
    st.header("Step 3. 코일 및 파라미터 상세 튜닝 (Manual)")
    st.warning("🔧 백엔드 계산 모듈 연동 예정")
    c1, c2 = st.columns(2)
    with c1:
        st.button("⬅️ 이전 단계로", on_click=go_to_step, args=(2,), use_container_width=True)
    with c2:
        st.button("시뮬레이션 및 최종 리포트 생성 ➔", on_click=go_to_step, args=(4,), use_container_width=True)

elif st.session_state.step == 4:
    st.header("Step 4 & 5. 최종 설계 리포트 및 Export")
    st.success("🎉 설계 및 데이터 다운로드 준비 완료 (기존 계산 모듈 병합 예정)")
    st.button("🔄 새로운 프로젝트 설계하기 (초기화)", on_click=go_to_step, args=(0,))
