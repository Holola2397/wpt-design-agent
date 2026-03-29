import streamlit as st
import numpy as np
import math
import pandas as pd
import altair as alt
import google.generativeai as genai
import json

# ==========================================
# [System Setup & Custom CSS]
# ==========================================
st.set_page_config(page_title="Intelligent WPT Platform", layout="wide", initial_sidebar_state="expanded")

# 적응형(Adaptive) 스타일 적용
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .card-container {
        background-color: var(--secondary-background-color);
        border: 1px solid var(--border-color, rgba(128, 128, 128, 0.2));
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 24px;
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select { border-radius: 8px !important; }
    .stButton>button {
        background-color: #0A84FF !important; color: white !important;
        border-radius: 10px !important; border: none; padding: 0.6rem 1.2rem; font-weight: 600; transition: all 0.2s ease;
    }
    .stButton>button:hover { background-color: #0071E3 !important; transform: scale(1.02); }
    .stProgress > div > div > div > div { background-color: #0A84FF !important; }
    h1, h2, h3, h4 { font-weight: 600 !important; letter-spacing: -0.5px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# [Session State Initialization]
# ==========================================
if 'step' not in st.session_state: st.session_state.step = 0
if 'mode' not in st.session_state: st.session_state.mode = None
if 'project_data' not in st.session_state: st.session_state.project_data = {}
if 'llm_result' not in st.session_state: st.session_state.llm_result = None
if 'tuning_data' not in st.session_state: st.session_state.tuning_data = {}

def go_to_step(step_num): st.session_state.step = step_num
def set_mode_and_next(mode):
    st.session_state.mode = mode
    st.session_state.step = 1
    st.session_state.llm_result = None

# ==========================================
# [Math Engine] 코일 역산 및 회로 시뮬레이션
# ==========================================
def estimate_coil_params(M_req, air_gap_mm, rx_weight_g):
    """제약조건 기반 Ltx, Lrx, k 자동 분배 알고리즘"""
    # 1. k 추정 (경험적 지수 감쇠 모델 활용)
    k_est = max(0.05, 0.4 * math.exp(-air_gap_mm / 60.0))
    
    # 2. Lrx 상한선 설정 (무게 제약 고려, 최대 80uH 캡)
    Lrx_max_possible = min(80.0, max(10.0, rx_weight_g / 5.0))
    Lrx_target = Lrx_max_possible * 0.8 # 20% 마진
    
    # 3. Ltx 역산
    Ltx_req = ((M_req * 1e-6) / k_est)**2 / (Lrx_target * 1e-6)
    
    # 물리적 한계 보정
    if Ltx_req > 300e-6:
        Ltx_req = 300e-6
        Lrx_target = ((M_req * 1e-6) / k_est)**2 / Ltx_req
        
    return {
        "k": round(k_est, 3),
        "Lrx": round(Lrx_target, 1),
        "Ltx": round(Ltx_req * 1e6, 1)
    }

def calculate_lccs(Vin, Vout, Pout, f0, Ltx, Lrx, k, Rtx, Rrx):
    w = 2 * math.pi * f0
    Vin_ac_rms = 2 * math.sqrt(2) / math.pi * Vin
    Iout = Pout / Vout
    RL = Vout / Iout
    RLeq = RL * (math.pi**2) / 8
    
    Vrect_in = Vout * math.pi / (2 * math.sqrt(2))
    Iout_ac = Iout * (2 * math.sqrt(2)) / math.pi
    
    M = k * math.sqrt(Ltx * Lrx)
    Itx = Vrect_in / (w * M)
    Irx = Iout_ac
    
    Ls = Vin_ac_rms / (w * Itx)
    if Ls >= Ltx: return {"error": "Ls가 Ltx보다 큽니다. Ltx를 키우거나 입력 전압을 높이세요."}
        
    Cp = 1 / (w**2 * Ls)
    Cs = 1 / (w**2 * (Ltx - Ls))
    Crx = 1 / (w**2 * Lrx)
    Ldc_min = RL / (3 * w)
    
    V_Ltx_peak = math.sqrt(2) * w * Ltx * Itx
    V_Cp_peak = math.sqrt(2) * w * Ls * Itx
    V_Cs_peak = math.sqrt(2) * Itx / (w * Cs)
    V_Crx_peak = math.sqrt(2) * Irx / (w * Crx)

    P_out_ac = (Irx**2) * RLeq
    P_loss_tx = (Itx**2) * Rtx
    P_loss_rx = (Irx**2) * Rrx
    P_in_ac = P_out_ac + P_loss_tx + P_loss_rx
    efficiency = (P_out_ac / P_in_ac) * 100 if P_in_ac > 0 else 0
    
    return {"Itx": Itx, "Irx": Irx, "M": M, "Ls": Ls, "Cp": Cp, "Cs": Cs, "Crx": Crx, "Ldc_min": Ldc_min,
            "V_Ltx_peak": V_Ltx_peak, "V_Cp_peak": V_Cp_peak, "V_Cs_peak": V_Cs_peak, "V_Crx_peak": V_Crx_peak,
            "P_out_ac": P_out_ac, "P_loss_tx": P_loss_tx, "P_loss_rx": P_loss_rx, "efficiency": efficiency,
            "RLeq": RLeq, "Vin_ac": Vin_ac_rms}

def calculate_double_lcc(Vin, Vout, Pout, f0, Ltx, Lrx, k, current_ratio, Rtx, Rrx):
    w = 2 * math.pi * f0
    Vin_ac_rms = 2 * math.sqrt(2) / math.pi * Vin
    Iout = Pout / Vout
    RL = Vout / Iout
    RLeq = RL * 8 / (math.pi**2)
    
    Vrect_in = 2 * math.sqrt(2) / math.pi * Vout
    Iout_ac = Iout * math.sqrt(2) * math.pi / 4
    
    M = k * math.sqrt(Ltx * Lrx)
    L_product = (M * Vin_ac_rms) / (w * Iout_ac)
    L_ratio = current_ratio * (Vrect_in / Vin_ac_rms)
    
    Llcc_tx = math.sqrt(L_product / L_ratio)
    Llcc_rx = math.sqrt(L_product * L_ratio)
    
    if Llcc_tx >= Ltx or Llcc_rx >= Lrx: return {"error": "보상 인덕터(Llcc)가 메인 코일보다 큽니다."}
        
    Clcc_tx = 1 / (w**2 * Llcc_tx); Clcc_rx = 1 / (w**2 * Llcc_rx)
    Cp_tx = 1 / (w**2 * (Ltx - Llcc_tx)); Cp_rx = 1 / (w**2 * (Lrx - Llcc_rx))
    
    Itx = Vin_ac_rms / (w * Llcc_tx)
    Irx = Vrect_in / (w * Llcc_rx)

    V_Ltx_peak = math.sqrt(2) * w * Ltx * Itx; V_parallel_tx_peak = math.sqrt(2) * w * Llcc_tx * Itx; V_series_tx_peak = math.sqrt(2) * Itx / (w * Cp_tx)
    V_Lrx_peak = math.sqrt(2) * w * Lrx * Irx; V_parallel_rx_peak = math.sqrt(2) * w * Llcc_rx * Irx; V_series_rx_peak = math.sqrt(2) * Irx / (w * Cp_rx)

    P_out_ac = (Irx**2) * RLeq
    P_loss_tx = (Itx**2) * Rtx; P_loss_rx = (Irx**2) * Rrx
    P_in_ac = P_out_ac + P_loss_tx + P_loss_rx
    efficiency = (P_out_ac / P_in_ac) * 100 if P_in_ac > 0 else 0
    
    return {"Itx": Itx, "Irx": Irx, "M": M, "Llcc_tx": Llcc_tx, "Llcc_rx": Llcc_rx, "Clcc_tx": Clcc_tx, "Clcc_rx": Clcc_rx, "Cp_tx": Cp_tx, "Cp_rx": Cp_rx,
            "V_Ltx_peak": V_Ltx_peak, "V_parallel_tx_peak": V_parallel_tx_peak, "V_series_tx_peak": V_series_tx_peak,
            "V_Lrx_peak": V_Lrx_peak, "V_parallel_rx_peak": V_parallel_rx_peak, "V_series_rx_peak": V_series_rx_peak,
            "P_out_ac": P_out_ac, "P_loss_tx": P_loss_tx, "P_loss_rx": P_loss_rx, "efficiency": efficiency, "RLeq": RLeq, "Vin_ac": Vin_ac_rms}

def simulate_frequency_response(topology, res_dict, f0, Ltx, Lrx, M, Rtx, Rrx):
    f_arr = np.linspace(f0 - 10e3, f0 + 10e3, 200)
    w_arr = 2 * np.pi * f_arr
    Vin_ac, RLeq = res_dict["Vin_ac"], res_dict["RLeq"]
    
    if topology == "LCC-S (수신부 초경량화)":
        Ls, Cp, Cs, Crx = res_dict['Ls'], res_dict['Cp'], res_dict['Cs'], res_dict['Crx']
        Z_rx = RLeq + Rrx + 1j*w_arr*Lrx + 1/(1j*w_arr*Crx)
        Z_refl = (w_arr * M)**2 / Z_rx
        Z_tx_branch = Rtx + 1j*w_arr*Ltx + 1/(1j*w_arr*Cs) + Z_refl
        Z_p = (1/(1j*w_arr*Cp)) * Z_tx_branch / ((1/(1j*w_arr*Cp)) + Z_tx_branch)
        Z_in = 1j*w_arr*Ls + Z_p
        I_in = Vin_ac / Z_in
        I_tx = (I_in * Z_p) / Z_tx_branch
        I_rx = (1j * w_arr * M * I_tx) / Z_rx
        P_out_arr = (np.abs(I_rx)**2) * RLeq
        P_in_arr = np.real(Vin_ac * np.conj(I_in))
    else: 
        Llcc_tx, Clcc_tx, Cp_tx = res_dict['Llcc_tx'], res_dict['Clcc_tx'], res_dict['Cp_tx']
        Llcc_rx, Clcc_rx, Cp_rx = res_dict['Llcc_rx'], res_dict['Clcc_rx'], res_dict['Cp_rx']
        Z_load_branch = RLeq + 1j*w_arr*Llcc_rx + 1/(1j*w_arr*Clcc_rx)
        Z_p_rx = (1/(1j*w_arr*Cp_rx)) * Z_load_branch / ((1/(1j*w_arr*Cp_rx)) + Z_load_branch)
        Z_rx_total = Rrx + 1j*w_arr*Lrx + Z_p_rx
        Z_refl = (w_arr * M)**2 / Z_rx_total
        Z_tx_main = Rtx + 1j*w_arr*Ltx + Z_refl
        Z_p_tx = (1/(1j*w_arr*Cp_tx)) * Z_tx_main / ((1/(1j*w_arr*Cp_tx)) + Z_tx_main)
        Z_in = 1j*w_arr*Llcc_tx + 1/(1j*w_arr*Clcc_tx) + Z_p_tx
        I_in = Vin_ac / Z_in
        I_tx = (I_in * Z_p_tx) / Z_tx_main
        I_rx_main = (1j * w_arr * M * I_tx) / Z_rx_total
        I_out_ac = (I_rx_main * Z_p_rx) / Z_load_branch
        P_out_arr = (np.abs(I_out_ac)**2) * RLeq
        P_in_arr = np.real(Vin_ac * np.conj(I_in))

    eff_arr = np.where(P_in_arr > 0, (P_out_arr / P_in_arr) * 100, np.nan)
    return pd.DataFrame({"Frequency (kHz)": f_arr / 1000, "Output Power (W)": P_out_arr, "Efficiency (%)": eff_arr})

# ==========================================
# [Sidebar] API Key, Model Selection & Navigation
# ==========================================
with st.sidebar:
    st.markdown("### 🔑 Backend Config")
    api_key = st.text_input("Gemini API Key", type="password", help="AI 분석을 위한 구글 API 키를 입력하세요.")
    
    selected_model = st.selectbox(
        "LLM 엔진 선택", 
        ["gemini-1.5-flash", "gemini-1.5-pro"],  # <- gemini-pro는 지우고 이 두 개만 넣습니다.
        index=0,
        help="Flash는 응답 속도가 빠르며, Pro는 복잡한 추론에 유리하지만 시간이 더 걸립니다."
    )
    
    if api_key: 
        genai.configure(api_key=api_key)
    st.divider()

if st.session_state.step > 0:
    st.progress(st.session_state.step / 5.0, text=f"Step {st.session_state.step} / 5 진행 중... ({st.session_state.mode} Mode)")
    st.write("<br>", unsafe_allow_html=True)

# ==========================================
# [Phase 0] 진입 화면
# ==========================================
if st.session_state.step == 0:
    st.markdown("<h1 style='text-align: center; font-size: 3rem;'>Intelligent WPT Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: var(--text-color); opacity: 0.7; font-size: 1.2rem; margin-bottom: 3rem;'>무선전력전송 모듈의 요구사항 분석부터 파라미터 도출까지, 원스톱 통합 설계 솔루션.</p>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 2])
    with col2:
        st.info("💡 **Auto Mode (초심자 / 기획자용)**")
        st.write("제약조건만 입력하면, AI가 최적의 토폴로지와 초기 파라미터를 역산하여 추천합니다.")
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
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    
    st.subheader("🔋 어플리케이션 및 배터리 시스템")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        app_type = st.selectbox("어플리케이션 분야", ["드론 (UAV)", "사족보행 로봇", "AGV/AMR", "전기차 (EV)", "모바일/가전"])
        target_power = st.number_input("목표 충전 전력 (W)", value=300.0, step=50.0)
    with c2:
        battery_type = st.selectbox("배터리 셀 화학 조성", ["Li-ion (3.7V)", "LiPo (3.7V)", "LFP (3.2V)"])
        b1, b2, b3 = st.columns([1, 1.2, 1])
        with b1: battery_cells = st.number_input("직렬 셀 (S)", min_value=1, value=13, step=1)
        with b2: battery_capacity = st.number_input("용량 (mAh)", min_value=100, value=22000, step=1000)
        with b3:
            unit_v = 3.2 if "LFP" in battery_type else 3.7
            battery_vol = unit_v * battery_cells
            battery_wh = (battery_capacity / 1000) * battery_vol
            st.metric("팩 공칭 전압", f"{battery_vol:.1f} V", f"총 {battery_wh:.0f} Wh", delta_color="off")
    
    st.divider()
    st.subheader("📐 제약 조건 (Space & Weight)")
    pad_c1, pad_c2, w_c1 = st.columns(3)
    with pad_c1:
        tx_coil_w = st.number_input("Tx 가로 제약 (mm)", value=200, step=10)
        tx_coil_l = st.number_input("Tx 세로 제약 (mm)", value=200, step=10)
    with pad_c2:
        rx_coil_w = st.number_input("Rx 가로 제약 (mm)", value=100, step=10)
        rx_coil_l = st.number_input("Rx 세로 제약 (mm)", value=100, step=10)
    with w_c1:
        rx_weight_limit = st.number_input("수신부(Rx) 허용 무게 (g)", value=400, step=50)
        air_gap = st.number_input("목표 이격 거리 (Air Gap, mm)", value=50, step=5)

    st.write("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([4, 1])
    with c2:
        if st.button("AI 분석 시작하기 ➔", use_container_width=True):
            st.session_state.project_data = {
                "app_type": app_type, "battery_vol": battery_vol,
                "battery": f"{battery_cells}S {battery_type} ({battery_vol:.1f}V)",
                "target_power": target_power, "rx_weight": rx_weight_limit, "air_gap": air_gap,
                "tx_size": f"{tx_coil_w}x{tx_coil_l}", "rx_size": f"{rx_coil_w}x{rx_coil_l}"
            }
            st.session_state.llm_result = None 
            go_to_step(2)
            st.rerun()
            
    st.markdown('</div>', unsafe_allow_html=True)
    st.button("⬅️ 메인으로 돌아가기", on_click=go_to_step, args=(0,))

# ==========================================
# [Phase 2] 지능형 추천 (LLM 연동)
# ==========================================
elif st.session_state.step == 2:
    st.header("Step 2. AI 기반 토폴로지 및 스펙 추천")
    
    if not api_key:
        st.error("🚨 사이드바에 Gemini API Key를 입력하셔야 AI 엔진이 구동됩니다.")
        st.button("⬅️ 이전 단계로 돌아가기", on_click=go_to_step, args=(1,))
    else:
        saved_data = st.session_state.project_data
        if st.session_state.llm_result is None:
            with st.status("🧠 지능형 AI 분석을 시작합니다...", expanded=True) as status:
                try:
                    st.write("⏳ 1. 사용자의 물리적/전기적 제약 조건 분석 중...")
                    prompt = f"""
                    당신은 무선전력전송 설계 엔지니어입니다. 입력값을 바탕으로 'Double LCC' 또는 'LCC-S'를 추천하세요.
                    [조건] App: {saved_data['app_type']}, 전력: {saved_data['target_power']}W, 수신부 허용무게: {saved_data['rx_weight']}g
                    수신부 무게가 500g 이하면 반드시 LCC-S를 추천.
                    반드시 아래 JSON 포맷으로만 응답:
                    {{"topology": "LCC-S 또는 Double LCC", "reasoning": "선정 사유 2문장", "recommended_vin": 정수, "recommended_f0": 85, "estimated_m": 실수}}
                    """
                    
                    st.write(f"⚙️ 2. 최적 토폴로지 및 파라미터 역산 중 ({selected_model})...")
                    model = genai.GenerativeModel(selected_model)
                    
                    response = model.generate_content(
                        prompt, 
                        request_options={"timeout": 15.0}
                    )
                    
                    st.write("🗂️ 3. 결과 데이터 JSON 파싱 및 UI 바인딩 중...")
                    raw_text = response.text.replace('```json', '').replace('```', '').strip()
                    st.session_state.llm_result = json.loads(raw_text)
                    
                    status.update(label="✅ 분석이 완료되었습니다!", state="complete", expanded=False)
                    
                except Exception as e:
                    status.update(label="⚠️ API 통신 지연 또는 에러 발생", state="error", expanded=True)
                    st.error(f"🚨 에러 상세 내용: {e}")
                    st.warning("서버 응답 오류로 내부 휴리스틱 알고리즘으로 우회합니다.")
                    
                    topo = "LCC-S (수신부 초경량화)" if saved_data['rx_weight'] <= 500 else "Double LCC (고효율/CC충전)"
                    st.session_state.llm_result = {"topology": topo, "reasoning": "내부 알고리즘 평가 결과입니다.", "recommended_vin": 100, "recommended_f0": 85, "estimated_m": 15.0}
        
        res = st.session_state.llm_result
        topo_name = "LCC-S (수신부 초경량화)" if "LCC-S" in res['topology'] else "Double LCC (고효율/CC충전)"
        st.session_state.tuning_data['topology'] = topo_name
        st.session_state.tuning_data['Vin'] = float(res['recommended_vin'])
        st.session_state.tuning_data['f0'] = float(res['recommended_f0']) * 1000
        
        coil_params = estimate_coil_params(res['estimated_m'], saved_data['air_gap'], saved_data['rx_weight'])
        st.session_state.tuning_data.update(coil_params)

        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader(f"✅ 추천 토폴로지: **{res['topology']}**")
        st.markdown(f"> **분석 코멘트:**\n> {res['reasoning']}")
        st.divider()
        st.write("### 역산 도출 파라미터 (Initial Spec)")
        col1, col2, col3 = st.columns(3)
        col1.metric("권장 입력 전압 (Vin)", f"{res['recommended_vin']} V")
        col2.metric("요구 상호 인덕턴스 (M)", f"{res['estimated_m']} μH")
        col3.metric("자동 분배된 Ltx / Lrx", f"{coil_params['Ltx']} / {coil_params['Lrx']} μH", help="목표 상호 인덕턴스(M)와 제약 조건을 바탕으로 결정론적 알고리즘이 역산한 값입니다.")
        st.markdown('</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1: st.button("⬅️ 제약 조건 다시 입력하기", on_click=go_to_step, args=(1,), use_container_width=True)
        with c2:
            if st.session_state.mode == 'Auto':
                st.button("시뮬레이션 결과 리포트 생성 ➔", on_click=go_to_step, args=(4,), type="primary", use_container_width=True)
            else:
                st.button("엔지니어 상세 튜닝 (Manual) ➔", on_click=go_to_step, args=(3,), type="primary", use_container_width=True)

# ==========================================
# [Phase 3] 파라미터 상세 설계 (Manual Mode)
# ==========================================
elif st.session_state.step == 3:
    st.header("Step 3. 파라미터 상세 튜닝 (Expert)")
    t_data = st.session_state.tuning_data
    
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("마그네틱 파라미터")
        Ltx = st.slider("송신 코일 Ltx (μH)", 10.0, 300.0, float(t_data.get('Ltx', 80.0)), step=1.0)
        Lrx = st.slider("수신 코일 Lrx (μH)", 10.0, 150.0, float(t_data.get('Lrx', 30.0)), step=1.0)
        k = st.slider("결합 계수 k", 0.05, 0.50, float(t_data.get('k', 0.2)), step=0.001)
        st.session_state.tuning_data.update({'Ltx': Ltx, 'Lrx': Lrx, 'k': k})
    with c2:
        st.subheader("전기적 파라미터")
        Vin = st.number_input("입력 전압 Vin (V)", value=float(t_data.get('Vin', 100.0)), step=10.0)
        Rtx_m = st.number_input("송신 코일 저항 Rtx (mΩ)", value=85.0, step=1.0)
        Rrx_m = st.number_input("수신 코일 저항 Rrx (mΩ)", value=74.0, step=1.0)
        st.session_state.tuning_data.update({'Vin': Vin, 'Rtx': Rtx_m * 1e-3, 'Rrx': Rrx_m * 1e-3})
        
        if "Double" in t_data['topology']:
            current_ratio = st.slider("전류 비율 (Itx/Irx)", 0.5, 3.0, 1.5, step=0.1)
            st.session_state.tuning_data['current_ratio'] = current_ratio
            
    st.markdown('</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.button("⬅️ 이전 단계로", on_click=go_to_step, args=(2,), use_container_width=True)
    with c2: st.button("시뮬레이션 가동 및 리포트 생성 ➔", on_click=go_to_step, args=(4,), type="primary", use_container_width=True)

# ==========================================
# [Phase 4 & 5] 시뮬레이션 및 데이터 내보내기
# ==========================================
elif st.session_state.step == 4:
    st.header("Step 4 & 5. 통합 시뮬레이션 리포트")
    
    t_data = st.session_state.tuning_data
    p_data = st.session_state.project_data
    
    topology = t_data['topology']
    Vin, f0 = t_data.get('Vin', 100.0), t_data.get('f0', 85e3)
    Ltx, Lrx, k = t_data['Ltx'] * 1e-6, t_data['Lrx'] * 1e-6, t_data['k']
    Rtx, Rrx = t_data.get('Rtx', 85e-3), t_data.get('Rrx', 74e-3)
    Vout, Pout = p_data['battery_vol'], p_data['target_power']

    if "LCC-S" in topology:
        res = calculate_lccs(Vin, Vout, Pout, f0, Ltx, Lrx, k, Rtx, Rrx)
    else:
        current_ratio = t_data.get('current_ratio', 1.5)
        res = calculate_double_lcc(Vin, Vout, Pout, f0, Ltx, Lrx, k, current_ratio, Rtx, Rrx)

    if "error" in res:
        st.error(f"🚨 설계 오류: {res['error']}")
        st.button("⬅️ 파라미터 튜닝으로 돌아가기", on_click=go_to_step, args=(3,))
    else:
        st.subheader("📊 AC-AC 전송 효율 및 마그네틱 상태")
        e1, e2, e3 = st.columns(3)
        e1.metric("예상 전송 효율", f"{res['efficiency']:.1f} %", delta="발열 통제 가능" if res['efficiency'] >= 90 else "효율 저하 경고", delta_color="normal" if res['efficiency'] >= 90 else "inverse", help="수식: (P_out_ac / P_in_ac) * 100\n설명: 인버터 및 정류기 손실을 제외한 순수 코일 단에서의 전송 효율입니다.")
        e2.metric("송신 코일 전류 (Itx)", f"{res['Itx']:.2f} A", delta="주의: 30A 초과" if res['Itx'] >= 30 else "안정", delta_color="inverse" if res['Itx'] >= 30 else "normal", help="목표 출력을 내기 위해 메인 코일에 흘려야 하는 1차측 순환 전류입니다.")
        e3.metric("상호 인덕턴스 (M)", f"{res['M']*1e6:.2f} μH", help="수식: M = k * √(Ltx * Lrx)")
        
        st.divider()
        st.subheader("⚡ 보상 소자값 및 내압(Voltage Stress) 분석")
        def display_voltage_metric(label, cap, vol, help_text=""): st.metric(label, f"{cap*1e9:.2f} nF", f"Peak: {vol:.0f} V", delta_color="off", help=help_text)
        
        if "LCC-S" in topology:
            c1, c2, c3 = st.columns(3)
            with c1: display_voltage_metric("Tx 병렬 커패시터 (Cp)", res['Cp'], res['V_Cp_peak'], "수식: Cp = 1 / (ω² * Ls)")
            with c2: display_voltage_metric("Tx 직렬 커패시터 (Cs)", res['Cs'], res['V_Cs_peak'], "수식: Cs = 1 / (ω² * (Ltx - Ls))")
            with c3: display_voltage_metric("Rx 직렬 커패시터 (Crx)", res['Crx'], res['V_Crx_peak'], "수식: Crx = 1 / (ω² * Lrx)")
            st.info(f"💡 정류기 후단 LC 필터 최소 요구치: **Ldc ≥ {res['Ldc_min']*1e6:.2f} μH**")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1: display_voltage_metric("Tx 병렬 커패시터", res['Clcc_tx'], res['V_parallel_tx_peak'], "수식: Clcc_tx = 1 / (ω² * Llcc_tx)")
            with c2: display_voltage_metric("Tx 직렬 커패시터", res['Cp_tx'], res['V_series_tx_peak'], "수식: Cp_tx = 1 / (ω² * (Ltx - Llcc_tx))")
            with c3: display_voltage_metric("Rx 병렬 커패시터", res['Clcc_rx'], res['V_parallel_rx_peak'], "수식: Clcc_rx = 1 / (ω² * Llcc_rx)")
            with c4: display_voltage_metric("Rx 직렬 커패시터", res['Cp_rx'], res['V_series_rx_peak'], "수식: Cp_rx = 1 / (ω² * (Lrx - Llcc_rx))")

        st.divider()
        st.subheader("📈 AC 주파수 응답 (Frequency Response) 시뮬레이션")
        df_freq = simulate_frequency_response(topology, res, f0, Ltx, Lrx, res['M'], Rtx, Rrx)
        
        tab1, tab2 = st.tabs(["전송 효율 (Efficiency)", "출력 전력 (Output Power)"])
        with tab1:
            eff_chart = alt.Chart(df_freq).mark_line(color='#ff7f0e', strokeWidth=3).encode(
                x=alt.X('Frequency (kHz)', scale=alt.Scale(zero=False)),
                y=alt.Y('Efficiency (%)', scale=alt.Scale(zero=False)), tooltip=['Frequency (kHz)', 'Efficiency (%)']
            ).interactive()
            st.altair_chart(eff_chart, use_container_width=True)
        with tab2:
            power_chart = alt.Chart(df_freq).mark_line(color='#1f77b4', strokeWidth=3).encode(
                x=alt.X('Frequency (kHz)', scale=alt.Scale(zero=False)),
                y=alt.Y('Output Power (W)', scale=alt.Scale(zero=False)), tooltip=['Frequency (kHz)', 'Output Power (W)']
            ).interactive()
            st.altair_chart(power_chart, use_container_width=True)
            
        st.divider()
        
        st.subheader("📥 사업계획서/보고서용 데이터 Export")
        export_data = {
            "분류": ["조건", "조건", "조건", "조건", "조건", "결과", "결과", "결과", "결과", "결과", "결과", "결과"],
            "파라미터 항목": ["어플리케이션", "배터리 스펙", "가용 공간(Tx)", "가용 공간(Rx)", "목표 출력(W)", "적용 토폴로지", "입력 전압(V)", "동작 주파수(kHz)", "송신 인덕턴스(uH)", "수신 인덕턴스(uH)", "송신 전류(A)", "AC 효율(%)"],
            "값": [p_data['app_type'], p_data['battery'], p_data['tx_size'], p_data['rx_size'], p_data['target_power'], topology, Vin, f0/1000, Ltx*1e6, Lrx*1e6, res['Itx'], res['efficiency']]
        }
        df_export = pd.DataFrame(export_data)
        csv_data = df_export.to_csv(index=False).encode('utf-8-sig')
        
        c1, c2 = st.columns([1, 1])
        with c1: st.download_button(label="📊 전체 설계 데이터 다운로드 (.csv)", data=csv_data, file_name="wpt_full_report.csv", mime="text/csv", type="primary", use_container_width=True)
        with c2: st.button("🔄 처음부터 다시 설계하기", on_click=go_to_step, args=(0,), use_container_width=True)
