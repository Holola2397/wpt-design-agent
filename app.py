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
    k_est = max(0.05, 0.4 * math.exp(-air_gap_mm / 60.0))
    Lrx_max = min(80.0, max(10.0, rx_weight_g / 5.0))
    Lrx_target = Lrx_max * 0.8
    Ltx_req = ((M_req * 1e-6) / k_est)**2 / (Lrx_target * 1e-6)
    if Ltx_req > 300e-6:
        Ltx_req = 300e-6
        Lrx_target = ((M_req * 1e-6) / k_est)**2 / Ltx_req
    return {"k": round(k_est, 3), "Lrx": round(Lrx_target, 1), "Ltx": round(Ltx_req * 1e6, 1)}

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
    Cp = 1 / (w**2 * Ls); Cs = 1 / (w**2 * (Ltx - Ls)); Crx = 1 / (w**2 * Lrx)
    Ldc_min = RL / (3 * w)
    V_Ltx_peak = math.sqrt(2) * w * Ltx * Itx; V_Cp_peak = math.sqrt(2) * w * Ls * Itx
    V_Cs_peak = math.sqrt(2) * Itx / (w * Cs); V_Crx_peak = math.sqrt(2) * Irx / (w * Crx)
    P_out_ac = (Irx**2) * RLeq; P_loss_tx = (Itx**2) * Rtx; P_loss_rx = (Irx**2) * Rrx
    P_in_ac = P_out_ac + P_loss_tx + P_loss_rx
    efficiency = (P_out_ac / P_in_ac) * 100 if P_in_ac > 0 else 0
    return {"Itx": Itx, "Irx": Irx, "M": M, "Ls": Ls, "Cp": Cp, "Cs": Cs, "Crx": Crx, "Ldc_min": Ldc_min,
            "V_Ltx_peak": V_Ltx_peak, "V_Cp_peak": V_Cp_peak, "V_Cs_peak": V_Cs_peak, "V_Crx_peak": V_Crx_peak,
            "P_out_ac": P_out_ac, "P_loss_tx": P_loss_tx, "P_loss_rx": P_loss_rx, "efficiency": efficiency, "RLeq": RLeq, "Vin_ac": Vin_ac_rms}

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
    Llcc_tx = math.sqrt(L_product / L_ratio); Llcc_rx = math.sqrt(L_product * L_ratio)
    if Llcc_tx >= Ltx or Llcc_rx >= Lrx: return {"error": "보상 인덕터(Llcc)가 메인 코일보다 큽니다."}
    Clcc_tx = 1 / (w**2 * Llcc_tx); Clcc_rx = 1 / (w**2 * Llcc_rx)
    Cp_tx = 1 / (w**2 * (Ltx - Llcc_tx)); Cp_rx = 1 / (w**2 * (Lrx - Llcc_rx))
    Itx = Vin_ac_rms / (w * Llcc_tx); Irx = Vrect_in / (w * Llcc_rx)
    V_Ltx_peak = math.sqrt(2) * w * Ltx * Itx; V_parallel_tx_peak = math.sqrt(2) * w * Llcc_tx * Itx; V_series_tx_peak = math.sqrt(2) * Itx / (w * Cp_tx)
    V_Lrx_peak = math.sqrt(2) * w * Lrx * Irx; V_parallel_rx_peak = math.sqrt(2) * w * Llcc_rx * Irx; V_series_rx_peak = math.sqrt(2) * Irx / (w * Cp_rx)
    P_out_ac = (Irx**2) * RLeq; P_loss_tx = (Itx**2) * Rtx; P_loss_rx = (Irx**2) * Rrx
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
    if "LCC-S" in topology:
        Ls, Cp, Cs, Crx = res_dict['Ls'], res_dict['Cp'], res_dict['Cs'], res_dict['Crx']
        Z_rx = RLeq + Rrx + 1j*w_arr*Lrx + 1/(1j*w_arr*Crx)
        Z_refl = (w_arr * M)**2 / Z_rx
        Z_tx_branch = Rtx + 1j*w_arr*Ltx + 1/(1j*w_arr*Cs) + Z_refl
        Z_p = (1/(1j*w_arr*Cp)) * Z_tx_branch / ((1/(1j*w_arr*Cp)) + Z_tx_branch)
        Z_in = 1j*w_arr*Ls + Z_p
        I_in = Vin_ac / Z_in; I_tx = (I_in * Z_p) / Z_tx_branch; I_rx = (1j * w_arr * M * I_tx) / Z_rx
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
        I_in = Vin_ac / Z_in; I_tx = (I_in * Z_p_tx) / Z_tx_main
        I_rx_main = (1j * w_arr * M * I_tx) / Z_rx_total; I_out_ac = (I_rx_main * Z_p_rx) / Z_load_branch
    P_out_arr = (np.abs(I_rx if "LCC-S" in topology else I_out_ac)**2) * RLeq
    P_in_arr = np.real(Vin_ac * np.conj(I_in))
    eff_arr = np.where(P_in_arr > 0, (P_out_arr / P_in_arr) * 100, np.nan)
    return pd.DataFrame({"Frequency (kHz)": f_arr / 1000, "Output Power (W)": P_out_arr, "Efficiency (%)": eff_arr})

# ==========================================
# [Sidebar] API Key & Dynamic Model Sync
# ==========================================
with st.sidebar:
    st.markdown("### 🔑 Backend Config")
    api_key = st.text_input("Gemini API Key", type="password")
    available_models = []
    if api_key:
        try:
            genai.configure(api_key=api_key)
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    available_models.append(m.name)
        except: pass
    if not available_models: available_models = ["models/gemini-1.5-flash", "models/gemini-1.5-pro"]
    selected_model = st.selectbox("LLM 엔진 선택 (서버 동기화)", available_models, index=0)
    st.divider()

if st.session_state.step > 0:
    st.progress(st.session_state.step / 5.0, text=f"Step {st.session_state.step} / 5 진행 중...")

# ==========================================
# [Phase 0] Entry
# ==========================================
if st.session_state.step == 0:
    st.markdown("<h1 style='text-align: center;'>Intelligent WPT Platform</h1>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1: st.button("Auto Mode (초심자)", use_container_width=True, on_click=set_mode_and_next, args=('Auto',))
    with col2: st.button("Manual Mode (고급)", use_container_width=True, on_click=set_mode_and_next, args=('Manual',))

# ==========================================
# [Phase 1] Requirements
# ==========================================
elif st.session_state.step == 1:
    st.header("Step 1. 시스템 요구사항 및 제약 조건")
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    app_type = st.selectbox("적용 분야", ["드론 (UAV)", "사족보행 로봇", "AGV/AMR", "전기차 (EV)", "모바일/가전"])
    p_c1, p_c2, p_c3 = st.columns(3)
    target_p = p_c1.number_input("목표 전력 (W)", value=300.0)
    batt_t = p_c2.selectbox("배터리 셀", ["Li-ion (3.7V)", "LFP (3.2V)"])
    batt_s = p_c3.number_input("직렬 셀 (S)", min_value=1, value=13)
    unit_v = 3.2 if "LFP" in batt_t else 3.7
    batt_vol = unit_v * batt_s
    st.metric("배터리 공칭 전압", f"{batt_vol:.1f} V")
    st.divider()
    s_c1, s_c2, s_c3 = st.columns(3)
    tx_s = s_c1.text_input("Tx 공간 (mm)", "200x200")
    rx_s = s_c2.text_input("Rx 공간 (mm)", "100x100")
    rx_w = s_c3.number_input("Rx 무게 제약 (g)", value=400)
    gap = st.number_input("이격 거리 (mm)", value=50)
    if st.button("AI 분석 시작 ➔", use_container_width=True):
        st.session_state.project_data = {"app_type": app_type, "battery_vol": batt_vol, "target_power": target_p, "rx_weight": rx_w, "air_gap": gap, "tx_size": tx_s, "rx_size": rx_s}
        go_to_step(2); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# [Phase 2] AI Recommendation
# ==========================================
elif st.session_state.step == 2:
    st.header("Step 2. AI 기반 토폴로지 추천")
    if not api_key: st.error("API Key 필요"); st.button("⬅️ 이전", on_click=go_to_step, args=(1,))
    else:
        sd = st.session_state.project_data
        if st.session_state.llm_result is None:
            with st.status("🧠 AI 분석 중...", expanded=True) as status:
                try:
                    model = genai.GenerativeModel(selected_model)
                    prompt = f"WPT Engineer: App {sd['app_type']}, Power {sd['target_power']}W, RxWeight {sd['rx_weight']}g. Recommend 'LCC-S' or 'Double LCC' in JSON only: {{\"topology\": \"string\", \"reasoning\": \"string\", \"recommended_vin\": int, \"recommended_f0\": 85, \"estimated_m\": float}}"
                    resp = model.generate_content(prompt, request_options={"timeout": 15.0})
                    st.session_state.llm_result = json.loads(resp.text.replace('```json', '').replace('```', '').strip())
                    status.update(label="✅ 분석 완료", state="complete")
                except Exception as e:
                    st.error(f"Error: {e}"); topo = "LCC-S" if sd['rx_weight'] <= 500 else "Double LCC"
                    st.session_state.llm_result = {"topology": topo, "reasoning": "내부 알고리즘 평가", "recommended_vin": 100, "recommended_f0": 85, "estimated_m": 15.0}
        res = st.session_state.llm_result
        st.subheader(f"✅ 추천: {res['topology']}"); st.info(res['reasoning'])
        c_params = estimate_coil_params(res['estimated_m'], sd['air_gap'], sd['rx_weight'])
        st.session_state.tuning_data = {"topology": res['topology'], "Vin": float(res['recommended_vin']), "f0": res['recommended_f0']*1000, **c_params}
        st.divider()
        c1, c2 = st.columns(2)
        c1.button("⬅️ 재입력", on_click=go_to_step, args=(1,))
        next_step = 4 if st.session_state.mode == 'Auto' else 3
        c2.button("다음 단계 ➔", on_click=go_to_step, args=(next_step,), type="primary")

# ==========================================
# [Phase 3] Expert Tuning
# ==========================================
elif st.session_state.step == 3:
    st.header("Step 3. 파라미터 상세 튜닝")
    td = st.session_state.tuning_data
    with st.form("tuning"):
        Ltx = st.slider("Ltx (uH)", 10.0, 300.0, float(td['Ltx']))
        Lrx = st.slider("Lrx (uH)", 10.0, 150.0, float(td['Lrx']))
        k = st.slider("k", 0.05, 0.5, float(td['k']))
        vin = st.number_input("Vin (V)", value=td['Vin'])
        if st.form_submit_button("설계 확정"):
            st.session_state.tuning_data.update({"Ltx": Ltx, "Lrx": Lrx, "k": k, "Vin": vin, "Rtx": 0.085, "Rrx": 0.074})
            go_to_step(4); st.rerun()

# ==========================================
# [Phase 4 & 5] Report & Export
# ==========================================
elif st.session_state.step == 4:
    st.header("Step 4 & 5. 설계 리포트")
    td = st.session_state.tuning_data; pd_data = st.session_state.project_data
    # 파라미터 보정 (Auto 모드 대비)
    if 'Rtx' not in td: td.update({'Rtx': 0.085, 'Rrx': 0.074})
    Ltx_h, Lrx_h = td['Ltx']*1e-6, td['Lrx']*1e-6
    if "LCC-S" in td['topology']: res = calculate_lccs(td['Vin'], pd_data['battery_vol'], pd_data['target_power'], td['f0'], Ltx_h, Lrx_h, td['k'], td['Rtx'], td['Rrx'])
    else: res = calculate_double_lcc(td['Vin'], pd_data['battery_vol'], pd_data['target_power'], td['f0'], Ltx_h, Lrx_h, td['k'], 1.5, td['Rtx'], td['Rrx'])
    
    if "error" in res: st.error(res['error']); st.button("⬅️ 수정", on_click=go_to_step, args=(3,))
    else:
        st.metric("AC-AC 효율", f"{res['efficiency']:.1f} %")
        df_f = simulate_frequency_response(td['topology'], res, td['f0'], Ltx_h, Lrx_h, res['M'], td['Rtx'], td['Rrx'])
        st.altair_chart(alt.Chart(df_f).mark_line().encode(x='Frequency (kHz)', y='Efficiency (%)').interactive(), use_container_width=True)
        csv = pd.DataFrame({"Param": ["Topology", "Eff", "Itx"], "Value": [td['topology'], res['efficiency'], res['Itx']]}).to_csv(index=False).encode('utf-8-sig')
        st.download_button("📊 보고서 다운로드", csv, "report.csv", "text/csv")
        st.button("🔄 초기화", on_click=go_to_step, args=(0,))
