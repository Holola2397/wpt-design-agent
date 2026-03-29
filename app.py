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
    .stMetric { background: var(--background-color); padding: 10px; border-radius: 10px; border: 1px solid var(--border-color); }
    .stButton>button { border-radius: 10px !important; font-weight: 600 !important; transition: all 0.2s ease; }
    .main-next-btn>div>button { background-color: #0A84FF !important; color: white !important; }
    .main-next-btn>div>button:hover { background-color: #0071E3 !important; transform: scale(1.02); }
    .cap-table { width: 100%; text-align: center; border-collapse: collapse; margin-top: 10px; }
    .cap-table th, .cap-table td { border: 1px solid var(--border-color); padding: 8px; }
    .cap-table th { background-color: rgba(128, 128, 128, 0.1); }
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
def reset_project(): 
    st.session_state.step = 0
    st.session_state.mode = None
    st.session_state.llm_result = None
    st.session_state.project_data = {}
    st.session_state.tuning_data = {}

# ==========================================
# [Math Engine] 핵심 설계 수식 모듈 
# ==========================================
def estimate_coil_params(M_req, air_gap_mm, rx_weight_g):
    k_est = max(0.05, 0.4 * math.exp(-air_gap_mm / 60.0))
    Lrx_max = min(80.0, max(10.0, rx_weight_g / 5.0))
    Lrx_target = Lrx_max * 0.8
    Ltx_req = ((M_req * 1e-6) / k_est)**2 / (Lrx_target * 1e-6) if k_est > 0 else 100e-6
    if Ltx_req > 300e-6:
        Ltx_req = 300e-6
        Lrx_target = ((M_req * 1e-6) / k_est)**2 / Ltx_req
    return {"k": round(k_est, 3), "Lrx": round(Lrx_target, 1), "Ltx": round(Ltx_req * 1e6, 1)}

def calculate_ss(Vin, Vout, Ptarget, f0, Ltx, Lrx, k, Rtx, Rrx):
    try:
        w = 2 * math.pi * f0
        Vin_ac_rms = 2 * math.sqrt(2) / math.pi * Vin
        RL = Vout**2 / Ptarget
        RLeq = RL * 8 / (math.pi**2)
        M = k * math.sqrt(Ltx * Lrx)
        Z_refl = (w * M)**2 / (Rrx + RLeq)
        Itx = Vin_ac_rms / (Rtx + Z_refl)
        Irx = (w * M * Itx) / (Rrx + RLeq)
        Pout_actual = (Irx**2) * RLeq
        Iout = Pout_actual / Vout
        Ctx = 1 / (w**2 * Ltx)
        Crx = 1 / (w**2 * Lrx)
        P_loss_tx = (Itx**2) * Rtx; P_loss_rx = (Irx**2) * Rrx
        eff = (Pout_actual / (Pout_actual + P_loss_tx + P_loss_rx)) * 100
        
        caps = {"Tx 직렬 (Ctx)": {"val": Ctx, "form": r"C_{tx} = \frac{1}{\omega^2 L_{tx}}"}, 
                "Rx 직렬 (Crx)": {"val": Crx, "form": r"C_{rx} = \frac{1}{\omega^2 L_{rx}}"}}
        return {"Itx": Itx, "Irx": Irx, "Iout": Iout, "Vout": Vout, "M": M, "efficiency": eff, "Vin_ac": Vin_ac_rms, "RLeq": RLeq, "Pout_actual": Pout_actual, "Ctx": Ctx, "Crx": Crx, "caps": caps}
    except Exception as e: return {"error": f"SS 계산 오류: {str(e)}"}

def calculate_sp(Vin, Vout, Ptarget, f0, Ltx, Lrx, k, Rtx, Rrx):
    try:
        w = 2 * math.pi * f0
        Vin_ac_rms = 2 * math.sqrt(2) / math.pi * Vin
        RL = Vout**2 / Ptarget
        RLeq = RL * (math.pi**2) / 8
        M = k * math.sqrt(Ltx * Lrx)
        Crx = 1 / (w**2 * Lrx)
        Req_s = RLeq / (1 + (w * Crx * RLeq)**2)
        Xeq_s = -(w * Crx * RLeq**2) / (1 + (w * Crx * RLeq)**2)
        Zrx_s = complex(Rrx + Req_s, w*Lrx + Xeq_s)
        Zrefl = (w * M)**2 / Zrx_s
        Ctx = 1 / (w * (w * Ltx + Zrefl.imag))
        Itx = Vin_ac_rms / (Rtx + Zrefl.real)
        Irx_coil = (w * M * Itx) / abs(Zrx_s)
        Vrx_p = Irx_coil * abs(complex(Req_s, Xeq_s))
        Irx_out = Vrx_p / RLeq
        Pout_actual = (Vrx_p**2) / RLeq
        Iout = Pout_actual / Vout
        P_loss_tx = (Itx**2) * Rtx; P_loss_rx = (Irx_coil**2) * Rrx
        eff = (Pout_actual / (Pout_actual + P_loss_tx + P_loss_rx)) * 100
        
        caps = {"Tx 직렬 (Ctx)": {"val": Ctx, "form": r"C_{tx} = \frac{1}{\omega (\omega L_{tx} + Im(Z_{refl}))}"}, 
                "Rx 병렬 (Crx)": {"val": Crx, "form": r"C_{rx} = \frac{1}{\omega^2 L_{rx}}"}}
        return {"Itx": Itx, "Irx": Irx_coil, "Iout": Iout, "Vout": Vout, "M": M, "efficiency": eff, "Vin_ac": Vin_ac_rms, "RLeq": RLeq, "Pout_actual": Pout_actual, "Ctx": Ctx, "Crx": Crx, "caps": caps}
    except Exception as e: return {"error": f"SP 계산 오류: {str(e)}"}

def calculate_lccs(Vin, Vout, Pout, f0, Ltx, Lrx, k, Rtx, Rrx):
    try:
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
        if Ls >= Ltx: return {"error": "Ls > Ltx (보상 불가: Ltx를 키우거나 k를 높이세요)"}
        Cp = 1 / (w**2 * Ls)
        Cs = 1 / (w**2 * (Ltx - Ls))
        Crx = 1 / (w**2 * Lrx)
        P_out_ac = (Irx**2) * RLeq
        P_loss_tx = (Itx**2) * Rtx; P_loss_rx = (Irx**2) * Rrx
        eff = (P_out_ac / (P_out_ac + P_loss_tx + P_loss_rx)) * 100
        
        caps = {"Tx 병렬 (Cp)": {"val": Cp, "form": r"C_p = \frac{1}{\omega^2 L_s}"}, 
                "Tx 직렬 (Cs)": {"val": Cs, "form": r"C_s = \frac{1}{\omega^2 (L_{tx} - L_s)}"}, 
                "Rx 직렬 (Crx)": {"val": Crx, "form": r"C_{rx} = \frac{1}{\omega^2 L_{rx}}"}}
        return {"Itx": Itx, "Irx": Irx, "Iout": Iout, "Vout": Vout, "M": M, "Ls": Ls, "Cp": Cp, "Cs": Cs, "Crx": Crx, "efficiency": eff, "Vin_ac": Vin_ac_rms, "RLeq": RLeq, "Pout_actual": Pout, "caps": caps}
    except Exception as e: return {"error": f"LCC-S 계산 오류: {str(e)}"}

def calculate_double_lcc(Vin, Vout, Pout, f0, Ltx, Lrx, k, current_ratio, Rtx, Rrx):
    try:
        w = 2 * math.pi * f0
        Vin_ac_rms = 2 * math.sqrt(2) / math.pi * Vin
        Iout = Pout / Vout
        RL = Vout / Iout
        RLeq = RL * 8 / (math.pi**2)
        Vrect_in = 2 * math.sqrt(2) / math.pi * Vout
        Iout_ac = Iout * math.sqrt(2) * math.pi / 4
        M = k * math.sqrt(Ltx * Lrx)
        L_prod = (M * Vin_ac_rms) / (w * Iout_ac)
        L_rat = current_ratio * (Vrect_in / Vin_ac_rms)
        Llcc_tx = math.sqrt(L_prod / L_rat)
        Llcc_rx = math.sqrt(L_prod * L_rat)
        if Llcc_tx >= Ltx or Llcc_rx >= Lrx: return {"error": "Llcc > Ltx/Lrx (설계불가: 코일 인덕턴스 부족)"}
        Clcc_tx = 1 / (w**2 * Llcc_tx); Clcc_rx = 1 / (w**2 * Llcc_rx)
        Cp_tx = 1 / (w**2 * (Ltx - Llcc_tx)); Cp_rx = 1 / (w**2 * (Lrx - Llcc_rx))
        Itx = Vin_ac_rms / (w * Llcc_tx)
        Irx = Vrect_in / (w * Llcc_rx)
        P_out_ac = (Irx**2) * RLeq
        P_loss_tx = (Itx**2) * Rtx; P_loss_rx = (Irx**2) * Rrx
        eff = (P_out_ac / (P_out_ac + P_loss_tx + P_loss_rx)) * 100
        
        caps = {"Tx 직렬 (Clcc_tx)": {"val": Clcc_tx, "form": r"C_{f,tx} = \frac{1}{\omega^2 L_{f,tx}}"}, 
                "Tx 병렬 (Cp_tx)": {"val": Cp_tx, "form": r"C_{p,tx} = \frac{1}{\omega^2 (L_{tx} - L_{f,tx})}"}, 
                "Rx 직렬 (Clcc_rx)": {"val": Clcc_rx, "form": r"C_{f,rx} = \frac{1}{\omega^2 L_{f,rx}}"}, 
                "Rx 병렬 (Cp_rx)": {"val": Cp_rx, "form": r"C_{p,rx} = \frac{1}{\omega^2 (L_{rx} - L_{f,rx})}"}}
        return {"Itx": Itx, "Irx": Irx, "Iout": Iout, "Vout": Vout, "M": M, "Llcc_tx": Llcc_tx, "Llcc_rx": Llcc_rx, "efficiency": eff, "Vin_ac": Vin_ac_rms, "RLeq": RLeq, "Pout_actual": Pout, "caps": caps}
    except Exception as e: return {"error": f"Double LCC 계산 오류: {str(e)}"}

def simulate_frequency_response(topology, res_dict, f0, Ltx, Lrx, M, Rtx, Rrx):
    f_arr = np.linspace(f0 - 15e3, f0 + 15e3, 200)
    w_arr = 2 * np.pi * f_arr
    Vin_ac, RLeq = res_dict["Vin_ac"], res_dict["RLeq"]
    try:
        if topology == "SS":
            Ctx, Crx = res_dict['Ctx'], res_dict['Crx']
            Z_rx = RLeq + Rrx + 1j*w_arr*Lrx + 1/(1j*w_arr*Crx)
            Z_in = Rtx + 1j*w_arr*Ltx + 1/(1j*w_arr*Ctx) + (w_arr * M)**2 / Z_rx
            I_rx = (1j * w_arr * M * (Vin_ac / Z_in)) / Z_rx
            P_out_arr = (np.abs(I_rx)**2) * RLeq
        elif topology == "SP":
            Ctx, Crx = res_dict['Ctx'], res_dict['Crx']
            Z_p = 1 / (1/RLeq + 1j*w_arr*Crx)
            Z_rx = Rrx + 1j*w_arr*Lrx + Z_p
            Z_in = Rtx + 1j*w_arr*Ltx + 1/(1j*w_arr*Ctx) + (w_arr * M)**2 / Z_rx
            I_rx_coil = (1j * w_arr * M * (Vin_ac / Z_in)) / Z_rx
            P_out_arr = (np.abs(I_rx_coil * Z_p)**2) / RLeq
        elif topology == "LCC-S":
            Ls, Cp, Cs, Crx = res_dict['Ls'], res_dict['Cp'], res_dict['Cs'], res_dict['Crx']
            Z_rx = RLeq + Rrx + 1j*w_arr*Lrx + 1/(1j*w_arr*Crx)
            Z_tx_branch = Rtx + 1j*w_arr*Ltx + 1/(1j*w_arr*Cs) + (w_arr * M)**2 / Z_rx
            Z_p = (1/(1j*w_arr*Cp)) * Z_tx_branch / ((1/(1j*w_arr*Cp)) + Z_tx_branch)
            I_tx = ((Vin_ac / (1j*w_arr*Ls + Z_p)) * Z_p) / Z_tx_branch
            I_rx = (1j * w_arr * M * I_tx) / Z_rx
            P_out_arr = (np.abs(I_rx)**2) * RLeq
        else: # Double LCC
            P_out_arr = np.full_like(f_arr, res_dict['Pout_actual']) # 단순화를 위해 고정 출력 반환 (전체 시뮬 복잡도 방지)
            
        P_in_arr = P_out_arr / (res_dict['efficiency']/100) # 근사 효율 곡선
        eff_arr = np.where(P_in_arr > 0, (P_out_arr / P_in_arr) * 100, np.nan)
    except:
        P_out_arr, eff_arr = np.zeros_like(f_arr), np.zeros_like(f_arr)
    return pd.DataFrame({"Frequency (kHz)": f_arr / 1000, "Output Power (W)": P_out_arr, "Efficiency (%)": eff_arr})

# ==========================================
# [Sidebar] API & Model Selection
# ==========================================
with st.sidebar:
    st.markdown("### 🔑 Backend Config")
    api_key = st.text_input("Gemini API Key", type="password")
    available_models = ["models/gemini-1.5-flash"]
    if api_key:
        try:
            genai.configure(api_key=api_key)
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        except: pass
    selected_model = st.selectbox("LLM 엔진 선택", available_models, index=0)
    st.divider()
    if st.button("🏠 플랫폼 홈으로", use_container_width=True):
        reset_project(); st.rerun()

if st.session_state.step > 0:
    st.progress(st.session_state.step / 5.0, text=f"Step {st.session_state.step} / 5 진행 중...")

# ==========================================
# [Phases 0 ~ 3: Entry -> Tuning (전문가 모드)] 
# ==========================================
# (기존 코드와 동일하므로, Step 0~3은 축약 없이 그대로 유지합니다.)
if st.session_state.step == 0:
    st.markdown("<h1 style='text-align: center; font-size: 3rem; margin-top: 50px;'>Intelligent WPT Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; opacity: 0.7; font-size: 1.2rem;'>무선충전 시스템 설계의 지능형 파트너</p>", unsafe_allow_html=True)
    st.write("<br><br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.info("💡 **Auto Mode**\n\n초심자용. 제약조건 입력 시 AI가 모든 스펙을 추천합니다.")
        st.button("Auto Mode 시작", use_container_width=True, on_click=lambda: (st.session_state.update({"mode": "Auto", "step": 1})))
    with col2:
        st.warning("⚙️ **Manual Mode**\n\n전문가용. AI 추천값을 기반으로 정밀 튜닝이 가능합니다.")
        st.button("Manual Mode 시작", use_container_width=True, on_click=lambda: (st.session_state.update({"mode": "Manual", "step": 1})))

elif st.session_state.step == 1:
    st.header("Step 1. 시스템 요구사항 및 제약 조건")
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    app_type = st.selectbox("적용 분야", ["드론 (UAV)", "사족보행 로봇", "AGV/AMR", "전기차 (EV)", "모바일/가전"])
    c1, c2, c3 = st.columns(3)
    target_p = c1.number_input("목표 전력 (W)", value=300.0)
    batt_t = c2.selectbox("배터리 셀", ["Li-ion (3.7V)", "LFP (3.2V)"])
    batt_s = c3.number_input("직렬 셀 (S)", min_value=1, value=13)
    unit_v = 3.2 if "LFP" in batt_t else 3.7
    batt_vol = unit_v * batt_s
    st.metric("배터리 팩 공칭 전압", f"{batt_vol:.1f} V")
    st.divider()
    s1, s2, s3 = st.columns(3)
    tx_s = s1.text_input("Tx 가용 면적 (mm)", "200x200")
    rx_s = s2.text_input("Rx 가용 면적 (mm)", "150x150")
    rx_w = s3.number_input("Rx 무게 제약 (g)", value=400)
    gap = st.number_input("이격 거리 (Air Gap, mm)", value=50)
    st.write("<br>", unsafe_allow_html=True)
    
    n1, n2, n3 = st.columns([1, 1, 2])
    n1.button("⬅️ 홈으로", on_click=reset_project)
    if n3.button("제약 조건 확정 및 분석 시작 ➔", use_container_width=True, type="primary"):
        st.session_state.project_data = {"app_type": app_type, "battery_vol": batt_vol, "target_power": target_p, "rx_weight": rx_w, "air_gap": gap, "tx_size": tx_s, "rx_size": rx_s, "battery_info": f"{batt_s}S {batt_t}"}
        go_to_step(2); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.step == 2:
    st.header("Step 2. AI 기반 최적 토폴로지 추천")
    if not api_key: 
        st.error("사이드바에 API Key를 입력해주세요.")
        st.button("⬅️ Step 1로 돌아가기", on_click=go_to_step, args=(1,))
    else:
        sd = st.session_state.project_data
        if st.session_state.llm_result is None:
            with st.status("🧠 AI 수석 엔지니어 분석 중 (SS, SP, LCC-S, Double LCC)...", expanded=True) as status:
                try:
                    model = genai.GenerativeModel(selected_model)
                    prompt = f"WPT Engineer: App {sd['app_type']}, Power {sd['target_power']}W, Vbatt {sd['battery_vol']}V, RxWeight {sd['rx_weight']}g. Recommend one WPT topology among ['SS', 'SP', 'LCC-S', 'Double LCC']. JSON only: {{\"topology\": \"string\", \"reasoning\": \"string\", \"recommended_vin\": int, \"recommended_f0\": 85, \"estimated_m\": float}}"
                    resp = model.generate_content(prompt, request_options={"timeout": 15.0})
                    st.session_state.llm_result = json.loads(resp.text.replace('```json', '').replace('```', '').strip())
                    status.update(label="✅ 분석 완료", state="complete")
                except:
                    if sd['rx_weight'] < 100 or sd['target_power'] < 50: topo = "SS"
                    elif sd['battery_vol'] > 100: topo = "SP"
                    elif sd['target_power'] > 1000: topo = "Double LCC"
                    else: topo = "LCC-S"
                    st.session_state.llm_result = {"topology": topo, "reasoning": "내부 알고리즘 기반 추천 완료.", "recommended_vin": 100, "recommended_f0": 85, "estimated_m": 15.0}
        
        res = st.session_state.llm_result
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader(f"✅ 추천 토폴로지: **{res['topology']}**")
        st.info(res['reasoning'])
        
        cp = estimate_coil_params(res['estimated_m'], sd['air_gap'], sd['rx_weight'])
        st.session_state.tuning_data = {
            "topology": res['topology'], "Vin": float(res['recommended_vin']), 
            "f0": res['recommended_f0']*1000, "Ltx": float(cp['Ltx']), "Lrx": float(cp['Lrx']), 
            "k": float(cp['k']), "ratio": 1.5, "Rtx": 0.085, "Rrx": 0.074
        }
        
        st.write("<br>", unsafe_allow_html=True)
        n1, n2, n3 = st.columns([1, 1, 2])
        n1.button("⬅️ 이전 (Step 1)", on_click=go_to_step, args=(1,))
        n2.button("🏠 홈", on_click=reset_project)
        next_step = 4 if st.session_state.mode == 'Auto' else 3
        if n3.button("다음 단계로 ➔", use_container_width=True, type="primary"):
            go_to_step(next_step); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.step == 3:
    st.header("Step 3. 파라미터 상세 튜닝 (Expert)")
    td = st.session_state.tuning_data; sd = st.session_state.project_data
    
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        st.subheader("🛠️ 설계 파라미터")
        topo_sel = st.selectbox("토폴로지 변경", ["SS", "SP", "LCC-S", "Double LCC"], index=["SS", "SP", "LCC-S", "Double LCC"].index(td['topology']))
        Ltx = st.slider("송신 코일 Ltx (uH)", 10.0, 300.0, float(td['Ltx']))
        Lrx = st.slider("수신 코일 Lrx (uH)", 10.0, 150.0, float(td['Lrx']))
        k = st.slider("결합 계수 k", 0.05, 0.5, float(td['k']))
        vin = st.number_input("입력 전압 Vin (V)", value=td['Vin'])
        f0 = st.number_input("주파수 f0 (kHz)", value=td['f0']/1000) * 1000
        ratio = st.slider("전류 비율 (Itx/Irx)", 0.5, 3.0, 1.5) if "Double" in topo_sel else 1.0
        
    with col2:
        st.subheader("⚡ 실시간 설계 프리뷰")
        if topo_sel == "SS": res = calculate_ss(vin, sd['battery_vol'], sd['target_power'], f0, Ltx*1e-6, Lrx*1e-6, k, 0.085, 0.074)
        elif topo_sel == "SP": res = calculate_sp(vin, sd['battery_vol'], sd['target_power'], f0, Ltx*1e-6, Lrx*1e-6, k, 0.085, 0.074)
        elif topo_sel == "LCC-S": res = calculate_lccs(vin, sd['battery_vol'], sd['target_power'], f0, Ltx*1e-6, Lrx*1e-6, k, 0.085, 0.074)
        else: res = calculate_double_lcc(vin, sd['battery_vol'], sd['target_power'], f0, Ltx*1e-6, Lrx*1e-6, k, ratio, 0.085, 0.074)
            
        if "error" in res: st.error(res['error'])
        else:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric("상호 인덕턴스 (M)", f"{res['M']*1e6:.1f} uH")
            p2.metric("예상 효율", f"{res['efficiency']:.1f} %")
            p3.metric("Itx (송신 전류)", f"{res['Itx']:.2f} A")
            p4.metric("Irx (수신 전류)", f"{res['Irx']:.2f} A")
            
            p5, p6 = st.columns(2)
            p5.metric("목표 전력", f"{sd['target_power']:.1f} W")
            p6.metric("실제 설계 전력", f"{res['Pout_actual']:.1f} W")

            st.divider()
            cap_html = "<table class='cap-table'><tr>"
            for name in res['caps'].keys(): cap_html += f"<th>{name}</th>"
            cap_html += "</tr><tr>"
            for data in res['caps'].values(): cap_html += f"<td><b>{data['val']*1e9:.2f} nF</b></td>"
            cap_html += "</tr></table>"
            st.markdown(cap_html, unsafe_allow_html=True)
            
    st.write("<br>", unsafe_allow_html=True)
    n1, n2, n3 = st.columns([1, 1, 2])
    n1.button("⬅️ 이전 (Step 2)", on_click=go_to_step, args=(2,))
    n2.button("🏠 홈", on_click=reset_project)
    if n3.button("설계 확정 및 리포트 생성 ➔", use_container_width=True, type="primary"):
        if "error" in res: st.error("오류를 해결한 후 진행해주세요.")
        else:
            st.session_state.tuning_data.update({"topology": topo_sel, "Ltx": Ltx, "Lrx": Lrx, "k": k, "Vin": vin, "f0": f0, "ratio": ratio})
            go_to_step(4); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# [Phase 4 & 5] 통합 설계 리포트 (초심자/전문가 공통)
# ==========================================
elif st.session_state.step == 4:
    st.header("Step 4. 통합 설계 리포트")
    td = st.session_state.tuning_data; sd = st.session_state.project_data
    
    # 1. 계산 수행
    if td['topology'] == "SS": res = calculate_ss(td['Vin'], sd['battery_vol'], sd['target_power'], td['f0'], td['Ltx']*1e-6, td['Lrx']*1e-6, td['k'], 0.085, 0.074)
    elif td['topology'] == "SP": res = calculate_sp(td['Vin'], sd['battery_vol'], sd['target_power'], td['f0'], td['Ltx']*1e-6, td['Lrx']*1e-6, td['k'], 0.085, 0.074)
    elif td['topology'] == "LCC-S": res = calculate_lccs(td['Vin'], sd['battery_vol'], sd['target_power'], td['f0'], td['Ltx']*1e-6, td['Lrx']*1e-6, td['k'], 0.085, 0.074)
    else: res = calculate_double_lcc(td['Vin'], sd['battery_vol'], sd['target_power'], td['f0'], td['Ltx']*1e-6, td['Lrx']*1e-6, td['k'], td.get('ratio', 1.5), 0.085, 0.074)
    
    if "error" in res:
        st.error(f"설계 오류: {res['error']}")
        st.button("⬅️ Step 3로 돌아가기", on_click=go_to_step, args=(3,))
        st.stop()

    st.info("💡 팁: 결과값 우측 상단의 **[ ? ]** 아이콘에 마우스를 올리면 해당 값이 도출된 계산 수식을 확인할 수 있습니다.")

    # [Section 1] 시스템 & 입출력 파라미터
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.subheader("⚡ 시스템 성능 및 입출력 (System & Output)")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("입력 전압 (Vin)", f"{td['Vin']:.1f} V")
    s2.metric("출력 전압 (Vout)", f"{res['Vout']:.1f} V", help=r"배터리 공칭 전압과 동일하게 설정됨")
    s3.metric("동작 주파수 (f0)", f"{td['f0']/1000:.1f} kHz")
    s4.metric("최종 효율 (Eff)", f"{res['efficiency']:.1f} %", help=r"$\eta = \frac{P_{out}}{P_{out} + P_{loss}} \times 100$")
    
    s5, s6, s7, s8 = st.columns(4)
    s5.metric("토폴로지", td['topology'])
    s6.metric("부하 저항 (R_Leq)", f"{res['RLeq']:.2f} Ω", help=r"AC 등가 부하 저항 (토폴로지에 따라 $\frac{8}{\pi^2}$ 또는 $\frac{\pi^2}{8}$ 배율 적용)")
    s7.metric("출력 전력 (Pout)", f"{res['Pout_actual']:.1f} W", help=r"$P_{out} = I_{out} \times V_{out}$")
    s8.metric("출력 전류 (Iout)", f"{res['Iout']:.2f} A", help=r"$I_{out} = \frac{P_{out}}{V_{out}}$")
    st.markdown('</div>', unsafe_allow_html=True)

    # [Section 2] 코일 및 공진 소자
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.subheader("🧲 코일 & 공진 네트워크 (Resonators)")
    
    # 코일 인덕턴스와 전류
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("송신 코일 (Ltx)", f"{td['Ltx']:.1f} uH")
    c2.metric("수신 코일 (Lrx)", f"{td['Lrx']:.1f} uH")
    c3.metric("결합계수 (k)", f"{td['k']:.3f}")
    c4.metric("상호 인덕턴스 (M)", f"{res['M']*1e6:.1f} uH", help=r"$M = k \sqrt{L_{tx} L_{rx}}$")
    
    c5, c6 = st.columns(2)
    c5.metric("송신 코일 전류 (Itx_rms)", f"{res['Itx']:.2f} A", help="송신측 주공진 코일에 흐르는 RMS 전류")
    c6.metric("수신 코일 전류 (Irx_rms)", f"{res['Irx']:.2f} A", help="수신측 주공진 코일에 흐르는 RMS 전류")
    
    st.divider()
    st.markdown("**🔹 공진 커패시터 및 추가 인덕터 설계값**")
    
    # 공진 소자들을 컬럼 형태로 배치하여 Tooltip(help) 지원
    cap_cols = st.columns(len(res['caps']) + (1 if td['topology'] == 'LCC-S' else (2 if 'Double' in td['topology'] else 0)))
    idx = 0
    for name, data in res['caps'].items():
        cap_cols[idx].metric(name, f"{data['val']*1e9:.2f} nF", help=f"$${data['form']}$$")
        idx += 1
        
    if td['topology'] == "LCC-S":
        cap_cols[idx].metric("Tx 보상 인덕터 (Ls)", f"{res['Ls']*1e6:.1f} uH", help=r"$$L_s = \frac{V_{in,ac}}{\omega I_{tx}}$$")
    elif 'Double' in td['topology']:
        cap_cols[idx].metric("Tx 보상 인덕터 (Llcc_tx)", f"{res['Llcc_tx']*1e6:.1f} uH", help=r"전류 비율 및 M에 의해 산출됨")
        cap_cols[idx+1].metric("Rx 보상 인덕터 (Llcc_rx)", f"{res['Llcc_rx']*1e6:.1f} uH", help=r"전류 비율 및 M에 의해 산출됨")
    st.markdown('</div>', unsafe_allow_html=True)

    # [Section 3] Wire Sizing Guide (리쯔와이어 계산기)
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.subheader("🧵 권선(Wire) 설계 가이드 및 리쯔와이어 계산기")
    req_tx_sq = res['Itx'] / 7.0
    req_rx_sq = res['Irx'] / 7.0
    
    st.info(f"💡 통상적으로 코일에 흐르는 전류 **7A 당 1 SQ ($mm^2$)** 이상의 단면적을 가진 와이어를 권장합니다.\n\n"
            f"- **Tx 코일 필요 단면적:** $\ge$ {req_tx_sq:.2f} $mm^2$ \n"
            f"- **Rx 코일 필요 단면적:** $\ge$ {req_rx_sq:.2f} $mm^2$", icon="📏")
    
    st.write("**🧮 리쯔와이어(Litz Wire) 스펙 검증기**")
    w1, w2, w3 = st.columns(3)
    d_strand = w1.number_input("가닥 지름 (mm) ex) 0.1", value=0.10, step=0.01)
    n_strand = w2.number_input("가닥 수 (N) ex) 600", value=600, step=10)
    
    calc_sq = ((d_strand / 2)**2) * math.pi * n_strand
    w3.metric("계산된 Litz 단면적", f"{calc_sq:.2f} mm²", help=r"$$SQ = (\frac{d}{2})^2 \times \pi \times N$$")
    
    if calc_sq >= max(req_tx_sq, req_rx_sq):
        st.success(f"✅ 입력한 {d_strand}mm / {n_strand}가닥 리쯔와이어({calc_sq:.2f} sq)는 Tx/Rx 전류 규격을 모두 만족합니다.")
    else:
        st.error(f"⚠️ 입력한 리쯔와이어 스펙이 부족합니다. 가닥 수(N)를 더 늘리거나 두꺼운 선을 선택하세요.")
    st.markdown('</div>', unsafe_allow_html=True)

    # [Section 4] 시뮬레이션
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.subheader("📈 주파수 응답 특성 시뮬레이션")
    df_f = simulate_frequency_response(td['topology'], res, td['f0'], td['Ltx']*1e-6, td['Lrx']*1e-6, res['M'], 0.085, 0.074)
    st.altair_chart(alt.Chart(df_f).mark_line(color='#0A84FF').encode(x=alt.X('Frequency (kHz)', scale=alt.Scale(zero=False)), y=alt.Y('Efficiency (%)', scale=alt.Scale(zero=False)), tooltip=['Frequency (kHz)', 'Efficiency (%)', 'Output Power (W)']).interactive(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # [Section 5] CSV 데이터 생성 및 다운로드
    # Dictionary를 풀어서 CSV로 저장
    csv_dict = {
        "항목": ["적용 분야", "토폴로지", "목표 전력 (W)", "실제 전력 (W)", "입력 전압 Vin (V)", "출력 전압 Vout (V)", "출력 전류 Iout (A)", 
               "주파수 (kHz)", "최종 효율 (%)", "Tx 코일 (uH)", "Rx 코일 (uH)", "결합계수 (k)", "상호 인덕턴스 M (uH)", 
               "Itx (A)", "Irx (A)", "필요 Tx 권선 (sq)", "필요 Rx 권선 (sq)"],
        "값": [sd['app_type'], td['topology'], sd['target_power'], res['Pout_actual'], td['Vin'], res['Vout'], res['Iout'], 
              td['f0']/1000, res['efficiency'], td['Ltx'], td['Lrx'], td['k'], res['M']*1e6, 
              res['Itx'], res['Irx'], req_tx_sq, req_rx_sq]
    }
    # 공진 커패시터 정보 추가
    for name, data in res['caps'].items():
        csv_dict["항목"].append(name)
        csv_dict["값"].append(f"{data['val']*1e9:.2f} nF")
        
    csv_data = pd.DataFrame(csv_dict).to_csv(index=False).encode('utf-8-sig')
    
    # 하단 내비게이션 바
    n1, n2, n3 = st.columns([1, 1, 2])
    n1.button("⬅️ 설계 수정 (Step 3)", on_click=go_to_step, args=(3 if st.session_state.mode == 'Manual' else 2,))
    n2.button("🏠 홈으로", on_click=reset_project)
    n3.download_button("📊 전체 설계 리포트 다운로드 (CSV)", csv_data, "WPT_Full_Report.csv", "text/csv", use_container_width=True, type="primary")
