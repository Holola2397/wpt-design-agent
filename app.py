import streamlit as st
import numpy as np
import math
import pandas as pd
import altair as alt
import google.generativeai as genai
import json
import time
import re

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
    .ai-coach { background-color: rgba(10, 132, 255, 0.1); border-left: 4px solid #0A84FF; padding: 15px; border-radius: 8px; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# [Session State & Language Helper]
# ==========================================
if 'step' not in st.session_state: st.session_state.step = 0
if 'mode' not in st.session_state: st.session_state.mode = None
if 'project_data' not in st.session_state: st.session_state.project_data = {}
if 'llm_result' not in st.session_state: st.session_state.llm_result = None
if 'tuning_data' not in st.session_state: st.session_state.tuning_data = {}
if 'lang' not in st.session_state: st.session_state.lang = 'KR'

def go_to_step(step_num): st.session_state.step = step_num
def reset_project(): 
    st.session_state.step = 0
    st.session_state.mode = None
    st.session_state.llm_result = None
    st.session_state.project_data = {}
    st.session_state.tuning_data = {}

def t(kr, en): return kr if st.session_state.lang == 'KR' else en

# ==========================================
# [Math Engine] WPT 파라미터 계산 
# ==========================================
def estimate_coil_params(air_gap_mm, rx_weight_g):
    k_est = max(0.05, 0.4 * math.exp(-air_gap_mm / 60.0))
    Lrx_target = min(80.0, max(10.0, rx_weight_g / 5.0)) * 0.8
    Ltx_req = 100.0 if k_est > 0 else 100.0
    return {"k": round(k_est, 3), "Lrx": round(Lrx_target, 1), "Ltx": round(Ltx_req, 1)}

def calculate_ss(Vin, Vout_cv, Ptarget, f0, Ltx, Lrx, k, Rtx, Rrx):
    try:
        w = 2 * math.pi * f0; Vin_ac_rms = 2 * math.sqrt(2) / math.pi * Vin
        RL = Vout_cv**2 / Ptarget; RLeq = RL * 8 / (math.pi**2)
        M = k * math.sqrt(Ltx * Lrx)
        Z_refl = (w * M)**2 / (Rrx + RLeq)
        Itx = Vin_ac_rms / (Rtx + Z_refl); Irx = (w * M * Itx) / (Rrx + RLeq)
        Pout_actual = (Irx**2) * RLeq; Iout = Pout_actual / Vout_cv
        Ctx = 1 / (w**2 * Ltx); Crx = 1 / (w**2 * Lrx)
        P_loss_tx = (Itx**2) * Rtx; P_loss_rx = (Irx**2) * Rrx
        eff = (Pout_actual / (Pout_actual + P_loss_tx + P_loss_rx)) * 100
        caps = {"Tx (Ctx)": {"val": Ctx, "form": r"C_{tx} = 1/(\omega^2 L_{tx})"}, "Rx (Crx)": {"val": Crx, "form": r"C_{rx} = 1/(\omega^2 L_{rx})"}}
        return {"Itx": Itx, "Irx": Irx, "Iout": Iout, "Vout": Vout_cv, "M": M, "efficiency": eff, "Vin_ac": Vin_ac_rms, "RLeq": RLeq, "Pout_actual": Pout_actual, "Ctx": Ctx, "Crx": Crx, "caps": caps, "P_loss_tx": P_loss_tx, "P_loss_rx": P_loss_rx}
    except Exception as e: return {"error": f"Error: {str(e)}"}

def calculate_sp(Vin, Vout_cv, Ptarget, f0, Ltx, Lrx, k, Rtx, Rrx):
    try:
        w = 2 * math.pi * f0; Vin_ac_rms = 2 * math.sqrt(2) / math.pi * Vin
        RL = Vout_cv**2 / Ptarget; RLeq = RL * (math.pi**2) / 8
        M = k * math.sqrt(Ltx * Lrx); Crx = 1 / (w**2 * Lrx)
        Req_s = RLeq / (1 + (w * Crx * RLeq)**2); Xeq_s = -(w * Crx * RLeq**2) / (1 + (w * Crx * RLeq)**2)
        Zrx_s = complex(Rrx + Req_s, w*Lrx + Xeq_s); Zrefl = (w * M)**2 / Zrx_s
        Ctx = 1 / (w * (w * Ltx + Zrefl.imag))
        Itx = Vin_ac_rms / (Rtx + Zrefl.real); Irx_coil = (w * M * Itx) / abs(Zrx_s)
        Vrx_p = Irx_coil * abs(complex(Req_s, Xeq_s)); Irx_out = Vrx_p / RLeq
        Pout_actual = (Vrx_p**2) / RLeq; Iout = Pout_actual / Vout_cv
        P_loss_tx = (Itx**2) * Rtx; P_loss_rx = (Irx_coil**2) * Rrx
        eff = (Pout_actual / (Pout_actual + P_loss_tx + P_loss_rx)) * 100
        caps = {"Tx (Ctx)": {"val": Ctx, "form": r"C_{tx}"}, "Rx (Crx)": {"val": Crx, "form": r"C_{rx}"}}
        return {"Itx": Itx, "Irx": Irx_coil, "Iout": Iout, "Vout": Vout_cv, "M": M, "efficiency": eff, "Vin_ac": Vin_ac_rms, "RLeq": RLeq, "Pout_actual": Pout_actual, "Ctx": Ctx, "Crx": Crx, "caps": caps, "P_loss_tx": P_loss_tx, "P_loss_rx": P_loss_rx}
    except Exception as e: return {"error": f"Error: {str(e)}"}

def calculate_lccs(Vin, Vout_cv, Pout, f0, Ltx, Lrx, k, Rtx, Rrx):
    try:
        w = 2 * math.pi * f0; Vin_ac_rms = 2 * math.sqrt(2) / math.pi * Vin
        Iout = Pout / Vout_cv; RL = Vout_cv / Iout; RLeq = RL * (math.pi**2) / 8
        Vrect_in = Vout_cv * math.pi / (2 * math.sqrt(2)); Iout_ac = Iout * (2 * math.sqrt(2)) / math.pi
        M = k * math.sqrt(Ltx * Lrx)
        Itx = Vrect_in / (w * M); Irx = Iout_ac; Ls = Vin_ac_rms / (w * Itx)
        if Ls >= Ltx: return {"error": t("Ls > Ltx (보상 불가: 입력전압을 낮추거나, Ltx/k를 높이세요)", "Ls > Ltx (Cannot compensate. Lower Vin or increase Ltx/k)")}
        Cp = 1 / (w**2 * Ls); Cs = 1 / (w**2 * (Ltx - Ls)); Crx = 1 / (w**2 * Lrx)
        P_out_ac = (Irx**2) * RLeq; P_loss_tx = (Itx**2) * Rtx; P_loss_rx = (Irx**2) * Rrx
        eff = (P_out_ac / (P_out_ac + P_loss_tx + P_loss_rx)) * 100
        caps = {"Tx (Cp)": {"val": Cp, "form": ""}, "Tx (Cs)": {"val": Cs, "form": ""}, "Rx (Crx)": {"val": Crx, "form": ""}}
        return {"Itx": Itx, "Irx": Irx, "Iout": Iout, "Vout": Vout_cv, "M": M, "Ls": Ls, "Cp": Cp, "Cs": Cs, "Crx": Crx, "efficiency": eff, "Vin_ac": Vin_ac_rms, "RLeq": RLeq, "Pout_actual": Pout, "caps": caps, "P_loss_tx": P_loss_tx, "P_loss_rx": P_loss_rx}
    except Exception as e: return {"error": f"Error: {str(e)}"}

def calculate_double_lcc(Vin, Vout_cv, Pout, f0, Ltx, Lrx, k, current_ratio, Rtx, Rrx):
    try:
        w = 2 * math.pi * f0; Vin_ac_rms = 2 * math.sqrt(2) / math.pi * Vin
        Iout = Pout / Vout_cv; RL = Vout_cv / Iout; RLeq = RL * 8 / (math.pi**2)
        Vrect_in = 2 * math.sqrt(2) / math.pi * Vout_cv; Iout_ac = Iout * math.sqrt(2) * math.pi / 4
        M = k * math.sqrt(Ltx * Lrx)
        L_prod = (M * Vin_ac_rms) / (w * Iout_ac); L_rat = current_ratio * (Vrect_in / Vin_ac_rms)
        Llcc_tx = math.sqrt(L_prod / L_rat); Llcc_rx = math.sqrt(L_prod * L_rat)
        if Llcc_tx >= Ltx or Llcc_rx >= Lrx: return {"error": t("Llcc > Ltx/Lrx (설계 불가: 코일 인덕턴스 부족)", "Llcc > Ltx/Lrx (Inductance too small)")}
        Clcc_tx = 1 / (w**2 * Llcc_tx); Clcc_rx = 1 / (w**2 * Llcc_rx)
        Cp_tx = 1 / (w**2 * (Ltx - Llcc_tx)); Cp_rx = 1 / (w**2 * (Lrx - Llcc_rx))
        Itx = Vin_ac_rms / (w * Llcc_tx); Irx = Vrect_in / (w * Llcc_rx)
        P_out_ac = (Irx**2) * RLeq; P_loss_tx = (Itx**2) * Rtx; P_loss_rx = (Irx**2) * Rrx
        eff = (P_out_ac / (P_out_ac + P_loss_tx + P_loss_rx)) * 100
        caps = {"Tx(Clcc_tx)": {"val": Clcc_tx, "form": ""}, "Tx(Cp_tx)": {"val": Cp_tx, "form": ""}, "Rx(Clcc_rx)": {"val": Clcc_rx, "form": ""}, "Rx(Cp_rx)": {"val": Cp_rx, "form": ""}}
        return {"Itx": Itx, "Irx": Irx, "Iout": Iout, "Vout": Vout_cv, "M": M, "Llcc_tx": Llcc_tx, "Llcc_rx": Llcc_rx, "efficiency": eff, "Vin_ac": Vin_ac_rms, "RLeq": RLeq, "Pout_actual": Pout, "caps": caps, "P_loss_tx": P_loss_tx, "P_loss_rx": P_loss_rx}
    except Exception as e: return {"error": f"Error: {str(e)}"}

# 주파수 응답 AC 방정식 완벽 수정본
def simulate_frequency_response(topology, res_dict, f0, Ltx, Lrx, M, Rtx, Rrx):
    f_arr = np.linspace(f0 - 20e3, f0 + 20e3, 300)
    w_arr = 2 * np.pi * f_arr
    Vin_ac, RLeq = res_dict["Vin_ac"], res_dict["RLeq"]
    
    P_out_arr = np.zeros_like(f_arr)
    P_in_arr = np.zeros_like(f_arr)
    
    try:
        if topology == "SS":
            Ctx, Crx = res_dict['Ctx'], res_dict['Crx']
            Z_rx = RLeq + Rrx + 1j*w_arr*Lrx + 1/(1j*w_arr*Crx)
            Z_in = Rtx + 1j*w_arr*Ltx + 1/(1j*w_arr*Ctx) + (w_arr * M)**2 / Z_rx
            I_in = Vin_ac / Z_in
            I_rx = (1j * w_arr * M * I_in) / Z_rx
            P_out_arr = (np.abs(I_rx)**2) * RLeq
            P_in_arr = np.real(Vin_ac * np.conj(I_in))
            
        elif topology == "SP":
            Ctx, Crx = res_dict['Ctx'], res_dict['Crx']
            Z_p = 1 / (1/RLeq + 1j*w_arr*Crx)
            Z_rx = Rrx + 1j*w_arr*Lrx + Z_p
            Z_in = Rtx + 1j*w_arr*Ltx + 1/(1j*w_arr*Ctx) + (w_arr * M)**2 / Z_rx
            I_in = Vin_ac / Z_in
            I_rx_coil = (1j * w_arr * M * I_in) / Z_rx
            V_rx_load = I_rx_coil * Z_p
            P_out_arr = (np.abs(V_rx_load)**2) / RLeq
            P_in_arr = np.real(Vin_ac * np.conj(I_in))
            
        elif topology == "LCC-S":
            Ls, Cp, Cs, Crx = res_dict['Ls'], res_dict['Cp'], res_dict['Cs'], res_dict['Crx']
            Z_rx = RLeq + Rrx + 1j*w_arr*Lrx + 1/(1j*w_arr*Crx)
            Z_tx_main = Rtx + 1j*w_arr*Ltx + 1/(1j*w_arr*Cs) + (w_arr * M)**2 / Z_rx
            Z_p = (1/(1j*w_arr*Cp)) * Z_tx_main / ((1/(1j*w_arr*Cp)) + Z_tx_main)
            Z_in = 1j*w_arr*Ls + Z_p
            I_in = Vin_ac / Z_in
            I_tx = I_in * Z_p / Z_tx_main
            I_rx = (1j * w_arr * M * I_tx) / Z_rx
            P_out_arr = (np.abs(I_rx)**2) * RLeq
            P_in_arr = np.real(Vin_ac * np.conj(I_in))
            
        else: # Double LCC
            Llcc_tx, Cp_tx, Clcc_tx = res_dict['Llcc_tx'], res_dict['Cp_tx'], res_dict['Clcc_tx']
            Llcc_rx, Cp_rx, Clcc_rx = res_dict['Llcc_rx'], res_dict['Cp_rx'], res_dict['Clcc_rx']
            
            Z_load_b = RLeq + 1j*w_arr*Llcc_rx + 1/(1j*w_arr*Clcc_rx)
            Z_prx = (1/(1j*w_arr*Cp_rx)) * Z_load_b / ((1/(1j*w_arr*Cp_rx)) + Z_load_b)
            Z_rx_tot = Rrx + 1j*w_arr*Lrx + Z_prx
            Z_refl = (w_arr * M)**2 / Z_rx_tot
            
            Z_tx_main = Rtx + 1j*w_arr*Ltx + Z_refl
            Z_ptx = (1/(1j*w_arr*Cp_tx)) * Z_tx_main / ((1/(1j*w_arr*Cp_tx)) + Z_tx_main)
            Z_in = 1j*w_arr*Llcc_tx + 1/(1j*w_arr*Clcc_tx) + Z_ptx
            
            I_in = Vin_ac / Z_in
            I_tx = I_in * Z_ptx / Z_tx_main
            I_rx_coil = (1j * w_arr * M * I_tx) / Z_rx_tot
            I_out_ac = I_rx_coil * Z_prx / Z_load_b
            
            P_out_arr = (np.abs(I_out_ac)**2) * RLeq
            P_in_arr = np.real(Vin_ac * np.conj(I_in))
            
    except: pass
    
    eff_arr = np.where(P_in_arr > 0, (P_out_arr / P_in_arr) * 100, 0)
    eff_arr = np.clip(eff_arr, 0, 100) # Cap at 100% 
    return pd.DataFrame({"Frequency (kHz)": f_arr / 1000, "Output Power (W)": P_out_arr, "Efficiency (%)": eff_arr})

def generate_ai_coaching(res):
    advice = []
    if res['Itx'] > 15.0: advice.append(t(f"🔥 **Tx 발열 경고:** 송신 전류({res['Itx']:.1f}A)가 매우 높습니다. 동손({res['P_loss_tx']:.1f}W)을 줄이려면 Vin을 높이거나 Ltx를 키우세요.", f"🔥 **High Tx Temp:** Itx({res['Itx']:.1f}A) is too high. Increase Vin or Ltx."))
    if res['Irx'] > 10.0: advice.append(t(f"🔥 **Rx 발열 경고:** 수신 전류({res['Irx']:.1f}A)가 높아 발열({res['P_loss_rx']:.1f}W)이 우려됩니다.", f"🔥 **High Rx Temp:** Irx({res['Irx']:.1f}A) is high. Check cooling."))
    if res['efficiency'] < 85.0: advice.append(t("📉 **효율 저하:** 효율이 낮습니다. 결합계수(k)를 높이거나 오정렬을 줄이세요.", "📉 **Low Efficiency:** Try increasing coupling coefficient (k)."))
    if not advice: advice.append(t("✅ **설계 양호:** 코일 전류 및 손실, 효율이 안정적인 범위 내에 있습니다.", "✅ **Optimal:** Parameters are within safe operational limits."))
    return "<br>".join(advice)

# ==========================================
# [Sidebar - Language & Config]
# ==========================================
with st.sidebar:
    st.markdown("### 🌐 Language / 언어")
    lang_sel = st.radio("UI Language", ["KR", "EN"], horizontal=True, label_visibility="collapsed")
    st.session_state.lang = lang_sel
    
    st.divider()
    st.markdown(t("### 🔑 백엔드 설정", "### 🔑 Backend Config"))
    api_key = st.text_input("Gemini API Key", type="password")
    available_models = ["models/gemini-1.5-flash"]
    if api_key:
        try:
            genai.configure(api_key=api_key)
            available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        except: pass
    selected_model = st.selectbox(t("LLM 엔진 선택", "LLM Engine"), available_models, index=0)
    st.divider()
    if st.button(t("🏠 플랫폼 홈으로", "🏠 Home"), use_container_width=True):
        reset_project(); st.rerun()

if st.session_state.step > 0:
    st.progress(st.session_state.step / 5.0, text=f"Step {st.session_state.step} / 5...")

# ==========================================
# [Phase 0]
# ==========================================
if st.session_state.step == 0:
    st.markdown(f"<h1 style='text-align: center; font-size: 3rem; margin-top: 50px;'>Intelligent WPT Platform</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; opacity: 0.7; font-size: 1.2rem;'>{t('무선충전 시스템 설계의 지능형 파트너', 'Intelligent Partner for WPT System Design')}</p>", unsafe_allow_html=True)
    st.write("<br><br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.info(t("💡 **Auto Mode**\n\n초심자용. 제약조건 입력 시 AI가 코일 인덕턴스까지 포함하여 최적의 파라미터를 추천합니다.", "💡 **Auto Mode**\n\nFor beginners. AI recommends optimal parameters including coil inductance."))
        st.button(t("Auto Mode 시작", "Start Auto Mode"), use_container_width=True, on_click=lambda: (st.session_state.update({"mode": "Auto", "step": 1})))
    with col2:
        st.warning(t("⚙️ **Manual Mode**\n\n전문가용. AI 추천값을 초기값으로 두고 모든 파라미터를 정밀 튜닝합니다.", "⚙️ **Manual Mode**\n\nFor experts. Fine-tune all parameters based on AI initial values."))
        st.button(t("Manual Mode 시작", "Start Manual Mode"), use_container_width=True, on_click=lambda: (st.session_state.update({"mode": "Manual", "step": 1})))

# ==========================================
# [Phase 1] 
# ==========================================
elif st.session_state.step == 1:
    st.header(t("Step 1. 시스템 요구사항 및 제약 조건", "Step 1. System Requirements & Constraints"))
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    app_type = st.selectbox(t("적용 분야", "Application"), ["Drone (UAV)", "Quadruped Robot", "AGV/AMR", "EV", "Mobile/Home Appliance"])
    c1, c2, c3 = st.columns(3)
    target_p = c1.number_input(t("목표 전력 (W)", "Target Power (W)"), value=300.0)
    batt_t = c2.selectbox(t("배터리 셀 타입", "Battery Cell Type"), ["Li-ion (Nominal 3.7V / CV 4.2V)", "LFP (Nominal 3.2V / CV 3.65V)"])
    batt_s = c3.number_input(t("직렬 셀 (S)", "Series Cells (S)"), min_value=1, value=13)
    
    v_nom = 3.2 if "LFP" in batt_t else 3.7
    v_charge = 3.65 if "LFP" in batt_t else 4.2
    batt_vol_nom = v_nom * batt_s; batt_vol_charge = v_charge * batt_s
    
    st.info(t(f"🔋 **배터리 팩 분석:** 공칭 전압은 **{batt_vol_nom:.1f}V** 이며, 최대 충전(CV) 전압은 **{batt_vol_charge:.1f}V** 입니다. WPT 설계는 최대 충전 전압을 타겟($V_{{out}}$)으로 진행됩니다.", 
              f"🔋 **Battery Pack:** Nominal **{batt_vol_nom:.1f}V**, Max CV **{batt_vol_charge:.1f}V**. Design target is max CV."))
    
    st.divider()
    st.markdown(t("📐 **가용 공간 및 무게 제약**", "📐 **Space & Weight Constraints**"))
    # 가로, 세로, 높이 입력으로 UI 변경
    tx_c1, tx_c2, tx_c3 = st.columns(3)
    tx_w = tx_c1.number_input(t("Tx 코일 가로 (mm)", "Tx Width (mm)"), value=200)
    tx_l = tx_c2.number_input(t("Tx 코일 세로 (mm)", "Tx Length (mm)"), value=200)
    tx_h = tx_c3.number_input(t("Tx 가용 두께 (mm)", "Tx Height (mm)"), value=10)
    
    rx_c1, rx_c2, rx_c3 = st.columns(3)
    rx_w = rx_c1.number_input(t("Rx 코일 가로 (mm)", "Rx Width (mm)"), value=150)
    rx_l = rx_c2.number_input(t("Rx 코일 세로 (mm)", "Rx Length (mm)"), value=150)
    rx_h = rx_c3.number_input(t("Rx 가용 두께 (mm)", "Rx Height (mm)"), value=8)
    
    o1, o2 = st.columns(2)
    rx_weight = o1.number_input(t("Rx 무게 제약 (g)", "Rx Weight Limit (g)"), value=400)
    gap = o2.number_input(t("이격 거리 (Air Gap, mm)", "Air Gap (mm)"), value=50)
    st.write("<br>", unsafe_allow_html=True)
    
    n1, n2, n3 = st.columns([1, 1, 2])
    n1.button(t("⬅️ 홈으로", "⬅️ Home"), on_click=reset_project)
    if n3.button(t("제약 조건 확정 및 분석 시작 ➔", "Confirm & Start Analysis ➔"), use_container_width=True, type="primary"):
        st.session_state.project_data = {
            "app_type": app_type, "battery_vol_charge": batt_vol_charge, "battery_vol_nom": batt_vol_nom, 
            "target_power": target_p, "rx_weight": rx_weight, "air_gap": gap, 
            "tx_size": f"{tx_w}x{tx_l}x{tx_h}", "rx_size": f"{rx_w}x{rx_l}x{rx_h}", "battery_info": f"{batt_s}S {batt_t[:6]}"
        }
        go_to_step(2); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# [Phase 2] AI 추천 (Language Aware)
# ==========================================
elif st.session_state.step == 2:
    st.header(t("Step 2. AI 엔지니어 종합 설계 제안", "Step 2. AI Engineer Design Proposal"))
    if not api_key: 
        st.error(t("사이드바에 API Key를 입력해주세요.", "Please enter Gemini API Key in the sidebar."))
        st.button(t("⬅️ Step 1로 돌아가기", "⬅️ Back to Step 1"), on_click=go_to_step, args=(1,))
    else:
        sd = st.session_state.project_data
        if st.session_state.llm_result is None:
            with st.status(t("🧠 AI 수석 엔지니어 분석 중 (최대 2분 소요될 수 있습니다)...", "🧠 AI Analyzing (may take up to 60s)..."), expanded=True) as status:
                lang_target = "Korean" if st.session_state.lang == "KR" else "English"
                for attempt in range(3):
                    try:
                        model = genai.GenerativeModel(selected_model)
                        prompt = f"""
                        You are an expert WPT Engineer. Analyze:
                        App: {sd['app_type']}, Power: {sd['target_power']}W, Vout: {sd['battery_vol_charge']}V, RxWeight: {sd['rx_weight']}g, AirGap: {sd['air_gap']}mm, TxSize: {sd['tx_size']}, RxSize: {sd['rx_size']}.
                        
                        Based on real-world WPT design papers, provide practical recommendations for Ltx, Lrx, and k.
                        Respond ONLY in a flat JSON format. The JSON values (reasoning, coil_design, shielding_guide) MUST be written in natural {lang_target} language.
                        {{
                            "topology": "Choose ONE: SS, SP, LCC-S, Double LCC",
                            "reasoning": "Explain why this topology is best.",
                            "recommended_vin": <integer>,
                            "recommended_f0": 85,
                            "recommended_ltx": <float representing Tx inductance in uH>,
                            "recommended_lrx": <float representing Rx inductance in uH>,
                            "recommended_k": <float between 0.05 and 0.5>,
                            "coil_design": "Recommend a coil shape (Circular, Rectangular, DD).",
                            "shielding_guide": "Advice on ferrite core and shielding."
                        }}
                        """
                        resp = model.generate_content(prompt, request_options={"timeout": 120.0})
                        match = re.search(r'\{.*\}', resp.text, re.DOTALL)
                        if match:
                            st.session_state.llm_result = json.loads(match.group(0))
                            break
                        else: raise ValueError("JSON block not found")
                            
                    except Exception as e:
                        if attempt == 2:
                            if sd['rx_weight'] < 100 or sd['target_power'] < 50: topo = "SS"
                            elif sd['battery_vol_charge'] > 100: topo = "SP"
                            elif sd['target_power'] > 1000: topo = "Double LCC"
                            else: topo = "LCC-S"
                            cp = estimate_coil_params(sd['air_gap'], sd['rx_weight'])
                            st.session_state.llm_result = {
                                "topology": topo, 
                                "reasoning": t("API 지연으로 내부 알고리즘이 선정했습니다.", "Selected by internal fallback algorithm due to API delay."), 
                                "recommended_vin": 100, "recommended_f0": 85, 
                                "recommended_ltx": cp["Ltx"], "recommended_lrx": cp["Lrx"], "recommended_k": cp["k"],
                                "coil_design": t("원형(Circular) 코일 추천.", "Circular coil recommended."),
                                "shielding_guide": t("얇은 페라이트 시트 사용 권장.", "Thin ferrite sheet recommended.")
                            }
                status.update(label=t("✅ 분석 완료", "✅ Analysis Complete"), state="complete")
        
        res = st.session_state.llm_result
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader(t(f"✅ AI 수석 엔지니어 추천: **{res['topology']}** 토폴로지", f"✅ AI Recommendation: **{res['topology']}** Topology"))
        st.info(f"**💡 {t('추천 사유', 'Reasoning')}:** {res['reasoning']}")
        st.divider()
        c1, c2 = st.columns(2)
        c1.markdown(f"**🧲 {t('코일 형상 가이드', 'Coil Design Guide')}:**<br>{res['coil_design']}", unsafe_allow_html=True)
        c2.markdown(f"**🛡️ {t('차폐/무게 가이드', 'Shielding Guide')}:**<br>{res['shielding_guide']}", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader(t("📊 무선충전 4대 토폴로지 특성 비교표", "📊 4 Major WPT Topologies Comparison"))
        df_topo = pd.DataFrame({
            t("토폴로지", "Topology"): ["SS (Series-Series)", "SP (Series-Parallel)", "LCC-S", "Double LCC"],
            t("주요 특성", "Key Features"): [t("가장 단순한 구조, 부하에 따라 전류 변동 심함", "Simplest structure, highly load-dependent current"), 
                                      t("수신부 병렬 공진, 고전압 부하에 유리", "Parallel Rx resonance, good for high voltage loads"), 
                                      t("송신 전류(Itx) 일정 유지, 오정렬에 강함", "Constant Tx current, robust to misalignment"), 
                                      t("고전력 전송에 특화, 대칭 구조로 설계 용이", "Specialized for high power, symmetric design")],
            t("적용 분야", "Applications"): [t("소형 기기, 저전력 (100W 미만)", "Small devices, low power (<100W)"), 
                                      t("수신부 공간 제약, 고전압 배터리(로봇 등)", "High voltage batteries, limited Rx space"), 
                                      t("중/대용량 전력, 위치 변동이 잦은 환경 (드론, AGV)", "Medium/High power, variable gap (Drones, AGVs)"), 
                                      t("전기차(EV), 산업용 고전력(1kW 이상) 장비", "EVs, Industrial high power (>1kW)")]
        })
        st.table(df_topo)
        st.markdown('</div>', unsafe_allow_html=True)

        st.session_state.tuning_data = {
            "topology": res['topology'], "Vin": float(res['recommended_vin']), 
            "f0": res['recommended_f0']*1000, "Ltx": float(res['recommended_ltx']), "Lrx": float(res['recommended_lrx']), 
            "k": float(res['recommended_k']), "ratio": 1.5, "Rtx": 0.085, "Rrx": 0.074
        }
        
        n1, n2, n3 = st.columns([1, 1, 2])
        n1.button(t("⬅️ 이전 (Step 1)", "⬅️ Back (Step 1)"), on_click=go_to_step, args=(1,))
        n2.button(t("🏠 홈", "🏠 Home"), on_click=reset_project)
        next_step = 4 if st.session_state.mode == 'Auto' else 3
        if n3.button(t("다음 단계로 ➔", "Next Step ➔"), use_container_width=True, type="primary"):
            go_to_step(next_step); st.rerun()

# ==========================================
# [Phase 3] 상세 튜닝 (Expert)
# ==========================================
elif st.session_state.step == 3:
    st.header(t("Step 3. 파라미터 상세 튜닝 (Expert)", "Step 3. Detailed Parameter Tuning (Expert)"))
    td = st.session_state.tuning_data; sd = st.session_state.project_data
    
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1.2, 1.8])
    with col1:
        st.subheader(t("🛠️ 설계 파라미터 조작", "🛠️ Adjust Parameters"))
        topo_sel = st.selectbox(t("토폴로지", "Topology"), ["SS", "SP", "LCC-S", "Double LCC"], index=["SS", "SP", "LCC-S", "Double LCC"].index(td['topology']))
        safe_ltx = min(max(float(td['Ltx']), 10.0), 1000.0)
        safe_lrx = min(max(float(td['Lrx']), 10.0), 500.0)
        safe_k = min(max(float(td['k']), 0.01), 0.90)
        Ltx = st.slider(t("송신 코일 Ltx (uH)", "Tx Coil Ltx (uH)"), 10.0, 1000.0, safe_ltx)
        Lrx = st.slider(t("수신 코일 Lrx (uH)", "Rx Coil Lrx (uH)"), 10.0, 500.0, safe_lrx)
        k = st.slider(t("결합 계수 k", "Coupling Coefficient k"), 0.01, 0.90, safe_k)
        vin = st.number_input(t("입력 전압 Vin (V)", "Input Voltage Vin (V)"), value=td['Vin'])
        f0 = st.number_input(t("주파수 f0 (kHz)", "Frequency f0 (kHz)"), value=td['f0']/1000) * 1000
        ratio = st.slider(t("전류 비율 (Itx/Irx)", "Current Ratio (Itx/Irx)"), 0.5, 3.0, 1.5) if "Double" in topo_sel else 1.0
        
    with col2:
        st.subheader(t("⚡ 실시간 설계 프리뷰 & 코칭", "⚡ Real-time Preview & Coaching"))
        if topo_sel == "SS": res = calculate_ss(vin, sd['battery_vol_charge'], sd['target_power'], f0, Ltx*1e-6, Lrx*1e-6, k, 0.085, 0.074)
        elif topo_sel == "SP": res = calculate_sp(vin, sd['battery_vol_charge'], sd['target_power'], f0, Ltx*1e-6, Lrx*1e-6, k, 0.085, 0.074)
        elif topo_sel == "LCC-S": res = calculate_lccs(vin, sd['battery_vol_charge'], sd['target_power'], f0, Ltx*1e-6, Lrx*1e-6, k, 0.085, 0.074)
        else: res = calculate_double_lcc(vin, sd['battery_vol_charge'], sd['target_power'], f0, Ltx*1e-6, Lrx*1e-6, k, ratio, 0.085, 0.074)
            
        if "error" in res: st.error(res['error'])
        else:
            p1, p2, p3, p4 = st.columns(4)
            p1.metric(t("상호 인덕턴스(M)", "Mutual Ind.(M)"), f"{res['M']*1e6:.1f} uH")
            p2.metric(t("예상 효율", "Est. Efficiency"), f"{res['efficiency']:.1f} %")
            p3.metric("Itx", f"{res['Itx']:.2f} A")
            p4.metric("Irx", f"{res['Irx']:.2f} A")
            
            p5, p6 = st.columns(2)
            p5.metric(t("목표 전력", "Target Power"), f"{sd['target_power']:.1f} W")
            p6.metric(t("실제 설계 전력", "Actual Power"), f"{res['Pout_actual']:.1f} W")

            st.divider()
            st.markdown(t("**🔹 공진 커패시터 및 인덕터**", "**🔹 Resonant Caps & Inductors**"))
            cap_cols = st.columns(len(res['caps']) + (1 if topo_sel == 'LCC-S' else (2 if 'Double' in topo_sel else 0)))
            idx = 0
            for name, data in res['caps'].items(): cap_cols[idx].metric(name, f"{data['val']*1e9:.2f} nF"); idx += 1
            if topo_sel == "LCC-S": cap_cols[idx].metric("Ls", f"{res['Ls']*1e6:.1f} uH")
            elif 'Double' in topo_sel: cap_cols[idx].metric("Llcc_tx", f"{res['Llcc_tx']*1e6:.1f} uH"); cap_cols[idx+1].metric("Llcc_rx", f"{res['Llcc_rx']*1e6:.1f} uH")
            st.markdown(f"<div class='ai-coach'>💡 <b>AI Coaching</b><br>{generate_ai_coaching(res)}</div>", unsafe_allow_html=True)
            
    st.write("<br>", unsafe_allow_html=True)
    n1, n2, n3 = st.columns([1, 1, 2])
    n1.button(t("⬅️ 이전 (Step 2)", "⬅️ Back (Step 2)"), on_click=go_to_step, args=(2,))
    n2.button(t("🏠 홈", "🏠 Home"), on_click=reset_project)
    if n3.button(t("설계 확정 및 리포트 생성 ➔", "Finalize & Gen Report ➔"), use_container_width=True, type="primary"):
        if "error" in res: st.error(t("오류를 해결한 후 진행해주세요.", "Resolve errors before proceeding."))
        else:
            st.session_state.tuning_data.update({"topology": topo_sel, "Ltx": Ltx, "Lrx": Lrx, "k": k, "Vin": vin, "f0": f0, "ratio": ratio})
            go_to_step(4); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# [Phase 4] 통합 설계 리포트 
# ==========================================
elif st.session_state.step == 4:
    st.header(t("Step 4. 통합 설계 리포트", "Step 4. Integrated Design Report"))
    td = st.session_state.tuning_data; sd = st.session_state.project_data
    
    if td['topology'] == "SS": res = calculate_ss(td['Vin'], sd['battery_vol_charge'], sd['target_power'], td['f0'], td['Ltx']*1e-6, td['Lrx']*1e-6, td['k'], 0.085, 0.074)
    elif td['topology'] == "SP": res = calculate_sp(td['Vin'], sd['battery_vol_charge'], sd['target_power'], td['f0'], td['Ltx']*1e-6, td['Lrx']*1e-6, td['k'], 0.085, 0.074)
    elif td['topology'] == "LCC-S": res = calculate_lccs(td['Vin'], sd['battery_vol_charge'], sd['target_power'], td['f0'], td['Ltx']*1e-6, td['Lrx']*1e-6, td['k'], 0.085, 0.074)
    else: res = calculate_double_lcc(td['Vin'], sd['battery_vol_charge'], sd['target_power'], td['f0'], td['Ltx']*1e-6, td['Lrx']*1e-6, td['k'], td.get('ratio', 1.5), 0.085, 0.074)
    
    if "error" in res:
        st.error(t(f"설계 오류: {res['error']}", f"Design Error: {res['error']}"))
        st.button(t("⬅️ Step 3로 돌아가기", "⬅️ Back to Step 3"), on_click=go_to_step, args=(3,))
        st.stop()

    # [Section 1]
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.subheader(t("⚡ 시스템 성능 및 입출력 (System & Output)", "⚡ System Performance & I/O"))
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Vin", f"{td['Vin']:.1f} V")
    s2.metric("V_charge", f"{res['Vout']:.1f} V")
    s3.metric("f0", f"{td['f0']/1000:.1f} kHz")
    s4.metric("Efficiency", f"{res['efficiency']:.1f} %")
    
    s5, s6, s7, s8 = st.columns(4)
    s5.metric("Topology", td['topology'])
    s6.metric("R_Leq", f"{res['RLeq']:.2f} Ω")
    s7.metric("Pout", f"{res['Pout_actual']:.1f} W")
    s8.metric("Iout", f"{res['Iout']:.2f} A")
    st.markdown('</div>', unsafe_allow_html=True)

    # [Section 2]
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.subheader(t("🧲 코일 & 공진 네트워크 (Resonators)", "🧲 Coils & Resonant Network"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ltx", f"{td['Ltx']:.1f} uH")
    c2.metric("Lrx", f"{td['Lrx']:.1f} uH")
    c3.metric("k", f"{td['k']:.3f}")
    c4.metric("M", f"{res['M']*1e6:.1f} uH")
    
    c5, c6 = st.columns(2)
    c5.metric("Itx_rms", f"{res['Itx']:.2f} A")
    c6.metric("Irx_rms", f"{res['Irx']:.2f} A")
    
    st.divider()
    cap_cols = st.columns(len(res['caps']) + (1 if td['topology'] == 'LCC-S' else (2 if 'Double' in td['topology'] else 0)))
    idx = 0
    for name, data in res['caps'].items(): cap_cols[idx].metric(name, f"{data['val']*1e9:.2f} nF"); idx += 1
    if td['topology'] == "LCC-S": cap_cols[idx].metric("Ls", f"{res['Ls']*1e6:.1f} uH")
    elif 'Double' in td['topology']: cap_cols[idx].metric("Llcc_tx", f"{res['Llcc_tx']*1e6:.1f} uH"); cap_cols[idx+1].metric("Llcc_rx", f"{res['Llcc_rx']*1e6:.1f} uH")
    st.markdown('</div>', unsafe_allow_html=True)

    # [Section 3]
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.subheader(t("🧵 리쯔와이어 계산기 (Litz Wire Sizing)", "🧵 Litz Wire Calculator"))
    req_tx_sq = res['Itx'] / 7.0; req_rx_sq = res['Irx'] / 7.0
    st.info(t(f"💡 권장 와이어 굵기: Tx $\ge$ {req_tx_sq:.2f} $mm^2$, Rx $\ge$ {req_rx_sq:.2f} $mm^2$ (기준: 7A/$mm^2$)", f"💡 Required Wire: Tx $\ge$ {req_tx_sq:.2f} $mm^2$, Rx $\ge$ {req_rx_sq:.2f} $mm^2$"))
    w1, w2, w3 = st.columns(3)
    d_strand = w1.number_input(t("가닥 지름 (mm)", "Strand Dia. (mm)"), value=0.10, step=0.01)
    n_strand = w2.number_input(t("가닥 수 (N)", "Number of Strands"), value=600, step=10)
    calc_sq = ((d_strand / 2)**2) * math.pi * n_strand
    w3.metric(t("계산된 Litz 단면적", "Calculated SQ"), f"{calc_sq:.2f} mm²")
    if calc_sq >= max(req_tx_sq, req_rx_sq): st.success("✅ " + t("발열 규격 만족", "Meets thermal requirements"))
    else: st.error("⚠️ " + t("스펙 부족. 가닥 수(N)를 늘리세요.", "Insufficient spec. Increase strands."))
    st.markdown('</div>', unsafe_allow_html=True)

    # [Section 4] 주파수 응답 시뮬레이션
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.subheader(t("📈 주파수 응답 시뮬레이션 (Frequency Response)", "📈 Frequency Response Simulation"))
    
    # 1. 데이터를 먼저 생성하여 최대/최소 효율 파악
    df_f = simulate_frequency_response(td['topology'], res, td['f0'], td['Ltx']*1e-6, td['Lrx']*1e-6, res['M'], 0.085, 0.074)
    
    # 2. 그래프 형태가 잘 보이도록 초기 Y축 자동 스케일링 (최대 효율 기준 -15% ~ +5%)
    eff_max = df_f['Efficiency (%)'].max()
    default_ymax = min(100.0, math.ceil(eff_max + 5.0))
    default_ymin = max(0.0, math.floor(eff_max - 15.0))
    
    # 3. 슬라이더 대신 숫자 입력창(number_input) 배치
    col_min, col_max = st.columns(2)
    y_min = col_min.number_input(t("그래프 Y축 최소값 (%)", "Y-Axis Min (%)"), min_value=0.0, max_value=100.0, value=float(default_ymin), step=1.0)
    y_max = col_max.number_input(t("그래프 Y축 최대값 (%)", "Y-Axis Max (%)"), min_value=0.0, max_value=100.0, value=float(default_ymax), step=1.0)
    
    if y_min >= y_max:  # 에러 방지용
        y_max = min(100.0, y_min + 1.0)

    # 4. 차트 렌더링
    chart = alt.Chart(df_f).mark_line(color='#0A84FF').encode(
        x=alt.X('Frequency (kHz)', scale=alt.Scale(zero=False)),
        y=alt.Y('Efficiency (%)', scale=alt.Scale(domain=[y_min, y_max], clamp=True)),
        tooltip=['Frequency (kHz)', 'Efficiency (%)', 'Output Power (W)']
    ).interactive()
    
    st.altair_chart(chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # [Section 5] CSV 다운로드
    csv_dict = {
        "Item": ["Topology", "Pout (W)", "Vin (V)", "Vout (V)", "Iout (A)", "f0 (kHz)", "Eff (%)", "Ltx (uH)", "Lrx (uH)", "k", "M (uH)", "Itx (A)", "Irx (A)"],
        "Value": [td['topology'], res['Pout_actual'], td['Vin'], res['Vout'], res['Iout'], td['f0']/1000, res['efficiency'], td['Ltx'], td['Lrx'], td['k'], res['M']*1e6, res['Itx'], res['Irx']]
    }
    for name, data in res['caps'].items(): csv_dict["Item"].append(name); csv_dict["Value"].append(f"{data['val']*1e9:.2f} nF")
    csv_data = pd.DataFrame(csv_dict).to_csv(index=False).encode('utf-8-sig')
    
    n1, n2, n3 = st.columns([1, 1, 2])
    n1.button(t("⬅️ 설계 수정 (Step 3)", "⬅️ Back (Step 3)"), on_click=go_to_step, args=(3 if st.session_state.mode == 'Manual' else 2,))
    n2.button(t("🏠 홈으로", "🏠 Home"), on_click=reset_project)
    n3.download_button(t("📊 전체 리포트 다운로드 (CSV)", "📊 Download Full Report (CSV)"), csv_data, "WPT_Full_Report.csv", "text/csv", use_container_width=True, type="primary")
