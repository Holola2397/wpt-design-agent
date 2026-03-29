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
    /* 버튼 스타일 통일 */
    .stButton>button {
        border-radius: 10px !important; font-weight: 600 !important; transition: all 0.2s ease;
    }
    .main-next-btn>div>button { background-color: #0A84FF !important; color: white !important; }
    .main-next-btn>div>button:hover { background-color: #0071E3 !important; transform: scale(1.02); }
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

# ==========================================
# [Math Engine] 핵심 설계 수식 모듈
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
        if Ls >= Ltx: return {"error": "Ls > Ltx (보상 불가)"}
        Cp = 1 / (w**2 * Ls); Cs = 1 / (w**2 * (Ltx - Ls)); Crx = 1 / (w**2 * Lrx)
        P_out_ac = (Irx**2) * RLeq; P_loss_tx = (Itx**2) * Rtx; P_loss_rx = (Irx**2) * Rrx
        eff = (P_out_ac / (P_out_ac + P_loss_tx + P_loss_rx)) * 100
        return {"Itx": Itx, "Irx": Irx, "M": M, "Ls": Ls, "Cp": Cp, "Cs": Cs, "Crx": Crx, "efficiency": eff, "Vin_ac": Vin_ac_rms, "RLeq": RLeq}
    except: return {"error": "계산 불능"}

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
        Llcc_tx = math.sqrt(L_prod / L_rat); Llcc_rx = math.sqrt(L_prod * L_rat)
        if Llcc_tx >= Ltx or Llcc_rx >= Lrx: return {"error": "Llcc > Ltx/Lrx (설계불가)"}
        Clcc_tx = 1 / (w**2 * Llcc_tx); Clcc_rx = 1 / (w**2 * Llcc_rx)
        Cp_tx = 1 / (w**2 * (Ltx - Llcc_tx)); Cp_rx = 1 / (w**2 * (Lrx - Llcc_rx))
        Itx = Vin_ac_rms / (w * Llcc_tx); Irx = Vrect_in / (w * Llcc_rx)
        P_out_ac = (Irx**2) * RLeq; P_loss_tx = (Itx**2) * Rtx; P_loss_rx = (Irx**2) * Rrx
        eff = (P_out_ac / (P_out_ac + P_loss_tx + P_loss_rx)) * 100
        return {"Itx": Itx, "Irx": Irx, "M": M, "Llcc_tx": Llcc_tx, "Llcc_rx": Llcc_rx, "Clcc_tx": Clcc_tx, "Clcc_rx": Clcc_rx, "Cp_tx": Cp_tx, "Cp_rx": Cp_rx, "efficiency": eff, "Vin_ac": Vin_ac_rms, "RLeq": RLeq}
    except: return {"error": "계산 불능"}

def simulate_frequency_response(topology, res_dict, f0, Ltx, Lrx, M, Rtx, Rrx):
    f_arr = np.linspace(f0 - 15e3, f0 + 15e3, 200)
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
# [Sidebar] API & Model Selection
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
    if not available_models: available_models = ["models/gemini-1.5-flash"]
    selected_model = st.selectbox("LLM 엔진 선택", available_models, index=0)
    st.divider()
    if st.button("🏠 플랫폼 홈으로", use_container_width=True):
        reset_project(); st.rerun()

if st.session_state.step > 0:
    st.progress(st.session_state.step / 5.0, text=f"Step {st.session_state.step} / 5 진행 중...")

# ==========================================
# [Phase 0] Entry
# ==========================================
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

# ==========================================
# [Phase 1] 요구사항 입력
# ==========================================
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
    
    # 내비게이션 버튼 바
    n1, n2, n3 = st.columns([1, 1, 2])
    n1.button("⬅️ 홈으로", on_click=reset_project)
    if n3.button("제약 조건 확정 및 분석 시작 ➔", use_container_width=True, type="primary"):
        st.session_state.project_data = {"app_type": app_type, "battery_vol": batt_vol, "target_power": target_p, "rx_weight": rx_w, "air_gap": gap, "tx_size": tx_s, "rx_size": rx_s, "battery_info": f"{batt_s}S {batt_t}"}
        go_to_step(2); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# [Phase 2] AI 추천
# ==========================================
elif st.session_state.step == 2:
    st.header("Step 2. AI 기반 최적 토폴로지 추천")
    if not api_key: 
        st.error("사이드바에 API Key를 입력해주세요.")
        st.button("⬅️ Step 1로 돌아가기", on_click=go_to_step, args=(1,))
    else:
        sd = st.session_state.project_data
        if st.session_state.llm_result is None:
            with st.status("🧠 AI 수석 엔지니어 분석 중...", expanded=True) as status:
                try:
                    model = genai.GenerativeModel(selected_model)
                    prompt = f"WPT Engineer: App {sd['app_type']}, Power {sd['target_power']}W, RxWeight {sd['rx_weight']}g. Recommend 'LCC-S' or 'Double LCC' in JSON only: {{\"topology\": \"string\", \"reasoning\": \"string\", \"recommended_vin\": int, \"recommended_f0\": 85, \"estimated_m\": float}}"
                    resp = model.generate_content(prompt, request_options={"timeout": 15.0})
                    st.session_state.llm_result = json.loads(resp.text.replace('```json', '').replace('```', '').strip())
                    status.update(label="✅ 분석 완료", state="complete")
                except:
                    topo = "LCC-S" if sd['rx_weight'] <= 500 else "Double LCC"
                    st.session_state.llm_result = {"topology": topo, "reasoning": "서버 응답 지연으로 내부 알고리즘이 대신 선정하였습니다.", "recommended_vin": 100, "recommended_f0": 85, "estimated_m": 15.0}
        
        res = st.session_state.llm_result
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.subheader(f"✅ 추천 토폴로지: **{res['topology']}**")
        st.info(res['reasoning'])
        cp = estimate_coil_params(res['estimated_m'], sd['air_gap'], sd['rx_weight'])
        st.session_state.tuning_data = {"topology": res['topology'], "Vin": float(res['recommended_vin']), "f0": res['recommended_f0']*1000, **cp, "Rtx": 0.085, "Rrx": 0.074}
        
        st.write("<br>", unsafe_allow_html=True)
        n1, n2, n3 = st.columns([1, 1, 2])
        n1.button("⬅️ 이전 (Step 1)", on_click=go_to_step, args=(1,))
        n2.button("🏠 홈", on_click=reset_project)
        next_step = 4 if st.session_state.mode == 'Auto' else 3
        if n3.button("다음 단계로 ➔", use_container_width=True, type="primary"):
            go_to_step(next_step); st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# [Phase 3] 상세 튜닝
# ==========================================
elif st.session_state.step == 3:
    st.header("Step 3. 파라미터 상세 튜닝 (Expert)")
    td = st.session_state.tuning_data; sd = st.session_state.project_data
    
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.subheader("🛠️ 설계 파라미터")
        Ltx = st.slider("송신 코일 Ltx (uH)", 10.0, 300.0, float(td['Ltx']))
        Lrx = st.slider("수신 코일 Lrx (uH)", 10.0, 150.0, float(td['Lrx']))
        k = st.slider("결합 계수 k", 0.05, 0.5, float(td['k']))
        vin = st.number_input("입력 전압 Vin (V)", value=td['Vin'])
        f0 = st.number_input("주파수 f0 (kHz)", value=td['f0']/1000) * 1000
        ratio = st.slider("전류 비율 (Itx/Irx)", 0.5, 3.0, 1.5) if "Double" in td['topology'] else 1.0
        
    with col2:
        st.subheader("⚡ 실시간 설계 프리뷰")
        if "LCC-S" in td['topology']: res = calculate_lccs(vin, sd['battery_vol'], sd['target_power'], f0, Ltx*1e-6, Lrx*1e-6, k, 0.085, 0.074)
        else: res = calculate_double_lcc(vin, sd['battery_vol'], sd['target_power'], f0, Ltx*1e-6, Lrx*1e-6, k, ratio, 0.085, 0.074)
        
        if "error" in res: st.error(res['error'])
        else:
            p1, p2, p3 = st.columns(3)
            p1.metric("M (uH)", f"{res['M']*1e6:.2f}")
            p2.metric("예상 효율", f"{res['efficiency']:.1f}%")
            p3.metric("Itx (A)", f"{res['Itx']:.2f}")
            st.divider()
            st.write("**공진 커패시터:**")
            if "LCC-S" in td['topology']: st.write(f"Cp: {res['Cp']*1e9:.2f}nF | Cs: {res['Cs']*1e9:.2f}nF | Crx: {res['Crx']*1e9:.2f}nF")
            else: st.write(f"Tx(Clcc,Cp): {res['Clcc_tx']*1e9:.2f}n, {res['Cp_tx']*1e9:.2f}n | Rx: {res['Clcc_rx']*1e9:.2f}n, {res['Cp_rx']*1e9:.2f}n")
            
    st.write("<br>", unsafe_allow_html=True)
    n1, n2, n3 = st.columns([1, 1, 2])
    n1.button("⬅️ 이전 (Step 2)", on_click=go_to_step, args=(2,))
    n2.button("🏠 홈", on_click=reset_project)
    if n3.button("설계 확정 및 리포트 생성 ➔", use_container_width=True, type="primary"):
        st.session_state.tuning_data.update({"Ltx": Ltx, "Lrx": Lrx, "k": k, "Vin": vin, "f0": f0, "ratio": ratio})
        go_to_step(4); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# [Phase 4 & 5] 리포트
# ==========================================
elif st.session_state.step == 4:
    st.header("Step 4 & 5. 통합 설계 리포트")
    td = st.session_state.tuning_data; sd = st.session_state.project_data
    
    if "LCC-S" in td['topology']: res = calculate_lccs(td['Vin'], sd['battery_vol'], sd['target_power'], td['f0'], td['Ltx']*1e-6, td['Lrx']*1e-6, td['k'], 0.085, 0.074)
    else: res = calculate_double_lcc(td['Vin'], sd['battery_vol'], sd['target_power'], td['f0'], td['Ltx']*1e-6, td['Lrx']*1e-6, td['k'], td.get('ratio', 1.5), 0.085, 0.074)
    
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.subheader("📋 설계 환경 요약")
    r1, r2, r3 = st.columns(3)
    r1.write(f"**App:** {sd['app_type']} / **목표:** {sd['target_power']}W")
    r2.write(f"**배터리:** {sd.get('battery_info')} ({sd['battery_vol']}V)")
    r3.write(f"**이격/무게:** {sd['air_gap']}mm / {sd['rx_weight']}g")
    
    st.divider()
    st.subheader("⚙️ 최종 설계 파라미터")
    d1, d2, d3 = st.columns(3)
    d1.metric("토폴로지", td['topology'])
    d2.metric("주파수", f"{td['f0']/1000}kHz")
    d3.metric("최종 효율", f"{res['efficiency']:.1f}%")
    
    st.divider()
    df_f = simulate_frequency_response(td['topology'], res, td['f0'], td['Ltx']*1e-6, td['Lrx']*1e-6, res['M'], 0.085, 0.074)
    st.altair_chart(alt.Chart(df_f).mark_line(color='#0A84FF').encode(x='Frequency (kHz)', y='Efficiency (%)').interactive(), use_container_width=True)
    
    csv = pd.DataFrame({"Item": ["App", "Power", "Vbatt", "Topo", "Eff"], "Val": [sd['app_type'], sd['target_power'], sd['battery_vol'], td['topology'], res['efficiency']]}).to_csv(index=False).encode('utf-8-sig')
    
    st.write("<br>", unsafe_allow_html=True)
    n1, n2, n3 = st.columns([1, 1, 2])
    n1.button("⬅️ 설계 수정 (Step 3)", on_click=go_to_step, args=(3 if st.session_state.mode == 'Manual' else 2,))
    n2.button("🏠 홈으로", on_click=reset_project)
    n3.download_button("📊 엑셀 리포트 다운로드", csv, "WPT_Report.csv", "text/csv", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)
