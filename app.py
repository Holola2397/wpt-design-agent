import streamlit as st
import numpy as np
import math
import pandas as pd

# --- [Math Engine] 파라미터 설계 및 단일 주파수 효율 계산 ---
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
    if Ls >= Ltx:
        return {"error": "Ls가 Ltx보다 큽니다. Ltx나 M을 더 키우거나 Vin을 높이세요."}
        
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
    efficiency = (P_out_ac / P_in_ac) * 100
    
    return {
        "Itx": Itx, "Irx": Irx, "M": M,
        "Ls": Ls, "Cp": Cp, "Cs": Cs, "Crx": Crx, "Ldc_min": Ldc_min,
        "V_Ltx_peak": V_Ltx_peak, "V_Cp_peak": V_Cp_peak, "V_Cs_peak": V_Cs_peak, "V_Crx_peak": V_Crx_peak,
        "P_out_ac": P_out_ac, "P_loss_tx": P_loss_tx, "P_loss_rx": P_loss_rx, "efficiency": efficiency,
        "RLeq": RLeq, "Vin_ac": Vin_ac_rms
    }

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
    
    if Llcc_tx >= Ltx or Llcc_rx >= Lrx:
        return {"error": "보상 인덕터(Llcc)가 메인 코일보다 커서 물리적으로 구현 불가합니다."}
        
    Clcc_tx = 1 / (w**2 * Llcc_tx)
    Clcc_rx = 1 / (w**2 * Llcc_rx)
    Cp_tx = 1 / (w**2 * (Ltx - Llcc_tx))
    Cp_rx = 1 / (w**2 * (Lrx - Llcc_rx))
    
    Itx = Vin_ac_rms / (w * Llcc_tx)
    Irx = Vrect_in / (w * Llcc_rx)

    V_Ltx_peak = math.sqrt(2) * w * Ltx * Itx
    V_parallel_tx_peak = math.sqrt(2) * w * Llcc_tx * Itx
    V_series_tx_peak = math.sqrt(2) * Itx / (w * Cp_tx)
    V_Lrx_peak = math.sqrt(2) * w * Lrx * Irx
    V_parallel_rx_peak = math.sqrt(2) * w * Llcc_rx * Irx
    V_series_rx_peak = math.sqrt(2) * Irx / (w * Cp_rx)

    P_out_ac = (Irx**2) * RLeq
    P_loss_tx = (Itx**2) * Rtx
    P_loss_rx = (Irx**2) * Rrx
    P_in_ac = P_out_ac + P_loss_tx + P_loss_rx
    efficiency = (P_out_ac / P_in_ac) * 100
    
    return {
        "Itx": Itx, "Irx": Irx, "M": M,
        "Llcc_tx": Llcc_tx, "Llcc_rx": Llcc_rx, "Clcc_tx": Clcc_tx, "Clcc_rx": Clcc_rx, "Cp_tx": Cp_tx, "Cp_rx": Cp_rx,
        "V_Ltx_peak": V_Ltx_peak, "V_parallel_tx_peak": V_parallel_tx_peak, "V_series_tx_peak": V_series_tx_peak,
        "V_Lrx_peak": V_Lrx_peak, "V_parallel_rx_peak": V_parallel_rx_peak, "V_series_rx_peak": V_series_rx_peak,
        "P_out_ac": P_out_ac, "P_loss_tx": P_loss_tx, "P_loss_rx": P_loss_rx, "efficiency": efficiency,
        "RLeq": RLeq, "Vin_ac": Vin_ac_rms
    }

# --- [AC Sweep Engine] 주파수 응답 복소 임피던스 시뮬레이션 ---
def simulate_frequency_response(topology, res_dict, f0, Ltx, Lrx, M, Rtx, Rrx):
    # 75kHz ~ 95kHz 구간 200개 포인트 샘플링
    f_arr = np.linspace(75e3, 95e3, 200)
    w_arr = 2 * np.pi * f_arr
    
    Vin_ac = res_dict["Vin_ac"]
    RLeq = res_dict["RLeq"]
    
    if topology == "LCC-S (수신부 초경량화)":
        Ls, Cp, Cs, Crx = res_dict['Ls'], res_dict['Cp'], res_dict['Cs'], res_dict['Crx']
        
        # Vectorized Complex Impedance Math
        Z_rx = RLeq + Rrx + 1j*w_arr*Lrx + 1/(1j*w_arr*Crx)
        Z_refl = (w_arr * M)**2 / Z_rx
        Z_tx_branch = Rtx + 1j*w_arr*Ltx + 1/(1j*w_arr*Cs) + Z_refl
        Z_p = (1/(1j*w_arr*Cp)) * Z_tx_branch / ((1/(1j*w_arr*Cp)) + Z_tx_branch)
        Z_in = 1j*w_arr*Ls + Z_p
        
        I_in = Vin_ac / Z_in
        V_Cp = I_in * Z_p
        I_tx = V_Cp / Z_tx_branch
        I_rx = (1j * w_arr * M * I_tx) / Z_rx
        
        P_out_arr = (np.abs(I_rx)**2) * RLeq
        P_in_arr = np.real(Vin_ac * np.conj(I_in))
        
    else: # Double LCC
        Llcc_tx, Clcc_tx, Cp_tx = res_dict['Llcc_tx'], res_dict['Clcc_tx'], res_dict['Cp_tx']
        Llcc_rx, Clcc_rx, Cp_rx = res_dict['Llcc_rx'], res_dict['Clcc_rx'], res_dict['Cp_rx']
        
        # Vectorized Complex Impedance Math
        Z_load_branch = RLeq + 1j*w_arr*Llcc_rx + 1/(1j*w_arr*Clcc_rx)
        Z_p_rx = (1/(1j*w_arr*Cp_rx)) * Z_load_branch / ((1/(1j*w_arr*Cp_rx)) + Z_load_branch)
        Z_rx_total = Rrx + 1j*w_arr*Lrx + Z_p_rx
        
        Z_refl = (w_arr * M)**2 / Z_rx_total
        Z_tx_main = Rtx + 1j*w_arr*Ltx + Z_refl
        Z_p_tx = (1/(1j*w_arr*Cp_tx)) * Z_tx_main / ((1/(1j*w_arr*Cp_tx)) + Z_tx_main)
        Z_in = 1j*w_arr*Llcc_tx + 1/(1j*w_arr*Clcc_tx) + Z_p_tx
        
        I_in = Vin_ac / Z_in
        V_Cp_tx = I_in * Z_p_tx
        I_tx = V_Cp_tx / Z_tx_main
        
        V_rx_ind = 1j * w_arr * M * I_tx
        I_rx_main = V_rx_ind / Z_rx_total
        V_Cp_rx = I_rx_main * Z_p_rx
        I_out_ac = V_Cp_rx / Z_load_branch
        
        P_out_arr = (np.abs(I_out_ac)**2) * RLeq
        P_in_arr = np.real(Vin_ac * np.conj(I_in))

    eff_arr = np.where(P_in_arr > 0, (P_out_arr / P_in_arr) * 100, 0)
    
    # DataFrame으로 묶어 반환
    df_freq = pd.DataFrame({
        "Frequency (kHz)": f_arr / 1000,
        "Output Power (W)": P_out_arr,
        "Efficiency (%)": eff_arr
    }).set_index("Frequency (kHz)")
    
    return df_freq

# --- [Web UI] 프론트엔드 화면 구성 ---
st.set_page_config(page_title="WPT Design & Simulator", layout="wide")
st.title("⚡ 무선전력전송(WPT) 자동 설계 & 시뮬레이터")

with st.sidebar:
    st.header("1. 시스템 스펙 입력")
    Vin = st.number_input("입력 전압 Vin (V)", min_value=10.0, value=100.0, step=10.0)
    Vout = st.number_input("목표 출력 전압 Vout (V)", min_value=10.0, value=200.0, step=10.0)
    Pout = st.number_input("목표 출력 전력 Pout (W)", min_value=10.0, value=300.0, step=50.0)
    f0_khz = st.number_input("중심 주파수 (kHz)", min_value=10.0, value=85.0, step=1.0)
    f0 = f0_khz * 1000

    st.header("2. 토폴로지 선택")
    topology = st.selectbox("적용할 구조를 선택하세요", ["LCC-S (수신부 초경량화)", "Double LCC (고효율/CC충전)"])

    st.header("3. 코일 파라미터 튜닝")
    Ltx_uH = st.slider("송신 코일 Ltx (μH)", 10.0, 150.0, 80.0, step=1.0)
    Lrx_uH = st.slider("수신 코일 Lrx (μH)", 10.0, 150.0, 80.0, step=1.0)
    k = st.slider("결합 계수 k (이격 거리 반영)", 0.05, 0.50, 0.196, step=0.001)
    
    st.header("4. 코일 저항 입력")
    Rtx_m = st.number_input("송신 코일 저항 Rtx (mΩ)", min_value=1.0, value=85.0, step=1.0)
    Rrx_m = st.number_input("수신 코일 저항 Rrx (mΩ)", min_value=1.0, value=74.0, step=1.0)
    Rtx = Rtx_m * 1e-3
    Rrx = Rrx_m * 1e-3
    
    current_ratio = 1.0
    if topology == "Double LCC (고효율/CC충전)":
        current_ratio = st.slider("전류 비율 (Itx / Irx)", 0.5, 3.0, 1.5, step=0.1)

    Ltx = Ltx_uH * 1e-6
    Lrx = Lrx_uH * 1e-6

# 메인 화면 결과 출력
if topology == "LCC-S (수신부 초경량화)":
    res = calculate_lccs(Vin, Vout, Pout, f0, Ltx, Lrx, k, Rtx, Rrx)
else:
    res = calculate_double_lcc(Vin, Vout, Pout, f0, Ltx, Lrx, k, current_ratio, Rtx, Rrx)

if "error" in res:
    st.error(f"🚨 설계 오류: {res['error']}")
else:
    # 1. 효율 분석 대시보드
    st.subheader("📊 설계 결과 (Center Frequency: 85kHz 기준)")
    e1, e2, e3 = st.columns(3)
    e1.metric("송신 코일 전류 (Itx)", f"{res['Itx']:.2f} A", delta="경고: 발열 주의" if res['Itx'] >= 30 else "안정", delta_color="inverse" if res['Itx'] >= 30 else "normal")
    e2.metric("수신 코일 전류 (Irx)", f"{res['Irx']:.2f} A")
    e3.metric("AC-AC 전송 효율", f"{res['efficiency']:.1f} %")

    # [NEW] 2. 주파수 응답 시뮬레이션
    st.divider()
    st.subheader("📈 AC 주파수 응답 (Frequency Response) 해석")
    st.markdown("85kHz 중심 주파수 대역(75kHz~95kHz)에서의 전력 전달 및 효율 특성입니다. **결합 계수(k)** 슬라이더를 조절하여 오정렬(Misalignment) 발생 시의 특성 변화(Bifurcation 등)를 관찰해 보세요.")
    
    # 시뮬레이션 데이터 생성
    df_freq = simulate_frequency_response(topology, res, f0, Ltx, Lrx, res['M'], Rtx, Rrx)
    
    tab1, tab2 = st.tabs(["출력 전력 (Output Power)", "전송 효율 (Efficiency)"])
    with tab1:
        st.line_chart(df_freq["Output Power (W)"], height=300)
    with tab2:
        st.line_chart(df_freq["Efficiency (%)"], height=300)

    st.divider()

    # 3. 데이터 다운로드
    st.subheader("📥 설계 데이터 다운로드")
    
    export_data = {
        "분류": ["시스템", "시스템", "시스템", "시스템", "코일", "코일", "코일", "코일", "코일", "코일", "효율", "효율", "효율"],
        "파라미터 항목": ["입력 전압 (Vin)", "출력 전압 (Vout)", "출력 전력 (Pout)", "주파수 (f0)", "송신 코일 (Ltx)", "수신 코일 (Lrx)", "상호 인덕턴스 (M)", "결합 계수 (k)", "송신 코일 전류 (Itx)", "수신 코일 전류 (Irx)", "예상 전송 효율", "Tx 발열량", "Rx 발열량"],
        "설계 값": [Vin, Vout, Pout, f0_khz, Ltx*1e6, Lrx*1e6, res['M']*1e6, k, res['Itx'], res['Irx'], res['efficiency'], res['P_loss_tx'], res['P_loss_rx']],
        "단위": ["V", "V", "W", "kHz", "μH", "μH", "μH", "", "A", "A", "%", "W", "W"]
    }
    
    if topology == "LCC-S (수신부 초경량화)":
        export_data["분류"].extend(["보상소자", "보상소자", "보상소자", "보상소자", "필터"])
        export_data["파라미터 항목"].extend(["Tx 직렬 인덕터 (Ls)", "Tx 병렬 커패시터 (Cp)", "Tx 직렬 커패시터 (Cs)", "Rx 직렬 커패시터 (Crx)", "최소 DC 인덕터 (Ldc_min)"])
        export_data["설계 값"].extend([res['Ls']*1e6, res['Cp']*1e9, res['Cs']*1e9, res['Crx']*1e9, res['Ldc_min']*1e6])
        export_data["단위"].extend(["μH", "nF", "nF", "nF", "μH"])
    else:
        export_data["분류"].extend(["보상소자", "보상소자", "보상소자", "보상소자", "보상소자", "보상소자"])
        export_data["파라미터 항목"].extend(["Tx 직렬 인덕터 (Llcc_tx)", "Rx 직렬 인덕터 (Llcc_rx)", "Tx 직렬 커패시터 (Clcc_tx)", "Rx 직렬 커패시터 (Clcc_rx)", "Tx 병렬 커패시터 (Cp_tx)", "Rx 병렬 커패시터 (Cp_rx)"])
        export_data["설계 값"].extend([res['Llcc_tx']*1e6, res['Llcc_rx']*1e6, res['Clcc_tx']*1e9, res['Clcc_rx']*1e9, res['Cp_tx']*1e9, res['Cp_rx']*1e9])
        export_data["단위"].extend(["μH", "μH", "nF", "nF", "nF", "nF"])

    df_export = pd.DataFrame(export_data)
    csv_data = df_export.to_csv(index=False).encode('utf-8-sig')
    
    st.download_button(
        label="📊 설계 파라미터 엑셀(CSV) 다운로드",
        data=csv_data,
        file_name="wpt_design_parameters_with_sim.csv",
        mime="text/csv",
    )
