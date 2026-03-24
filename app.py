import streamlit as st
import numpy as np
import math
import pandas as pd

# --- [Math Engine] WPT 파라미터, 전압 스트레스 및 효율 계산 함수 ---
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
    
    # Peak 전압(내압) 스트레스 계산
    V_Ltx_peak = math.sqrt(2) * w * Ltx * Itx
    V_Cp_peak = math.sqrt(2) * w * Ls * Itx
    V_Cs_peak = math.sqrt(2) * Itx / (w * Cs)
    V_Crx_peak = math.sqrt(2) * Irx / (w * Crx)

    # [추가됨] 효율 및 손실 계산 (구리 손실 위주)
    P_out_ac = (Irx**2) * RLeq
    P_loss_tx = (Itx**2) * Rtx
    P_loss_rx = (Irx**2) * Rrx
    P_in_ac = P_out_ac + P_loss_tx + P_loss_rx
    efficiency = (P_out_ac / P_in_ac) * 100
    
    return {
        "Itx": Itx, "Irx": Irx, "M": M,
        "Ls": Ls, "Cp": Cp, "Cs": Cs, "Crx": Crx, "Ldc_min": Ldc_min,
        "V_Ltx_peak": V_Ltx_peak, "V_Cp_peak": V_Cp_peak, "V_Cs_peak": V_Cs_peak, "V_Crx_peak": V_Crx_peak,
        "P_out_ac": P_out_ac, "P_loss_tx": P_loss_tx, "P_loss_rx": P_loss_rx, "efficiency": efficiency
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

    # Peak 전압(내압) 스트레스 계산
    V_Ltx_peak = math.sqrt(2) * w * Ltx * Itx
    V_parallel_tx_peak = math.sqrt(2) * w * Llcc_tx * Itx
    V_series_tx_peak = math.sqrt(2) * Itx / (w * Cp_tx)
    V_Lrx_peak = math.sqrt(2) * w * Lrx * Irx
    V_parallel_rx_peak = math.sqrt(2) * w * Llcc_rx * Irx
    V_series_rx_peak = math.sqrt(2) * Irx / (w * Cp_rx)

    # [추가됨] 효율 및 손실 계산
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
        "P_out_ac": P_out_ac, "P_loss_tx": P_loss_tx, "P_loss_rx": P_loss_rx, "efficiency": efficiency
    }

# --- [Web UI] 프론트엔드 화면 구성 ---
st.set_page_config(page_title="WPT Design Agent", layout="wide")
st.title("⚡ 무선전력전송(WPT) 자동 설계 & 효율 분석 에이전트")

with st.sidebar:
    st.header("1. 시스템 스펙 입력")
    Vin = st.number_input("입력 전압 Vin (V)", min_value=10.0, value=100.0, step=10.0)
    Vout = st.number_input("목표 출력 전압 Vout (V)", min_value=10.0, value=200.0, step=10.0)
    Pout = st.number_input("목표 출력 전력 Pout (W)", min_value=10.0, value=300.0, step=50.0)
    f0_khz = st.number_input("스위칭 주파수 (kHz)", min_value=10.0, value=85.0, step=1.0)
    f0 = f0_khz * 1000

    st.header("2. 토폴로지 선택")
    topology = st.selectbox("적용할 구조를 선택하세요", ["LCC-S (수신부 초경량화)", "Double LCC (고효율/CC충전)"])

    st.header("3. 코일 파라미터 튜닝")
    Ltx_uH = st.slider("송신 코일 Ltx (μH)", 10.0, 150.0, 80.0, step=1.0)
    Lrx_uH = st.slider("수신 코일 Lrx (μH)", 10.0, 150.0, 80.0, step=1.0)
    k = st.slider("결합 계수 k", 0.05, 0.50, 0.196, step=0.001)
    
    st.header("4. 코일 저항 입력 (효율 분석용)")
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
st.header("📊 통합 설계 결과 리포트")

if topology == "LCC-S (수신부 초경량화)":
    res = calculate_lccs(Vin, Vout, Pout, f0, Ltx, Lrx, k, Rtx, Rrx)
else:
    res = calculate_double_lcc(Vin, Vout, Pout, f0, Ltx, Lrx, k, current_ratio, Rtx, Rrx)

if "error" in res:
    st.error(f"🚨 설계 오류: {res['error']}")
else:
    # 1. 효율 분석 대시보드
    st.subheader("AC-AC 전송 효율 및 발열 분석")
    e1, e2 = st.columns([1, 2])
    
    with e1:
        eff_color = "normal" if res['efficiency'] >= 90 else "off"
        st.metric("예상 AC-AC 효율", f"{res['efficiency']:.1f} %", delta="목표치(90%) 달성" if res['efficiency'] >= 90 else "효율 개선 필요", delta_color=eff_color)
        st.write(f"- 송신 코일(Tx) 발열량: **{res['P_loss_tx']:.1f} W**")
        st.write(f"- 수신 코일(Rx) 발열량: **{res['P_loss_rx']:.1f} W**")
        
    with e2:
        # 손실 분포 차트
        loss_data = pd.DataFrame({
            "항목": ["전달된 전력 (AC Output)", "송신 코일 손실", "수신 코일 손실"],
            "전력 (W)": [res['P_out_ac'], res['P_loss_tx'], res['P_loss_rx']]
        })
        st.bar_chart(loss_data.set_index("항목"), height=200)

    st.divider()

    # 2. 마그네틱 코일 상태
    st.subheader("마그네틱 (Coil) 파라미터")
    col1, col2, col3 = st.columns(3)
    col1.metric("상호 인덕턴스 (M)", f"{res['M']*1e6:.2f} μH")
    col2.metric("송신 코일 전류 (Itx)", f"{res['Itx']:.2f} A", delta="경고: 30A 이상 발열 주의" if res['Itx'] >= 30 else "안정", delta_color="inverse" if res['Itx'] >= 30 else "normal")
    col3.metric("수신 코일 전류 (Irx)", f"{res['Irx']:.2f} A")

    st.divider()

    # 3. 보상 소자 및 전압 스트레스
    st.subheader("보상 소자값 및 내압(Voltage Stress) 리포트")
    def display_voltage_metric(label, capacitance, voltage):
        st.metric(label, f"{capacitance*1e9:.2f} nF", f"Peak: {voltage:.0f} V", delta_color="off")

    if topology == "LCC-S (수신부 초경량화)":
        c1, c2, c3 = st.columns(3)
        with c1:
            display_voltage_metric("Tx 병렬 커패시터 (Cp)", res['Cp'], res['V_Cp_peak'])
        with c2:
            display_voltage_metric("Tx 직렬 커패시터 (Cs)", res['Cs'], res['V_Cs_peak'])
        with c3:
            display_voltage_metric("Rx 직렬 커패시터 (Crx)", res['Crx'], res['V_Crx_peak'])
            
        st.info(f"💡 정류기 후단 LC 필터 최소 요구치: **Ldc ≥ {res['Ldc_min']*1e6:.2f} μH**")

    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            display_voltage_metric("Tx 병렬 커패시터", res['Clcc_tx'], res['V_parallel_tx_peak'])
        with c2:
            display_voltage_metric("Tx 직렬 커패시터", res['Cp_tx'], res['V_series_tx_peak'])
        with c3:
            display_voltage_metric("Rx 병렬 커패시터", res['Clcc_rx'], res['V_parallel_rx_peak'])
        with c4:
            display_voltage_metric("Rx 직렬 커패시터", res['Cp_rx'], res['V_series_rx_peak'])

# ---------------------------------------------------------
    # 4. 데이터 내보내기 (Export to CSV)
    st.divider()
    st.subheader("📥 설계 데이터 다운로드")
    
    # [오류 수정됨] "분류" 리스트의 개수를 13개로 정상화했습니다.
    export_data = {
        "분류": ["시스템", "시스템", "시스템", "시스템", "코일", "코일", "코일", "코일", "코일", "코일", "효율", "효율", "효율"],
        "파라미터 항목": ["입력 전압 (Vin)", "출력 전압 (Vout)", "출력 전력 (Pout)", "주파수 (f0)", "송신 코일 (Ltx)", "수신 코일 (Lrx)", "상호 인덕턴스 (M)", "결합 계수 (k)", "송신 코일 전류 (Itx)", "수신 코일 전류 (Irx)", "예상 전송 효율", "Tx 발열량", "Rx 발열량"],
        "설계 값": [Vin, Vout, Pout, f0_khz, Ltx*1e6, Lrx*1e6, res['M']*1e6, k, res['Itx'], res['Irx'], res['efficiency'], res['P_loss_tx'], res['P_loss_rx']],
        "단위": ["V", "V", "W", "kHz", "μH", "μH", "μH", "", "A", "A", "%", "W", "W"]
    }
    
    # 토폴로지에 따른 보상 소자 데이터 추가
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
    
    # 엑셀에서 한글이 깨지지 않도록 utf-8-sig 로 인코딩
    csv_data = df_export.to_csv(index=False).encode('utf-8-sig')
    
    st.download_button(
        label="📊 설계 파라미터 엑셀(CSV) 다운로드",
        data=csv_data,
        file_name="wpt_design_parameters.csv",
        mime="text/csv",
    )
    
    st.caption("※ 다운로드하신 CSV 파일은 엑셀에서 바로 열어보시거나, 한글이나 워드 문서의 표로 복사하여 사용하실 수 있습니다.")
