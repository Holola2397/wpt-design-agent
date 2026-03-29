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
            # [개선됨] 빙글빙글 도는 스피너 대신, 진행 상황을 텍스트로 보여주는 Status 창 적용
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
                    
                    st.write("⚙️ 2. 최적 토폴로지 및 파라미터 역산 중 (gemini-1.5-flash)...")
                    # 가장 빠르고 가벼운 flash 모델 사용
                    model = genai.GenerativeModel('gemini-1.5-flash')
                    
                    # [개선됨] 600초 무한 대기 방지: 15초 내에 답 안 오면 즉시 강제 종료(Timeout)
                    response = model.generate_content(
                        prompt, 
                        request_options={"timeout": 15.0}
                    )
                    
                    st.write("🗂️ 3. 결과 데이터 JSON 파싱 및 UI 바인딩 중...")
                    raw_text = response.text.replace('```json', '').replace('```', '').strip()
                    st.session_state.llm_result = json.loads(raw_text)
                    
                    # 완료 시 상태창을 닫고 성공 메시지로 변경
                    status.update(label="✅ 분석이 완료되었습니다!", state="complete", expanded=False)
                    
                except Exception as e:
                    status.update(label="⚠️ API 통신 지연 발생", state="error", expanded=True)
                    st.error(f"🚨 에러 상세 내용: {e}")
                    st.warning("구글 서버 응답이 지연되어 내부 휴리스틱 알고리즘으로 우회합니다.")
                    
                    topo = "LCC-S (수신부 초경량화)" if saved_data['rx_weight'] <= 500 else "Double LCC (고효율/CC충전)"
                    st.session_state.llm_result = {"topology": topo, "reasoning": "내부 알고리즘 평가 결과입니다.", "recommended_vin": 100, "recommended_f0": 85, "estimated_m": 15.0}
        
        # UI 출력부 (기존과 동일)
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
        col3.metric("자동 분배된 Ltx / Lrx", f"{coil_params['Ltx']} / {coil_params['Lrx']} μH")
        st.markdown('</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1: st.button("⬅️ 제약 조건 다시 입력하기", on_click=go_to_step, args=(1,), use_container_width=True)
        with c2:
            if st.session_state.mode == 'Auto':
                st.button("시뮬레이션 결과 리포트 생성 ➔", on_click=go_to_step, args=(4,), type="primary", use_container_width=True)
            else:
                st.button("엔지니어 상세 튜닝 (Manual) ➔", on_click=go_to_step, args=(3,), type="primary", use_container_width=True)
