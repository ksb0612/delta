import streamlit as st
import pandas as pd
import numpy as np
import copy
import yaml
import streamlit_authenticator as stauth
import os
from scipy.stats import weibull_min

# [FIX] Import helper functions from the new utils.py file
from utils import (
    load_config, format_number, calculate_errors,
    flatten_scenario_df, reconstruct_scenario_from_df
)
from advanced_simulator import AdvancedLaunchSimulator
from analysis import StrategicAnalyzer
from visualization import (
    create_roas_distribution_figure, create_timeseries_figure, create_retention_curve_figure,
    create_sensitivity_figure, create_profit_cdf_figure, create_convergence_figure,
    create_backtesting_figure, create_profit_histogram, create_profit_kde_plot
)

# --- Constants ---
# [UPDATE] Changed to a directory for individual scenario files
SCENARIOS_DIR = "scenarios"

# --- Caching Functions ---
@st.cache_data
def run_cached_simulation(_project_info, _assumptions, _num_simulations, _scenario_template, _arppu_params):
    """Runs a cached Monte Carlo simulation."""
    progress_bar = st.progress(0, text="시뮬레이션을 준비 중입니다...")
    def update_progress(progress, text):
        progress_bar.progress(progress, text=text)

    simulator = AdvancedLaunchSimulator(_project_info, _assumptions, _num_simulations, _scenario_template, _arppu_params)
    results = simulator.run_monte_carlo(progress_callback=update_progress)
    progress_bar.empty()
    return results

@st.cache_data
def run_cached_sensitivity_analysis(_config, _scenario_data):
    """Runs a cached sensitivity analysis."""
    analyzer = StrategicAnalyzer(_config)
    return analyzer.run_sensitivity_analysis(_scenario_data)

# --- Scenario Management Functions ---
def load_scenario_names():
    """Loads scenario names by scanning the scenarios directory."""
    if not os.path.exists(SCENARIOS_DIR):
        return []
    try:
        files = [f for f in os.listdir(SCENARIOS_DIR) if f.endswith((".yaml", ".yml"))]
        # Return names without extension, sorted alphabetically
        return sorted([os.path.splitext(f)[0] for f in files])
    except IOError as e:
        st.error(f"시나리오 폴더를 읽는 중 오류 발생: {e}")
        return []

def convert_numpy_to_native(data):
    """Recursively converts numpy types in a dict/list to native Python types for YAML serialization."""
    if isinstance(data, dict):
        return {key: convert_numpy_to_native(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_native(element) for element in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

def save_scenario(name, media_df, organic_df):
    """Saves the current scenario configuration as a separate .yaml file."""
    if not name:
        st.warning("시나리오 이름을 입력해주세요.")
        return

    # Basic sanitization for filename
    sanitized_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    if not sanitized_name:
        st.warning("유효한 시나리오 이름을 입력해주세요. (알파벳, 숫자, 공백, _, - 만 허용)")
        return
        
    filename = f"{sanitized_name}.yaml"
    filepath = os.path.join(SCENARIOS_DIR, filename)

    # Create the directory if it doesn't exist
    os.makedirs(SCENARIOS_DIR, exist_ok=True)

    scenario_data = {
        'media_mix': reconstruct_scenario_from_df(media_df.copy(), 'media_mix'),
        'organic_assumptions': reconstruct_scenario_from_df(organic_df.copy(), 'organic')
    }

    native_data = convert_numpy_to_native(scenario_data)

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(native_data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        st.success(f"'{name}' 시나리오가 '{filename}' 파일로 저장되었습니다!")
        # Refresh the list of scenarios in the session state
        st.session_state.scenario_files = load_scenario_names()
    except IOError as e:
        st.error(f"시나리오 저장 중 오류 발생: {e}")


# --- Page Config ---
st.set_page_config(layout="wide", page_title="미디어믹스 시뮬레이터")

# --- Session State Initialization ---
if 'config' not in st.session_state:
    st.session_state.config = load_config()
if 'project_info' not in st.session_state:
    st.session_state.project_info = st.session_state.config['project_info']
if 'assumptions' not in st.session_state:
    st.session_state.assumptions = st.session_state.config['assumptions']
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = {}
if 'media_mix_df' not in st.session_state:
    st.session_state.media_mix_df = flatten_scenario_df(st.session_state.config['scenario_template'], 'media_mix')
if 'organic_df' not in st.session_state:
    st.session_state.organic_df = flatten_scenario_df(st.session_state.config['scenario_template'], 'organic')
if 'scenario_files' not in st.session_state:
    st.session_state.scenario_files = load_scenario_names()
if 'show_add_organic_form' not in st.session_state:
    st.session_state.show_add_organic_form = False
if 'show_add_channel_form' not in st.session_state:
    st.session_state.show_add_channel_form = False


# --- UI Rendering Functions ---
def render_dashboard_summary(final_df, project_info, assumptions):
    st.subheader("🎯 최종 성과 요약")
    col1, col2, col3 = st.columns(3)
    median_roas = final_df['paid_roas'].median()
    median_revenue = final_df['paid_revenue'].median()
    total_budget = project_info['target_budget']
    median_profit = final_df['total_profit'].median()
    median_roi = median_profit / total_budget if total_budget > 0 else 0

    with col1:
        st.metric("예측 ROAS (중앙값)", f"{median_roas:.2%}", help="광고비 대비 유료 채널 매출")
        st.markdown(f"**예상 유료 매출:** {format_number(median_revenue, True)}")
    with col2:
        st.metric("예측 ROI (중앙값)", f"{median_roi:.2%}", help="광고비 대비 전체 순수익")
        st.markdown(f"**예상 전체 순수익:** {format_number(median_profit, True)}")
    with col3:
        render_target_guide(final_df, project_info, assumptions)

def render_target_guide(final_df, project_info, assumptions):
    with st.container(border=True):
        st.markdown("##### 🎯 목표 달성 가이드")
        target_roas = project_info['target_roas']
        st.markdown(f"**목표 ROAS:** `{target_roas:.1%}`")

        # [UPDATE] Calculate all required metrics for the guide
        paid_installs_median = final_df['paid_installs'].median()
        total_installs_median = final_df['total_installs'].median()
        paid_revenue_median = final_df['paid_revenue'].median()
        total_revenue_median = final_df['total_revenue'].median()
        blended_pcr_median = final_df['blended_pcr'].median()
        
        total_budget = project_info['target_budget']
        required_paid_revenue = target_roas * total_budget
        
        # Required Total/Blended CPI
        organic_revenue_median = total_revenue_median - paid_revenue_median
        required_total_revenue = required_paid_revenue + organic_revenue_median
        blended_cpi_median = total_budget / total_installs_median if total_installs_median > 0 else 0
        required_total_cpi = blended_cpi_median * (total_revenue_median / required_total_revenue) if required_total_revenue > 0 and total_revenue_median > 0 else np.nan

        # Required Paid CPI
        paid_cpi_median = total_budget / paid_installs_median if paid_installs_median > 0 else 0
        improvement_factor = required_paid_revenue / paid_revenue_median if paid_revenue_median > 0 else np.nan
        required_paid_cpi = paid_cpi_median / improvement_factor if not np.isnan(improvement_factor) else np.nan
        
        # Required ARPU & ARPPU
        required_total_arpu = required_total_revenue / total_installs_median if total_installs_median > 0 else 0
        total_paying_users_median = total_installs_median * blended_pcr_median
        required_total_arppu = required_total_revenue / total_paying_users_median if total_paying_users_median > 0 else 0

        st.markdown(f"**필요 Total CPI (상한):** `{format_number(required_total_cpi, True)}`")
        st.markdown(f"**필요 Paid CPI (상한):** `{format_number(required_paid_cpi, True)}`")
        st.markdown(f"**필요 전체 ARPU:** `{format_number(required_total_arpu, True)}`")
        st.markdown(f"**필요 전체 ARPPU:** `{format_number(required_total_arppu, True)}`")


def render_detailed_metrics(final_df):
    st.subheader("📈 상세 지표 분석 (중앙값 기준)")
    
    kpi_metrics = {
        "총수익": ("total_revenue", True, 0), 
        "총 설치 수": ("total_installs", False, 0),
        "유료 설치 수": ("paid_installs", False, 0),
        "전체 CPI": ("blended_cpi", True, 0),
        "유료 CPI": ("paid_cpi", True, 0),
        "전체 ARPU": ("blended_arpu", True, 0),
        "전체 ARPPU": ("blended_arppu", True, 0)
    }
    
    # [UPDATE] More dynamic layout generation for metrics
    keys = list(kpi_metrics.keys())
    num_metrics = len(keys)
    num_cols = 4  # Display 4 metrics per row

    for i in range(0, num_metrics, num_cols):
        row_keys = keys[i:i + num_cols]
        # [FIX] Create columns based on the number of items in the current row
        cols = st.columns(len(row_keys))
        for j, key in enumerate(row_keys):
            metric_key, is_currency, dec = kpi_metrics[key]
            with cols[j]:
                if metric_key in final_df.columns:
                    median_val = final_df[metric_key].median()
                    p10, p90 = final_df[metric_key].quantile(0.1), final_df[metric_key].quantile(0.9)
                    st.metric(key, format_number(median_val, is_currency, dec),
                              help=f"P10: {format_number(p10, is_currency, dec)}\nP90: {format_number(p90, is_currency, dec)}")
        # Add vertical space only if it's not the last row
        if i + num_cols < num_metrics:
            st.write("")


def render_main_charts(all_df, final_df, _config, _scenario_template):
    st.subheader("🔍 심층 분석 차트")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(create_roas_distribution_figure(final_df), use_container_width=True)
        analyzer_config = {'project_info': _config['project_info'], 'assumptions': _config['assumptions']}
        sensitivity_df = run_cached_sensitivity_analysis(analyzer_config, _scenario_template)
        st.plotly_chart(create_sensitivity_figure(sensitivity_df), use_container_width=True)
    with chart_col2:
        st.plotly_chart(create_timeseries_figure(all_df), use_container_width=True)
        st.plotly_chart(create_profit_cdf_figure(final_df), use_container_width=True)

    sim_viz = AdvancedLaunchSimulator(_config['project_info'], _config['assumptions'], 1, _scenario_template, {})
    retention_details = sim_viz.get_retention_model_details()
    st.plotly_chart(create_retention_curve_figure(retention_details), use_container_width=True)
    st.divider()
    st.subheader("📊 분포 분석")
    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        st.plotly_chart(create_profit_histogram(final_df), use_container_width=True)
    with dist_col2:
        st.plotly_chart(create_profit_kde_plot(final_df), use_container_width=True)

# ===================================================================
# 1. User Authentication
# ===================================================================
try:
    with open('auth.yaml') as file:
        auth_config = yaml.safe_load(file)
    authenticator = stauth.Authenticate(
        auth_config['credentials'], auth_config['cookie']['name'],
        auth_config['cookie']['key'], auth_config['cookie']['expiry_days']
    )
    authenticator.login()
except (FileNotFoundError, yaml.YAMLError) as e:
    st.error(f"인증 파일('auth.yaml') 로딩 중 오류 발생: {e}")
    st.stop()

if not st.session_state.get("authentication_status"):
    if st.session_state.get("authentication_status") is False: st.error('아이디 또는 비밀번호가 올바르지 않습니다.')
    elif st.session_state.get("authentication_status") is None: st.warning('아이디와 비밀번호를 입력해주세요.')
    st.stop()

# --- Main App Logic ---
st.sidebar.title(f"환영합니다, *{st.session_state['name']}* 님")
authenticator.logout('로그아웃', 'sidebar')
st.title("📈 동적 미디어믹스 시뮬레이터 (v6.6 - 지표 확장)")
tab1, tab2, tab3 = st.tabs(["📊 대시보드 & 결과", "⚙️ 시나리오 에디터", "🔍 모델 신뢰도 평가"])

# =====================================================================================
# Tab 2: Scenario Editor
# =====================================================================================
with tab2:
    st.header("⚙️ 시나리오 에디터")

    with st.expander("📁 시나리오 관리", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("현재 시나리오 저장")
            new_scenario_name = st.text_input("시나리오 이름", placeholder="예: 4분기 공격적 마케팅안")
            if st.button("💾 시나리오 저장"):
                save_scenario(new_scenario_name, st.session_state.media_mix_df, st.session_state.organic_df)
        
        with col2:
            st.subheader("저장된 시나리오 불러오기")
            if not st.session_state.scenario_files:
                st.caption("저장된 시나리오가 없습니다.")
            else:
                scenario_options = [""] + st.session_state.scenario_files
                selected_scenario = st.selectbox("불러올 시나리오를 선택하세요", options=scenario_options)
                if st.button("📂 시나리오 불러오기"):
                    if selected_scenario:
                        filepath = os.path.join(SCENARIOS_DIR, f"{selected_scenario}.yaml")
                        try:
                            with open(filepath, "r", encoding="utf-8") as f:
                                scenario_data = yaml.safe_load(f)
                            st.session_state.media_mix_df = flatten_scenario_df(scenario_data, 'media_mix')
                            st.session_state.organic_df = flatten_scenario_df(scenario_data, 'organic')
                            st.success(f"'{selected_scenario}' 시나리오를 불러왔습니다.")
                            st.rerun()
                        except (IOError, yaml.YAMLError) as e:
                            st.error(f"'{selected_scenario}' 시나리오를 불러오는 중 오류 발생: {e}")
                    else:
                        st.warning("불러올 시나리오를 선택해주세요.")
    
    st.subheader("1. ARPPU 시나리오")
    st.session_state.arppu_choice = st.radio(
        "분석할 ARPPU 시나리오를 선택하세요:",
        ('ARPPU D30 (기준)', 'ARPPU D60', 'ARPPU D90'),
        horizontal=True, key='arppu_radio'
    )
    st.divider()

    with st.expander("2. 프로젝트 정보"):
        p_info = st.session_state.project_info
        p_info['name'] = st.text_input("프로젝트 이름", value=p_info.get('name', ''))
        pc1, pc2, pc3 = st.columns(3)
        p_info['marketing_duration_days'] = pc1.number_input("마케팅 기간 (일)", 30, 365, p_info.get('marketing_duration_days', 180))
        p_info['target_budget'] = pc2.number_input("총 광고 예산 (원)", 1000000, None, p_info.get('target_budget', 1000000000), 1000000)
        p_info['target_roas'] = pc3.number_input("목표 ROAS (%)", 1.0, None, p_info.get('target_roas', 1.2) * 100, 1.0) / 100.0

    with st.expander("3. 고급 시뮬레이션 설정"):
        assump = st.session_state.assumptions
        st.markdown("##### 예산 집행 패턴")
        ac1, ac2 = st.columns(2)
        assump['budget_pacing']['burst_days'] = ac1.slider("초반 집중 기간 (일)", 1, 90, assump['budget_pacing'].get('burst_days', 30))
        assump['budget_pacing']['burst_intensity'] = ac2.slider("집중 기간 강도 (배수)", 1.0, 5.0, assump['budget_pacing'].get('burst_intensity', 2.5), 0.1)
        st.markdown("##### 광고 효과 모델링")
        ac3, ac4 = st.columns(2)
        assump['adstock']['decay_rate'] = ac3.slider("효과 이월 계수", 0.0, 1.0, assump['adstock'].get('decay_rate', 0.5), 0.05)
        assump['adstock']['saturation_alpha'] = ac4.slider("효과 포화 계수", 0.1, 1.0, assump['adstock'].get('saturation_alpha', 0.8), 0.05)
        st.markdown("##### 시뮬레이션 정밀도")
        assump['monte_carlo']['num_simulations'] = st.selectbox("샘플링 횟수", [100, 300, 500, 1000, 2000], index=3)

    def render_add_channel_form():
        with st.form(key="add_channel_form"):
            st.subheader("📢 새로운 유료 채널 추가")
            default_paid = st.session_state.config['default_rows']['paid']
            st.markdown("<h6>채널 정보</h6>", unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            os_val = c1.selectbox("OS", ["iOS", "Android"], index=0)
            country_val = c2.text_input("국가", default_paid['country'])
            name_val = c3.text_input("채널명", default_paid['name'])
            c4, c5, c6 = st.columns(3)
            product_val = c4.text_input("상품", default_paid['product'])
            type_val = c5.text_input("타입", default_paid['type'])
            budget_val = c6.number_input("예산 (절대값)", value=100000)
            st.markdown("<h6>성과 지표</h6>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown("**CPI**")
                cpi_loc_val = st.number_input("평균 (loc)", value=default_paid['cpi']['loc'], key="cpi_loc")
                cpi_scale_val = st.number_input("표준편차 (scale)", value=default_paid['cpi']['scale'], key="cpi_scale")
            with m2:
                st.markdown("**결제 전환율**")
                pcr_a_val = st.number_input("성공 (a)", value=default_paid['payer_conversion_rate']['a'], key="pcr_a")
                pcr_b_val = st.number_input("실패 (b)", value=default_paid['payer_conversion_rate']['b'], key="pcr_b")
            with m3:
                st.markdown("**ARPPU D30**")
                arppu_loc_val = st.number_input("평균 (loc)", value=default_paid['arppu_d30']['loc'], key="arppu_loc")
                arppu_scale_val = st.number_input("표준편차 (scale)", value=default_paid['arppu_d30']['scale'], key="arppu_scale")
            st.markdown("<h6>리텐션 지표</h6>", unsafe_allow_html=True)
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown("**D1**")
                ret_d1_loc = st.number_input("평균", value=default_paid['retention_d1']['loc'], key="d1_loc", format="%.4f")
                ret_d1_scale = st.number_input("표준편차", value=default_paid['retention_d1']['scale'], key="d1_scale", format="%.4f")
            with r2:
                st.markdown("**D7**")
                ret_d7_loc = st.number_input("평균", value=default_paid['retention_d7']['loc'], key="d7_loc", format="%.4f")
                ret_d7_scale = st.number_input("표준편차", value=default_paid['retention_d7']['scale'], key="d7_scale", format="%.4f")
            with r3:
                st.markdown("**D14**")
                ret_d14_loc = st.number_input("평균", value=default_paid['retention_d14']['loc'], key="d14_loc", format="%.4f")
                ret_d14_scale = st.number_input("표준편차", value=default_paid['retention_d14']['scale'], key="d14_scale", format="%.4f")
            with r4:
                st.markdown("**D30**")
                ret_d30_loc = st.number_input("평균", value=default_paid['retention_d30']['loc'], key="d30_loc", format="%.4f")
                ret_d30_scale = st.number_input("표준편차", value=default_paid['retention_d30']['scale'], key="d30_scale", format="%.4f")
            submitted = st.form_submit_button("채널 추가")
            if submitted:
                new_row_data = copy.deepcopy(default_paid)
                new_row_data.update({'os': os_val, 'country': country_val, 'name': name_val, 'product': product_val, 'type': type_val, 'budget_ratio': budget_val})
                new_row_data['cpi'] = {'loc': cpi_loc_val, 'scale': cpi_scale_val}
                new_row_data['payer_conversion_rate'] = {'a': pcr_a_val, 'b': pcr_b_val}
                new_row_data['arppu_d30'] = {'loc': arppu_loc_val, 'scale': arppu_scale_val}
                new_row_data['retention_d1'] = {'loc': ret_d1_loc, 'scale': ret_d1_scale}
                new_row_data['retention_d7'] = {'loc': ret_d7_loc, 'scale': ret_d7_scale}
                new_row_data['retention_d14'] = {'loc': ret_d14_loc, 'scale': ret_d14_scale}
                new_row_data['retention_d30'] = {'loc': ret_d30_loc, 'scale': ret_d30_scale}
                temp_scenario = {'media_mix': [{'os': os_val, 'channels': [{'country': country_val, 'media': [new_row_data]}]}]}
                flat_row = flatten_scenario_df(temp_scenario, 'media_mix')
                st.session_state.media_mix_df = pd.concat([st.session_state.media_mix_df, flat_row], ignore_index=True)
                st.session_state.show_add_channel_form = False
                st.rerun()

    if st.session_state.show_add_channel_form:
        render_add_channel_form()

    with st.expander("5. 유료 채널 믹스", expanded=not st.session_state.show_add_channel_form):
        if st.button("📢 새로운 채널 추가", key="add_paid_channel_btn"):
            st.session_state.show_add_channel_form = True
            st.rerun()
        st.session_state.media_mix_df = st.data_editor(st.session_state.media_mix_df, num_rows="dynamic", use_container_width=True, key="media_mix_editor", column_config={"budget_ratio": st.column_config.NumberColumn("예산 (절대값)", format="%d")})
        st.download_button(label="CSV로 내보내기", data=st.session_state.media_mix_df.to_csv(index=False).encode('utf-8-sig'), file_name='media_mix.csv', mime='text/csv')
    
    st.divider()

    def render_add_organic_form():
        with st.form(key="add_organic_form"):
            st.subheader("🌱 새로운 자연 유입 국가 추가")
            default_org = st.session_state.config['default_rows']['organic']
            country_val = st.text_input("국가", default_org['country'])
            st.markdown("<h6>성과 지표</h6>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown("**일별 설치 수**")
                installs_loc_val = st.number_input("평균 (loc)", value=default_org['daily_installs']['loc'], key="org_inst_loc")
                installs_scale_val = st.number_input("표준편차 (scale)", value=default_org['daily_installs']['scale'], key="org_inst_scale")
            with m2:
                st.markdown("**결제 전환율**")
                pcr_a_val = st.number_input("성공 (a)", value=default_org['payer_conversion_rate']['a'], key="org_pcr_a")
                pcr_b_val = st.number_input("실패 (b)", value=default_org['payer_conversion_rate']['b'], key="org_pcr_b")
            with m3:
                st.markdown("**ARPPU D30**")
                arppu_loc_val = st.number_input("평균 (loc)", value=default_org['arppu_d30']['loc'], key="org_arppu_loc")
                arppu_scale_val = st.number_input("표준편차 (scale)", value=default_org['arppu_d30']['scale'], key="org_arppu_scale")
            st.markdown("<h6>리텐션 지표</h6>", unsafe_allow_html=True)
            r1, r2, r3, r4 = st.columns(4)
            with r1:
                st.markdown("**D1**")
                ret_d1_loc = st.number_input("평균", value=default_org['retention_d1']['loc'], key="org_d1_loc", format="%.4f")
                ret_d1_scale = st.number_input("표준편차", value=default_org['retention_d1']['scale'], key="org_d1_scale", format="%.4f")
            with r2:
                st.markdown("**D7**")
                ret_d7_loc = st.number_input("평균", value=default_org['retention_d7']['loc'], key="org_d7_loc", format="%.4f")
                ret_d7_scale = st.number_input("표준편차", value=default_org['retention_d7']['scale'], key="org_d7_scale", format="%.4f")
            with r3:
                st.markdown("**D14**")
                ret_d14_loc = st.number_input("평균", value=default_org['retention_d14']['loc'], key="org_d14_loc", format="%.4f")
                ret_d14_scale = st.number_input("표준편차", value=default_org['retention_d14']['scale'], key="org_d14_scale", format="%.4f")
            with r4:
                st.markdown("**D30**")
                ret_d30_loc = st.number_input("평균", value=default_org['retention_d30']['loc'], key="org_d30_loc", format="%.4f")
                ret_d30_scale = st.number_input("표준편차", value=default_org['retention_d30']['scale'], key="org_d30_scale", format="%.4f")
            submitted = st.form_submit_button("국가 추가")
            if submitted:
                new_row_data = copy.deepcopy(default_org)
                new_row_data['country'] = country_val
                new_row_data['daily_installs'] = {'loc': installs_loc_val, 'scale': installs_scale_val}
                new_row_data['payer_conversion_rate'] = {'a': pcr_a_val, 'b': pcr_b_val}
                new_row_data['arppu_d30'] = {'loc': arppu_loc_val, 'scale': arppu_scale_val}
                new_row_data['retention_d1'] = {'loc': ret_d1_loc, 'scale': ret_d1_scale}
                new_row_data['retention_d7'] = {'loc': ret_d7_loc, 'scale': ret_d7_scale}
                new_row_data['retention_d14'] = {'loc': ret_d14_loc, 'scale': ret_d14_scale}
                new_row_data['retention_d30'] = {'loc': ret_d30_loc, 'scale': ret_d30_scale}
                flat_row = flatten_scenario_df({'organic_assumptions': [new_row_data]}, 'organic')
                st.session_state.organic_df = pd.concat([st.session_state.organic_df, flat_row], ignore_index=True)
                st.session_state.show_add_organic_form = False
                st.rerun()
    
    if st.session_state.show_add_organic_form:
        render_add_organic_form()

    with st.expander("4. 자연 유입 설정", expanded=not st.session_state.show_add_organic_form):
        if st.button("🌱 새로운 국가 추가", key="add_organic_country_btn"):
            st.session_state.show_add_organic_form = True
            st.rerun()
        st.session_state.organic_df = st.data_editor(st.session_state.organic_df, num_rows="dynamic", use_container_width=True, key="organic_editor")

# =====================================================================================
# Tab 1: Dashboard
# =====================================================================================
with tab1:
    selected_arppu_scenario = st.session_state.get('arppu_choice', 'ARPPU D30 (기준)')
    st.header(f"📊 대시보드 & 결과: {selected_arppu_scenario}")
    
    if st.button("🚀 시뮬레이션 실행", type="primary", key="run_sim_main"):
        with st.spinner(f"{selected_arppu_scenario} 시나리오로 시뮬레이션을 실행합니다..."):
            current_scenario_template = copy.deepcopy(st.session_state.config['scenario_template'])
            current_scenario_template['media_mix'] = reconstruct_scenario_from_df(st.session_state.media_mix_df.copy(), 'media_mix')
            current_scenario_template['organic_assumptions'] = reconstruct_scenario_from_df(st.session_state.organic_df.copy(), 'organic')
            arppu_key = selected_arppu_scenario.split(" ")[0].lower()
            arppu_params = {'scenario': selected_arppu_scenario, 'uplift_rate': st.session_state.assumptions['arppu_scenario_weights'].get(f"{arppu_key}_uplift", 1.0)}
            all_df, final_df = run_cached_simulation(st.session_state.project_info, st.session_state.assumptions, st.session_state.assumptions['monte_carlo']['num_simulations'], current_scenario_template, arppu_params)
            st.session_state.simulation_results[selected_arppu_scenario] = {"all_df": all_df, "final_df": final_df, "scenario_used": current_scenario_template}
        st.success(f"{selected_arppu_scenario} 시뮬레이션이 완료되었습니다!")

    if selected_arppu_scenario in st.session_state.simulation_results:
        results = st.session_state.simulation_results[selected_arppu_scenario]
        render_dashboard_summary(results['final_df'], st.session_state.project_info, st.session_state.assumptions)
        st.divider()
        render_detailed_metrics(results['final_df'])
        st.divider()
        render_main_charts(results['all_df'], results['final_df'], st.session_state.config, results['scenario_used'])
        st.divider()
        st.subheader("⬇️ 일별 예측 데이터 내보내기")
        export_df = results['all_df'].groupby(['day', 'os', 'country', 'name', 'product', 'type']).median().reset_index()
        st.download_button(label="일별 상세 결과 CSV 다운로드", data=export_df.to_csv(index=False).encode('utf-8-sig'), file_name=f"daily_prediction_{selected_arppu_scenario}.csv", mime="text/csv")
    else:
        st.info("'시뮬레이션 실행' 버튼을 눌러 결과를 확인하세요.")

# =====================================================================================
# Tab 3: Model Validation
# =====================================================================================
with tab3:
    st.header("🔍 모델 신뢰도 평가")
    st.subheader("1. 수렴 테스트")
    st.info("시뮬레이션 횟수를 늘려감에 따라 결과값이 안정적으로 수렴하는지 확인합니다.")
    if st.button("수렴 테스트 실행"):
        with st.spinner("수렴 테스트를 위한 시뮬레이션을 실행 중입니다..."):
            sim = AdvancedLaunchSimulator(st.session_state.project_info, st.session_state.assumptions, st.session_state.assumptions['monte_carlo']['num_simulations'], st.session_state.config['scenario_template'], {'scenario': 'ARPPU D30 (기준)', 'uplift_rate': 1.0})
            convergence_data = sim.run_convergence_test()
            st.plotly_chart(create_convergence_figure(convergence_data), use_container_width=True)
    
    st.divider()
    st.subheader("2. 백테스팅")
    st.info("과거 실제 데이터를 업로드하여, 현재 시나리오 설정이 과거 성과를 얼마나 잘 설명하는지 검증합니다.")
    uploaded_file = st.file_uploader("과거 성과 데이터 업로드 (CSV)", type="csv")
    if uploaded_file:
        try:
            actual_data = pd.read_csv(uploaded_file)
            required_cols = ['day', 'actual_roas']
            if all(col in actual_data.columns for col in required_cols):
                actual_data = actual_data[required_cols].astype({'day': 'int', 'actual_roas': 'float'})
                st.dataframe(actual_data.head())
                if st.button("백테스팅 실행"):
                    with st.spinner("백테스팅 시뮬레이션을 실행 중입니다..."):
                        backtest_scenario = copy.deepcopy(st.session_state.config['scenario_template'])
                        backtest_scenario['media_mix'] = reconstruct_scenario_from_df(st.session_state.media_mix_df.copy(), 'media_mix')
                        backtest_scenario['organic_assumptions'] = reconstruct_scenario_from_df(st.session_state.organic_df.copy(), 'organic')
                        all_df_backtest, _ = run_cached_simulation(st.session_state.project_info, st.session_state.assumptions, 300, backtest_scenario, {'scenario': 'ARPPU D30 (기준)', 'uplift_rate': 1.0})
                        sim_median_roas = all_df_backtest.groupby('day')['paid_roas'].median().reset_index().rename(columns={'paid_roas': 'predicted_roas'})
                        comparison_df = pd.merge(actual_data, sim_median_roas, on='day', how='left').dropna()
                        rmse, mape = calculate_errors(comparison_df['actual_roas'], comparison_df['predicted_roas'])
                        st.plotly_chart(create_backtesting_figure(all_df_backtest, actual_data), use_container_width=True)
                        c1, c2 = st.columns(2)
                        c1.metric("RMSE", f"{rmse:.4f}")
                        c2.metric("MAPE", f"{mape:.2f}%")
            else:
                st.error(f"CSV 파일에 다음 필수 컬럼이 포함되어야 합니다: {', '.join(required_cols)}")
        except Exception as e:
            st.error(f"파일 처리 중 오류가 발생했습니다: {e}")

