import streamlit as st
import pandas as pd
import numpy as np
import copy
import yaml
import streamlit_authenticator as stauth
import os
import json
from scipy.stats import weibull_min
import hashlib
from plotly.subplots import make_subplots
from visualization import (
    create_roas_distribution_figure, create_timeseries_figure, create_retention_curve_figure,
    create_sensitivity_figure, create_profit_cdf_figure, create_convergence_figure,
    create_backtesting_figure, create_profit_histogram, create_profit_kde_plot,
    create_cost_efficiency_analysis, create_performance_contribution_analysis
)



from utils import (
    load_config, format_number, calculate_errors,
    flatten_scenario_df, reconstruct_scenario_from_df
)
from advanced_simulator import AdvancedLaunchSimulator
from analysis import StrategicAnalyzer


SCENARIOS_DIR = "scenarios"

# --- Caching Functions ---
@st.cache_data
def create_settings_hash(project_info, assumptions, scenario_template, arppu_params):
    """모든 설정의 해시값을 생성하여 변경사항을 정확히 감지"""
    settings = {
        "project_info": project_info,
        "assumptions": assumptions,
        "scenario_template": scenario_template,
        "arppu_params": arppu_params
    }
    
    # JSON으로 직렬화 후 해시 생성
    settings_str = json.dumps(settings, default=convert_numpy_to_native, sort_keys=True)
    return hashlib.md5(settings_str.encode()).hexdigest()

@st.cache_data
def run_cached_simulation(settings_hash: str, project_info, assumptions, num_simulations, scenario_template, arppu_params):
    """설정 해시값을 기반으로 캐싱하는 시뮬레이션 함수"""
    simulator = AdvancedLaunchSimulator(project_info, assumptions, num_simulations, scenario_template, arppu_params)
    results = simulator.run_monte_carlo()
    return results


@st.cache_data
def run_cached_sensitivity_analysis(_config_str, _scenario_data_str):
    _config = json.loads(_config_str)
    _scenario_data = json.loads(_scenario_data_str)
    analyzer = StrategicAnalyzer(_config)
    return analyzer.run_sensitivity_analysis(_scenario_data)

# --- Scenario Management Functions ---
def load_scenario_names():
    if not os.path.exists(SCENARIOS_DIR):
        return []
    try:
        files = [f for f in os.listdir(SCENARIOS_DIR) if f.endswith((".yaml", ".yml"))]
        return sorted([os.path.splitext(f)[0] for f in files])
    except IOError as e:
        st.error(f"시나리오 폴더를 읽는 중 오류 발생: {e}")
        return []

def convert_numpy_to_native(data):
    if isinstance(data, dict):
        return {key: convert_numpy_to_native(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_native(element) for element in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, pd.DataFrame):
        return data.to_dict(orient='split')
    elif isinstance(data, np.ndarray):
        return data.tolist()
    else:
        return data

def save_scenario(name, media_df, organic_df):
    if not name:
        st.warning("시나리오 이름을 입력해주세요.")
        return
    sanitized_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    if not sanitized_name:
        st.warning("유효한 시나리오 이름을 입력해주세요. (알파벳, 숫자, 공백, _, - 만 허용)")
        return
    filename = f"{sanitized_name}.yaml"
    filepath = os.path.join(SCENARIOS_DIR, filename)
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
        st.session_state.scenario_files = load_scenario_names()
    except IOError as e:
        st.error(f"시나리오 저장 중 오류 발생: {e}")

# --- Page Config & Initialization ---
st.set_page_config(layout="wide", page_title="미디어믹스 시뮬레이터")

if 'config' not in st.session_state:
    st.session_state.config = load_config()
if 'project_info' not in st.session_state:
    st.session_state.project_info = st.session_state.config['project_info']
if 'assumptions' not in st.session_state:
    st.session_state.assumptions = st.session_state.config['assumptions']
if 'simulation_results' not in st.session_state:
    st.session_state.simulation_results = {}
if 'media_mix_df' not in st.session_state:
    df = flatten_scenario_df(st.session_state.config['scenario_template'], 'media_mix')
    df.drop(columns=[col for col in df.columns if 'retention' in col and 'scale' in col], inplace=True, errors='ignore')
    st.session_state.media_mix_df = df
if 'organic_df' not in st.session_state:
    df = flatten_scenario_df(st.session_state.config['scenario_template'], 'organic')
    df.drop(columns=[col for col in df.columns if 'retention' in col and 'scale' in col], inplace=True, errors='ignore')
    st.session_state.organic_df = df
if 'scenario_files' not in st.session_state:
    st.session_state.scenario_files = load_scenario_names()
if 'show_add_organic_form' not in st.session_state:
    st.session_state.show_add_organic_form = False
if 'show_add_channel_form' not in st.session_state:
    st.session_state.show_add_channel_form = False
if 'page' not in st.session_state:
    st.session_state.page = "대시보드 & 결과"
if 'last_run_settings_signature' not in st.session_state:
    st.session_state.last_run_settings_signature = None


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
        st.caption("`계산식: 유료 채널 매출 / 총 광고비`")
        st.markdown(f"**예상 유료 매출:** {format_number(median_revenue, True)}")
    with col2:
        st.metric("예측 ROI (중앙값)", f"{median_roi:.2%}", help="광고비 대비 전체 순수익")
        st.caption("`계산식: 총 순수익 / 총 광고비`")
        st.markdown(f"**예상 전체 순수익:** {format_number(median_profit, True)}")
    with col3:
        render_target_guide(final_df, project_info, assumptions)

def render_target_guide(final_df, project_info, assumptions):
    with st.container(border=True):
        st.markdown("##### 🎯 목표 달성 가이드")
        target_roas = project_info['target_roas']
        st.markdown(f"**목표 ROAS:** `{target_roas:.1%}`")

        # 기본 계산값들 (유료 채널 기준)
        paid_installs_median = final_df['paid_installs'].median()
        paid_revenue_median = final_df['paid_revenue'].median()
        paid_pcr_median = final_df['paid_paying_users'].median() / paid_installs_median if paid_installs_median > 0 else 0
        total_budget = project_info['target_budget']

        # 목표 달성을 위한 필요 유료 수익
        required_paid_revenue = target_roas * total_budget
        
        # CPI (상한)
        paid_cpi_median = final_df['paid_cpi'].median()
        improvement_factor = required_paid_revenue / paid_revenue_median if paid_revenue_median > 0 else np.nan
        required_paid_cpi = paid_cpi_median / improvement_factor if not np.isnan(improvement_factor) else np.nan

        # LTV, ARPU, ARPPU를 30일 기준으로 재계산
        from scipy.stats import weibull_min
        ltv_cfg = assumptions['ltv_curve']
        shape, scale = ltv_cfg['shape'], ltv_cfg['scale']
        
        # 전체 LTV 기간 대비 30일차의 매출 비중
        ltv_duration = project_info.get('ltv_duration_days', 30)
        total_ltv_ratio = weibull_min.cdf(ltv_duration, c=shape, scale=scale)
        ltv_ratio_d30 = weibull_min.cdf(30, c=shape, scale=scale)
        
        # 전체 기간 동안의 필요 유료 수익을 30일 기준으로 환산
        required_paid_revenue_d30 = required_paid_revenue * (ltv_ratio_d30 / total_ltv_ratio) if total_ltv_ratio > 0 else required_paid_revenue
        
        # 30일 기준 목표 ARPU 및 ARPPU
        required_paid_arpu_d30 = required_paid_revenue_d30 / paid_installs_median if paid_installs_median > 0 else 0
        paid_paying_users_median = paid_installs_median * paid_pcr_median
        required_paid_arppu_d30 = required_paid_revenue_d30 / paid_paying_users_median if paid_paying_users_median > 0 else 0

        # 단기 LTV 목표 계산
        days = [3, 7, 14]
        cumulative_ltv_ratios = {day: weibull_min.cdf(day, c=shape, scale=scale) for day in days}
        
        # 30일 ARPU를 기준으로 각 시점별 LTV 계산
        required_ltv_d3 = required_paid_arpu_d30 * (cumulative_ltv_ratios[3] / ltv_ratio_d30) if ltv_ratio_d30 > 0 else 0
        required_ltv_d7 = required_paid_arpu_d30 * (cumulative_ltv_ratios[7] / ltv_ratio_d30) if ltv_ratio_d30 > 0 else 0
        required_ltv_d14 = required_paid_arpu_d30 * (cumulative_ltv_ratios[14] / ltv_ratio_d30) if ltv_ratio_d30 > 0 else 0
        
        # 결과 출력 (D+30일 LTV는 ARPU D30과 동일)
        st.markdown(f"**유료 CPI (상한):** `{format_number(required_paid_cpi, True)}`")
        st.markdown(f"**유료 ARPPU D30 (최소):** `{format_number(required_paid_arppu_d30, True)}`")
        st.markdown(f"**유료 ARPU D30 (최소):** `{format_number(required_paid_arpu_d30, True)}`")
        st.markdown(f"**유료 D+3일 LTV (최소):** `{format_number(required_ltv_d3, True)}`")
        st.markdown(f"**유료 D+7일 LTV (최소):** `{format_number(required_ltv_d7, True)}`")
        st.markdown(f"**유료 D+14일 LTV (최소):** `{format_number(required_ltv_d14, True)}`")
        st.markdown(f"**유료 D+30일 LTV (최소):** `{format_number(required_paid_arpu_d30, True)}`")


def render_detailed_metrics(final_df):
    st.subheader("📈 상세 지표 분석 (중앙값 기준)")
    
    kpi_metrics = {
        "총수익": ("total_revenue", True, 0, "유료 수익 + 오가닉 수익"),
        "유료 수익": ("paid_revenue", True, 0, "모든 유료 채널에서 발생한 누적 매출"),
        "총 순수익": ("total_profit", True, 0, "총수익 - 총 광고비"),
        "총 설치 수": ("total_installs", False, 0, "유료 설치 수 + 오가닉 설치 수"),
        "오가닉 설치 수": ("organic_installs", False, 0, "자연적으로 발생한 누적 설치 수"),
        "유료 설치 수": ("paid_installs", False, 0, "모든 유료 채널에서 발생한 누적 설치 수"),
        "전체 CPI": ("blended_cpi", True, 0, "총 광고비 / 총 설치 수"),
        "유료 CPI": ("paid_cpi", True, 0, "총 광고비 / 유료 설치 수"),
        "전체 ARPU": ("blended_arpu", True, 0, "총수익 / 총 설치 수"),
        "전체 ARPPU": ("blended_arppu", True, 0, "총수익 / 총 결제 유저 수"),
    }

    df_for_display = final_df.copy()
    if 'total_installs' in df_for_display.columns and 'paid_installs' in df_for_display.columns:
        df_for_display['organic_installs'] = df_for_display['total_installs'] - df_for_display['paid_installs']

    # 데이터를 담을 리스트 초기화
    metrics_data = []

    # 각 지표에 대해 계산 및 포맷팅
    for key, (metric_key, is_currency, dec, explanation) in kpi_metrics.items():
        if metric_key in df_for_display.columns:
            median_val = df_for_display[metric_key].median()
            p10 = df_for_display[metric_key].quantile(0.1)
            p90 = df_for_display[metric_key].quantile(0.9)
            
            metrics_data.append({
                "지표": key,
                "P10": format_number(p10, is_currency, dec),
                "중앙값 (P50)": format_number(median_val, is_currency, dec),
                "P90": format_number(p90, is_currency, dec),
                "계산식": explanation
            })

    # 데이터프레임 생성 및 출력
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)

def render_breakdown_tables(all_df):
    st.subheader("📊 채널별/국가별 상세 성과")
    st.caption("각 채널 및 국가별 성과의 중앙값(Median)입니다. 컬럼 헤더를 클릭하여 정렬할 수 있습니다.")

    paid_df = all_df[all_df['type'] != 'Organic'].copy()
    if paid_df.empty:
        st.info("성과를 분석할 유료 채널 데이터가 없습니다.")
        return

    # 각 시뮬레이션별, 채널별 최종 성과 집계
    channel_summary_per_sim = paid_df.groupby(['sim_id', 'os', 'country', 'name']).agg(
        total_spend=('spend', 'sum'),
        total_revenue=('revenue', 'sum'),
        total_installs=('installs', 'sum')
    ).reset_index()

    # 채널별 중앙값 성과 계산
    median_performance = channel_summary_per_sim.drop(columns='sim_id').groupby(['os', 'country', 'name']).median().reset_index()

    # 최종 지표 계산
    median_performance['ROAS'] = median_performance['total_revenue'] / median_performance['total_spend'].replace(0, np.nan)
    median_performance['CPI'] = median_performance['total_spend'] / median_performance['total_installs'].replace(0, np.nan)
    median_performance['Profit'] = median_performance['total_revenue'] - median_performance['total_spend']

    # 국가별 성과 집계
    country_summary = median_performance.groupby('country').agg(
        total_spend=('total_spend', 'sum'),
        total_revenue=('total_revenue', 'sum'),
        total_installs=('total_installs', 'sum')
    ).reset_index()
    country_summary['ROAS'] = country_summary['total_revenue'] / country_summary['total_spend'].replace(0, np.nan)
    country_summary['CPI'] = country_summary['total_spend'] / country_summary['total_installs'].replace(0, np.nan)
    country_summary['Profit'] = country_summary['total_revenue'] - country_summary['total_spend']


    # UI 탭 생성
    tab1, tab2 = st.tabs(["📈 국가별 요약", "📋 채널별 상세"])

    with tab1:
        st.dataframe(
            country_summary.style.format({
                "total_spend": "{:,.0f}원",
                "total_revenue": "{:,.0f}원",
                "total_installs": "{:,.0f}",
                "ROAS": "{:.2%}",
                "CPI": "{:,.0f}원",
                "Profit": "{:,.0f}원"
            }),
            use_container_width=True
        )
    
    with tab2:
        # 컬럼 이름 변경 및 순서 정리
        channel_detail_display = median_performance.rename(columns={
            'os': 'OS', 'country': '국가', 'name': '채널',
            'total_spend': '총 지출', 'total_revenue': '총 수익', 'total_installs': '총 설치 수'
        })
        
        st.dataframe(
            channel_detail_display[['OS', '국가', '채널', 'ROAS', 'CPI', 'Profit', '총 수익', '총 지출', '총 설치 수']]
            .style.format({
                "총 지출": "{:,.0f}원",
                "총 수익": "{:,.0f}원",
                "총 설치 수": "{:,.0f}",
                "ROAS": "{:.2%}",
                "CPI": "{:,.0f}원",
                "Profit": "{:,.0f}원"
            }),
            use_container_width=True
        )

def render_main_charts(all_df, final_df, _config, _scenario_template):
    st.subheader("📊 심층 분석 차트")
    
    # 기존 차트들 (첫 번째 행)
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(create_roas_distribution_figure(final_df), use_container_width=True)
        config_str = json.dumps(convert_numpy_to_native(_config), sort_keys=True)
        scenario_str = json.dumps(convert_numpy_to_native(_scenario_template), sort_keys=True)
        sensitivity_df = run_cached_sensitivity_analysis(config_str, scenario_str)
        st.plotly_chart(create_sensitivity_figure(sensitivity_df), use_container_width=True)
    with chart_col2:
        st.plotly_chart(create_timeseries_figure(all_df), use_container_width=True)
        st.plotly_chart(create_profit_cdf_figure(final_df), use_container_width=True)
    
    # 새로 추가되는 차트들 (두 번째 행)
    st.subheader("📈 채널 성과 분석")
    analysis_col1, analysis_col2 = st.columns(2)
    with analysis_col1:
        st.plotly_chart(create_cost_efficiency_analysis(all_df), use_container_width=True)
    with analysis_col2:
        st.plotly_chart(create_performance_contribution_analysis(all_df), use_container_width=True)
    
    # 기존 리텐션 차트
    sim_viz = AdvancedLaunchSimulator(_config['project_info'], _config['assumptions'], 1, _scenario_template, {})
    retention_details = sim_viz.get_retention_model_details()
    st.plotly_chart(create_retention_curve_figure(retention_details), use_container_width=True)
    
    # 기존 분포 분석
    st.divider()
    st.subheader("📊 분포 분석")
    dist_col1, dist_col2 = st.columns(2)
    with dist_col1:
        st.plotly_chart(create_profit_histogram(final_df), use_container_width=True)
    with dist_col2:
        st.plotly_chart(create_profit_kde_plot(final_df), use_container_width=True)


# --- Authentication ---
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

# --- Main App Layout ---
st.sidebar.title(f"환영합니다, *{st.session_state['name']}* 님")
authenticator.logout('로그아웃', 'sidebar')
st.sidebar.divider()

st.sidebar.title("메뉴")
# 추가: 캐시 강제 초기화 버튼 (디버깅용)
if st.sidebar.button("🔄 캐시 초기화"):
    st.cache_data.clear()
    st.session_state.simulation_results.clear()
    st.success("캐시가 초기화되었습니다!")
page_options = ["📊 대시보드 & 결과", "⚙️ 시나리오 에디터", "🔍 모델 신뢰도 평가"]
st.session_state.page = st.sidebar.radio("페이지를 선택하세요", page_options, label_visibility="collapsed")

st.title("📈 동적 미디어믹스 시뮬레이터 (v9.0 - Final)")

# --- Page Content ---
if st.session_state.page == "📊 대시보드 & 결과":
    selected_arppu_scenario = st.session_state.get('arppu_choice', 'ARPPU D30 (기준)')
    st.header(f"📊 대시보드 & 결과: {selected_arppu_scenario}")
    
    if st.button("🚀 시뮬레이션 실행", type="primary", key="run_sim_main"):
        with st.spinner(f"'{selected_arppu_scenario}' 시나리오로 시뮬레이션을 실행합니다..."):
            current_scenario_template = {
                'media_mix': reconstruct_scenario_from_df(st.session_state.media_mix_df.copy(), 'media_mix'),
                'organic_assumptions': reconstruct_scenario_from_df(st.session_state.organic_df.copy(), 'organic')
            }
            
            parts = selected_arppu_scenario.split(" ")
            day_part = parts[1].lower() if len(parts) > 1 else "d30"
            uplift_config_key = f"{day_part}_uplift"

            arppu_params = {
                'scenario': selected_arppu_scenario,
                'uplift_rate': st.session_state.assumptions['arppu_scenario_weights'].get(uplift_config_key, 1.0),
            }
            
            # 설정 해시 생성
            settings_hash = create_settings_hash(
                st.session_state.project_info,
                st.session_state.assumptions, 
                current_scenario_template,
                arppu_params
            )
            
            # 해시를 포함한 캐싱된 시뮬레이션 실행
            all_df, final_df = run_cached_simulation(
                settings_hash,
                st.session_state.project_info,
                st.session_state.assumptions,
                st.session_state.assumptions['monte_carlo']['num_simulations'],
                current_scenario_template,
                arppu_params
            )
            
            st.session_state.simulation_results[selected_arppu_scenario] = {
                "all_df": all_df, 
                "final_df": final_df, 
                "scenario_used": current_scenario_template,
                "settings_hash": settings_hash  # 해시값도 저장
            }
            
        st.success(f"'{selected_arppu_scenario}' 시뮬레이션이 완료되었습니다!")

    if selected_arppu_scenario in st.session_state.simulation_results:
        results = st.session_state.simulation_results[selected_arppu_scenario]
        
        with st.container(border=True):
            render_dashboard_summary(results['final_df'], st.session_state.project_info, st.session_state.assumptions)
        
        st.divider()
        
        with st.container(border=True):
            render_detailed_metrics(results['final_df'])
        
        st.divider()

        with st.container(border=True):
            render_breakdown_tables(results['all_df']) # NEW: Breakdown tables
        
        st.divider()

        with st.container(border=True):
            current_config_for_viz = {
                'project_info': st.session_state.project_info,
                'assumptions': st.session_state.assumptions
            }
            render_main_charts(results['all_df'], results['final_df'], current_config_for_viz, results['scenario_used'])
        
        st.divider()

        with st.container(border=True):
            st.subheader("⬇️ 예측 데이터 내보내기")
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                export_df = results['all_df'].groupby(['day', 'os', 'country', 'name', 'product', 'type']).median().reset_index()
                st.download_button(label="일별 요약 결과 CSV 다운로드", data=export_df.to_csv(index=False).encode('utf-8-sig'), file_name=f"daily_summary_{selected_arppu_scenario}.csv", mime="text/csv")
            with dl_col2:
                st.download_button(label="시뮬레이션 Raw 데이터 다운로드", data=results['final_df'].to_csv(index=False).encode('utf-8-sig'), file_name=f"final_results_{selected_arppu_scenario}.csv", mime="text/csv")

    else:
        st.info("'시뮬레이션 실행' 버튼을 눌러 결과를 확인하세요.")

elif st.session_state.page == "⚙️ 시나리오 에디터":
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
                            
                            media_df = flatten_scenario_df(scenario_data, 'media_mix')
                            media_df.drop(columns=[col for col in media_df.columns if 'retention' in col and 'scale' in col], inplace=True, errors='ignore')
                            st.session_state.media_mix_df = media_df

                            organic_df = flatten_scenario_from_df(scenario_data, 'organic')
                            organic_df.drop(columns=[col for col in organic_df.columns if 'retention' in col and 'scale' in col], inplace=True, errors='ignore')
                            st.session_state.organic_df = organic_df
                            
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
        
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            p_info['marketing_duration_days'] = st.number_input(
                "마케팅 기간 (일)", 
                min_value=1, 
                max_value=365, 
                value=p_info.get('marketing_duration_days', 90),
                help="광고비를 집행하여 신규 유저를 획득하는 기간입니다."
            )
        with row1_col2:
            p_info['ltv_duration_days'] = st.number_input(
                "누적 매출 확인 기간 (일)", 
                min_value=p_info['marketing_duration_days'], 
                max_value=730, 
                value=max(p_info.get('ltv_duration_days', 180), p_info['marketing_duration_days']),
                help="마케팅으로 유입된 유저들의 누적 매출(LTV)을 추적하는 전체 시뮬레이션 기간입니다."
            )

        row2_col1, row2_col2 = st.columns(2)
        with row2_col1:
            p_info['target_budget'] = st.number_input("총 광고 예산 (원)", 1000000, None, p_info.get('target_budget', 1000000000), 1000000)
        with row2_col2:
            p_info['target_roas'] = row2_col2.number_input("목표 ROAS (%)", 1.0, None, p_info.get('target_roas', 1.2) * 100, 1.0) / 100.0

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
    
    # --- Add Forms ---
    def render_add_channel_form():
        with st.form(key="add_channel_form"):
            # ... (form content remains the same)
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
                cpi_loc_val = st.number_input("평균", value=default_paid['cpi']['loc'], key="cpi_loc")
                cpi_scale_val = st.number_input("표준편차", value=default_paid['cpi']['scale'], key="cpi_scale")
            with m2:
                st.markdown("**결제 전환율**")
                pcr_a_val = st.number_input("성공 (a)", value=default_paid['payer_conversion_rate']['a'], key="pcr_a")
                pcr_b_val = st.number_input("실패 (b)", value=default_paid['payer_conversion_rate']['b'], key="pcr_b")
            with m3:
                st.markdown("**ARPPU D30**")
                arppu_loc_val = st.number_input("평균", value=default_paid['arppu_d30']['loc'], key="arppu_loc")
                arppu_scale_val = st.number_input("표준편차", value=default_paid['arppu_d30']['scale'], key="arppu_scale")
            st.markdown("<h6>리텐션 지표 (평균값만 입력)</h6>", unsafe_allow_html=True)
            r1, r2, r3, r4 = st.columns(4)
            ret_d1_loc = r1.number_input("D1", value=default_paid['retention_d1']['loc'], key="d1_loc", format="%.4f")
            ret_d7_loc = r2.number_input("D7", value=default_paid['retention_d7']['loc'], key="d7_loc", format="%.4f")
            ret_d14_loc = r3.number_input("D14", value=default_paid['retention_d14']['loc'], key="d14_loc", format="%.4f")
            ret_d30_loc = r4.number_input("D30", value=default_paid['retention_d30']['loc'], key="d30_loc", format="%.4f")
            
            submitted = st.form_submit_button("채널 추가")
            if submitted:
                new_row_data = copy.deepcopy(default_paid)
                new_row_data.update({'os': os_val, 'country': country_val, 'name': name_val, 'product': product_val, 'type': type_val, 'budget_ratio': budget_val})
                new_row_data['cpi'] = {'loc': cpi_loc_val, 'scale': cpi_scale_val}
                new_row_data['payer_conversion_rate'] = {'a': pcr_a_val, 'b': pcr_b_val}
                new_row_data['arppu_d30'] = {'loc': arppu_loc_val, 'scale': arppu_scale_val}
                new_row_data['retention_d1'] = {'loc': ret_d1_loc}
                new_row_data['retention_d7'] = {'loc': ret_d7_loc}
                new_row_data['retention_d14'] = {'loc': ret_d14_loc}
                new_row_data['retention_d30'] = {'loc': ret_d30_loc}
                temp_scenario = {'media_mix': [{'os': os_val, 'channels': [{'country': country_val, 'media': [new_row_data]}]}]}
                flat_row = flatten_scenario_df(temp_scenario, 'media_mix')
                flat_row.drop(columns=[col for col in flat_row.columns if 'retention' in col and 'scale' in col], inplace=True, errors='ignore')
                st.session_state.media_mix_df = pd.concat([st.session_state.media_mix_df, flat_row], ignore_index=True)
                st.session_state.show_add_channel_form = False
                st.rerun()

    if st.session_state.show_add_channel_form:
        render_add_channel_form()

    with st.expander("4. 유료 채널 믹스", expanded=not st.session_state.show_add_channel_form):
        st.subheader("채널 추가 방식 선택")
        if st.button("📢 개별 채널 추가", key="add_paid_channel_btn"):
            st.session_state.show_add_channel_form = True
            st.rerun()
        
        st.markdown("---")
        st.subheader("CSV 일괄 업로드")
        
        col1, col2 = st.columns(2)
        with col1:
            media_mix_template_df = pd.DataFrame(columns=st.session_state.media_mix_df.columns)
            st.download_button(
                label="📦 CSV 템플릿 다운로드",
                data=media_mix_template_df.to_csv(index=False).encode('utf-8-sig'),
                file_name='paid_channel_template.csv',
                mime='text/csv',
            )
        with col2:
            uploaded_media_mix_file = st.file_uploader("📂 CSV 파일 업로드", type="csv", key="media_mix_uploader")

        if uploaded_media_mix_file is not None:
            try:
                new_media_mix_df = pd.read_csv(uploaded_media_mix_file)
                if list(new_media_mix_df.columns) == list(st.session_state.media_mix_df.columns):
                    st.session_state.media_mix_df = pd.concat(
                        [st.session_state.media_mix_df, new_media_mix_df],
                        ignore_index=True
                    )
                    st.success("유료 채널 목록이 성공적으로 업데이트되었습니다!")
                else:
                    st.error("CSV 파일의 컬럼이 기존 템플릿과 일치하지 않습니다.")
            except Exception as e:
                st.error(f"파일 처리 중 오류가 발생했습니다: {e}")

        st.markdown("---")
        st.subheader("확정 채널 리스트")
        st.caption("표의 셀을 더블클릭하여 직접 수정하거나, 행을 선택하고 삭제 아이콘을 눌러 제거할 수 있습니다. '+' 버튼으로 새 행을 추가할 수도 있습니다.")
        edited_media_df = st.data_editor(
            st.session_state.media_mix_df,
            num_rows="dynamic",
            use_container_width=True,
            key="media_mix_editor"
        )
        st.session_state.media_mix_df = edited_media_df

        st.download_button(label="현재 목록 CSV로 내보내기", data=st.session_state.media_mix_df.to_csv(index=False).encode('utf-8-sig'), file_name='media_mix.csv', mime='text/csv')

    st.divider()

    def render_add_organic_form():
        with st.form(key="add_organic_form"):
            # ... (form content remains the same)
            st.subheader("🌱 새로운 자연 유입 국가 추가")
            default_org = st.session_state.config['default_rows']['organic']
            country_val = st.text_input("국가", default_org['country'])
            st.markdown("<h6>성과 지표</h6>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown("**일별 설치 수**")
                installs_loc_val = st.number_input("평균", value=default_org['daily_installs']['loc'], key="org_inst_loc")
                installs_scale_val = st.number_input("표준편차", value=default_org['daily_installs']['scale'], key="org_inst_scale")
            with m2:
                st.markdown("**결제 전환율**")
                pcr_a_val = st.number_input("성공 (a)", value=default_org['payer_conversion_rate']['a'], key="org_pcr_a")
                pcr_b_val = st.number_input("실패 (b)", value=default_org['payer_conversion_rate']['b'], key="org_pcr_b")
            with m3:
                st.markdown("**ARPPU D30**")
                arppu_loc_val = st.number_input("평균", value=default_org['arppu_d30']['loc'], key="org_arppu_loc")
                arppu_scale_val = st.number_input("표준편차", value=default_org['arppu_d30']['scale'], key="org_arppu_scale")
            st.markdown("<h6>리텐션 지표 (평균값만 입력)</h6>", unsafe_allow_html=True)
            r1, r2, r3, r4 = st.columns(4)
            ret_d1_loc = r1.number_input("D1", value=default_org['retention_d1']['loc'], key="org_d1_loc", format="%.4f")
            ret_d7_loc = r2.number_input("D7", value=default_org['retention_d7']['loc'], key="org_d7_loc", format="%.4f")
            ret_d14_loc = r3.number_input("D14", value=default_org['retention_d14']['loc'], key="d14_loc", format="%.4f")
            ret_d30_loc = r4.number_input("D30", value=default_org['retention_d30']['loc'], key="d30_loc", format="%.4f")
                
            submitted = st.form_submit_button("국가 추가")
            if submitted:
                new_row_data = copy.deepcopy(default_org)
                new_row_data['country'] = country_val
                new_row_data['daily_installs'] = {'loc': installs_loc_val, 'scale': installs_scale_val}
                new_row_data['payer_conversion_rate'] = {'a': pcr_a_val, 'b': pcr_b_val}
                new_row_data['arppu_d30'] = {'loc': arppu_loc_val, 'scale': arppu_scale_val}
                new_row_data['retention_d1'] = {'loc': ret_d1_loc}
                new_row_data['retention_d7'] = {'loc': ret_d7_loc}
                new_row_data['retention_d14'] = {'loc': ret_d14_loc}
                new_row_data['retention_d30'] = {'loc': ret_d30_loc}
                flat_row = flatten_scenario_df({'organic_assumptions': [new_row_data]}, 'organic')
                flat_row.drop(columns=[col for col in flat_row.columns if 'retention' in col and 'scale' in col], inplace=True, errors='ignore')
                st.session_state.organic_df = pd.concat([st.session_state.organic_df, flat_row], ignore_index=True)
                st.session_state.show_add_organic_form = False
                st.rerun()
    
    if st.session_state.show_add_organic_form:
        render_add_organic_form()

    with st.expander("5. 자연 유입 설정", expanded=not st.session_state.show_add_organic_form):
        st.subheader("국가 추가 방식 선택")
        if st.button("🌱 개별 국가 추가", key="add_organic_country_btn"):
            st.session_state.show_add_organic_form = True
            st.rerun()
        
        st.markdown("---")
        st.subheader("CSV 일괄 업로드")

        col1, col2 = st.columns(2)
        with col1:
            organic_template_df = pd.DataFrame(columns=st.session_state.organic_df.columns)
            st.download_button(
                label="📦 CSV 템플릿 다운로드",
                data=organic_template_df.to_csv(index=False).encode('utf-8-sig'),
                file_name='organic_inflow_template.csv',
                mime='text/csv',
            )
        with col2:
            uploaded_organic_file = st.file_uploader("📂 CSV 파일 업로드", type="csv", key="organic_uploader")
        
        if uploaded_organic_file is not None:
            try:
                new_organic_df = pd.read_csv(uploaded_organic_file)
                if list(new_organic_df.columns) == list(st.session_state.organic_df.columns):
                    st.session_state.organic_df = pd.concat(
                        [st.session_state.organic_df, new_organic_df],
                        ignore_index=True
                    )
                    st.success("자연 유입 국가 목록이 성공적으로 업데이트되었습니다!")
                else:
                    st.error("CSV 파일의 컬럼이 기존 템플릿과 일치하지 않습니다.")
            except Exception as e:
                st.error(f"파일 처리 중 오류가 발생했습니다: {e}")

        st.markdown("---")
        st.subheader("확정 국가 리스트")
        st.caption("표의 셀을 더블클릭하여 직접 수정하거나, 행을 선택하고 삭제 아이콘을 눌러 제거할 수 있습니다.")
        edited_organic_df = st.data_editor(
            st.session_state.organic_df,
            num_rows="dynamic",
            use_container_width=True,
            key="organic_editor"
        )
        st.session_state.organic_df = edited_organic_df

    # [FIX] Logic to clear results if settings have changed
    current_scenario_template_for_check = {
        'media_mix': reconstruct_scenario_from_df(st.session_state.media_mix_df.copy(), 'media_mix'),
        'organic_assumptions': reconstruct_scenario_from_df(st.session_state.organic_df.copy(), 'organic')
    }
    current_settings_for_check = {
        "project_info": st.session_state.project_info,
        "assumptions": st.session_state.assumptions,
        "scenario_template": current_scenario_template_for_check,
        "arppu_choice": st.session_state.arppu_choice
    }
    current_signature = json.dumps(current_settings_for_check, default=convert_numpy_to_native, sort_keys=True)
    
    if st.session_state.last_run_settings_signature and current_signature != st.session_state.last_run_settings_signature:
        st.session_state.simulation_results.clear()
        st.session_state.last_run_settings_signature = None
        st.toast("⚙️ 설정이 변경되었습니다. 대시보드에서 시뮬레이션을 다시 실행해주세요.")


elif st.session_state.page == "🔍 모델 신뢰도 평가":
    st.header("🔍 모델 신뢰도 평가")
    st.subheader("1. 수렴 테스트")
    st.info("시뮬레이션 횟수를 늘려감에 따라 결과값이 안정적으로 수렴하는지 확인합니다.")
    if st.button("수렴 테스트 실행"):
        with st.spinner("수렴 테스트를 위한 시뮬레이션을 실행 중입니다..."):
            convergence_scenario = {
                'media_mix': reconstruct_scenario_from_df(st.session_state.media_mix_df.copy(), 'media_mix'),
                'organic_assumptions': reconstruct_scenario_from_df(st.session_state.organic_df.copy(), 'organic')
            }
            
            sim = AdvancedLaunchSimulator(st.session_state.project_info, st.session_state.assumptions, st.session_state.assumptions['monte_carlo']['num_simulations'], convergence_scenario, {'scenario': 'ARPPU D30 (기준)', 'uplift_rate': 1.0})
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
                        backtest_scenario = {
                            'media_mix': reconstruct_scenario_from_df(st.session_state.media_mix_df.copy(), 'media_mix'),
                            'organic_assumptions': reconstruct_scenario_from_df(st.session_state.organic_df.copy(), 'organic')
                        }
                        
                        settings_for_backtest = {
                            "project_info": st.session_state.project_info,
                            "assumptions": st.session_state.assumptions,
                            "num_simulations": 300, # Using a fixed number for backtesting
                            "scenario_template": backtest_scenario,
                            "arppu_params": {'scenario': 'ARPPU D30 (기준)', 'uplift_rate': 1.0}
                        }
                        backtest_signature = json.dumps(settings_for_backtest, default=convert_numpy_to_native, sort_keys=True)
                        all_df_backtest, _ = run_cached_simulation(backtest_signature)
                        
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

