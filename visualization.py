import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from typing import Dict

def create_roas_distribution_figure(final_df: pd.DataFrame) -> go.Figure:
    """Creates a violin chart of the final ROAS distribution."""
    fig = go.Figure()
    fig.add_trace(go.Violin(y=final_df['paid_roas'], name='ROAS', box_visible=True, meanline_visible=True, points='all', jitter=0.3))
    fig.update_layout(title_text="<b>최종 ROAS 분포</b>", yaxis_title="ROAS", showlegend=False)
    return fig

def create_timeseries_figure(all_df: pd.DataFrame) -> go.Figure:
    """Creates a time series chart of predicted ROAS."""
    daily_summary = all_df[all_df['type'] != 'Organic'].groupby(['sim_id', 'day']).agg(
        cum_revenue=('cum_revenue', 'sum'),
        cum_spend=('cum_spend', 'sum')
    ).reset_index()
    daily_summary['paid_roas'] = daily_summary['cum_revenue'] / daily_summary['cum_spend'].where(daily_summary['cum_spend'] > 0, np.nan)

    grouped = daily_summary.groupby('day')['paid_roas']
    
    p10 = grouped.quantile(0.1)
    p50 = grouped.quantile(0.5)
    p90 = grouped.quantile(0.9)

    fig = go.Figure([
        go.Scatter(name='P90', x=p90.index, y=p90, mode='lines', line=dict(width=0), showlegend=False, fill=None, line_color='rgba(0,100,80,0.2)'),
        go.Scatter(x=p10.index, y=p10, mode='lines', line=dict(width=0), fill='tonexty', showlegend=True, name='80% 신뢰구간', line_color='rgba(0,100,80,0.2)'),
        go.Scatter(name='Median (P50)', x=p50.index, y=p50, mode='lines', line=dict(color='rgb(0,100,80)')),
    ])
    fig.update_layout(title_text="<b>기간별 ROAS 예측</b>", yaxis_title="누적 ROAS", xaxis_title="캠페인 진행일")
    return fig

def create_retention_curve_figure(retention_details: Dict) -> go.Figure:
    """Creates a chart for retention curve fitting results."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=retention_details['original_days'], y=retention_details['original_values'], mode='markers', name='입력된 리텐션 값', marker=dict(size=10, color='red')))
    fig.add_trace(go.Scatter(x=retention_details['fitted_days'], y=retention_details['fitted_curve'], mode='lines', name='예측된 리텐션 커브', line=dict(dash='dash')))
    k, lam = retention_details['k'], retention_details['lambda']
    fig.update_layout(title_text=f"<b>모델 상세 분석: 리텐션 커브 (k={k:.2f}, λ={lam:.2f})</b>", xaxis_title="경과일", yaxis_title="리텐션", yaxis_tickformat=".0%")
    return fig

def create_sensitivity_figure(sensitivity_df: pd.DataFrame) -> go.Figure:
    """Creates a tornado chart for sensitivity analysis."""
    df = sensitivity_df.sort_values('impact_on_roas_pct', ascending=True)
    fig = go.Figure(go.Bar(y=df['parameter'], x=df['impact_on_roas_pct'], orientation='h', marker_color=['red' if x < 0 else 'green' for x in df['impact_on_roas_pct']]))
    fig.update_layout(title_text='<b>전략 분석: 민감도 분석</b>', xaxis_title="ROAS에 미치는 영향 (%)")
    return fig
    
def create_profit_cdf_figure(final_df: pd.DataFrame) -> go.Figure:
    """Creates a Cumulative Distribution Function (CDF) chart for profit."""
    sorted_profit = final_df['total_profit'].sort_values()
    cdf = np.arange(1, len(sorted_profit) + 1) / len(sorted_profit)
    loss_probability = (sorted_profit <= 0).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sorted_profit, y=cdf, mode='lines', name='CDF'))
    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="red")
    fig.update_layout(title_text=f"<b>전략 분석: 수익 누적 분포 (손실 확률: {loss_probability:.1%})</b>", xaxis_title="최종 순수익 (원)", yaxis_title="누적 확률", yaxis_tickformat=".0%", annotations=[dict(x=0, y=0.5, xref="x", yref="paper", text="손익분기점", showarrow=True, arrowhead=7, ax=80, ay=0)])
    return fig

def create_convergence_figure(convergence_data: pd.DataFrame) -> go.Figure:
    """Creates a chart for convergence test results."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=convergence_data['sim_count'], y=convergence_data['p90_roas'], fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', name='P90'))
    fig.add_trace(go.Scatter(x=convergence_data['sim_count'], y=convergence_data['p10_roas'], fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)', name='80% 신뢰구간'))
    fig.add_trace(go.Scatter(x=convergence_data['sim_count'], y=convergence_data['median_roas'], mode='lines+markers', line_color='rgb(0,100,80)', name='Median ROAS'))
    fig.update_layout(title_text="<b>모델 안정성: ROAS 수렴 과정</b>", xaxis_title="시뮬레이션 횟수", yaxis_title="ROAS", showlegend=True)
    return fig

def create_backtesting_figure(all_df_backtest: pd.DataFrame, actual_data: pd.DataFrame) -> go.Figure:
    """Creates a chart for backtesting results (actual vs. predicted)."""
    sim_grouped = all_df_backtest.groupby('day')
    sim_p10 = sim_grouped['paid_roas'].quantile(0.1)
    sim_p50 = sim_grouped['paid_roas'].quantile(0.5)
    sim_p90 = sim_grouped['paid_roas'].quantile(0.9)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sim_p90.index, y=sim_p90, fill=None, mode='lines', line_color='rgba(255,152,0,0.2)', name='P90 예측'))
    fig.add_trace(go.Scatter(x=sim_p10.index, y=sim_p10, fill='tonexty', mode='lines', line_color='rgba(255,152,0,0.2)', name='P10 예측'))
    fig.add_trace(go.Scatter(x=sim_p50.index, y=sim_p50, mode='lines', line_color='rgb(255,152,0)', name='Median 예측'))
    fig.add_trace(go.Scatter(x=actual_data['day'], y=actual_data['actual_roas'], mode='lines+markers', line_color='rgb(0,0,0)', name='실제 ROAS'))
    fig.update_layout(title_text="<b>모델 정확성: 백테스팅 (실제 vs. 예측)</b>", xaxis_title="캠페인 진행일", yaxis_title="ROAS", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def create_profit_histogram(final_df: pd.DataFrame) -> go.Figure:
    """Creates a histogram of the final total profit."""
    fig = go.Figure(data=[go.Histogram(x=final_df['total_profit'], nbinsx=50, name='수익')])
    fig.update_layout(
        title_text="<b>분포 분석: 최종 수익 히스토그램</b>",
        xaxis_title="최종 순수익 (원)",
        yaxis_title="시뮬레이션 횟수",
        bargap=0.1
    )
    return fig

def create_profit_kde_plot(final_df: pd.DataFrame) -> go.Figure:
    """Creates a Kernel Density Estimate (KDE) plot of the final total profit."""
    profit_data = final_df['total_profit'].dropna()
    
    fig = ff.create_distplot([profit_data], ['총수익'], show_hist=False, show_rug=False)
    
    fig.update_layout(
        title_text="<b>분포 분석: 최종 수익 밀도(KDE)</b>",
        xaxis_title="최종 순수익 (원)",
        yaxis_title="밀도",
        showlegend=False
    )
    return fig

