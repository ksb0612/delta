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

def create_cost_efficiency_analysis(all_df: pd.DataFrame) -> go.Figure:
    """비용 효율성 분석: 채널별 CPI vs ROAS 산점도"""
    # [FIX] 채널별 '전체 기간'의 지출과 설치를 '합산'하여 평균 CPI를 계산합니다.
    # 각 시뮬레이션 실행별로 채널의 최종 성과를 먼저 집계합니다.
    channel_summary = all_df[all_df['type'] != 'Organic'].groupby(['sim_id', 'os', 'country', 'name']).agg(
        total_spend=('spend', 'sum'),
        total_revenue=('revenue', 'sum'),
        total_installs=('installs', 'sum')
    ).reset_index()

    # 각 채널의 중앙값 성과를 사용하여 대표 값을 도출합니다.
    median_channel_performance = channel_summary.groupby(['os', 'country', 'name']).median().reset_index()
    
    # CPI와 ROAS 계산
    median_channel_performance['avg_cpi'] = median_channel_performance['total_spend'] / median_channel_performance['total_installs'].replace(0, np.nan)
    median_channel_performance['final_roas'] = median_channel_performance['total_revenue'] / median_channel_performance['total_spend'].replace(0, np.nan)
    
    # NaN 값 제거 및 데이터 유효성 검사
    median_channel_performance = median_channel_performance.dropna(subset=['avg_cpi', 'final_roas'])
    
    if len(median_channel_performance) == 0:
        fig = go.Figure()
        fig.add_annotation(text="분석할 데이터가 없습니다.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title_text="<b>비용 효율성 분석</b>")
        return fig
    
    # 채널명 및 버블 사이즈 생성
    median_channel_performance['channel_label'] = median_channel_performance['os'] + '_' + median_channel_performance['country'] + '_' + median_channel_performance['name']
    max_spend = median_channel_performance['total_spend'].max()
    median_channel_performance['bubble_size'] = (median_channel_performance['total_spend'] / max_spend) * 40 + 15 if max_spend > 0 else 20
    
    fig = go.Figure()
    
    # 차트 생성
    fig.add_trace(go.Scatter(
        x=median_channel_performance['avg_cpi'],
        y=median_channel_performance['final_roas'],
        mode='markers+text',
        marker=dict(
            size=median_channel_performance['bubble_size'],
            color=median_channel_performance['final_roas'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="ROAS"),
            line=dict(width=2, color='white')
        ),
        text=median_channel_performance['channel_label'],
        textposition="top center",
        textfont=dict(size=10),
        name='채널별 효율성',
        hovertemplate='<b>%{text}</b><br>CPI: %{x:,.0f}원<br>ROAS: %{y:.2f}<br>총 지출: %{customdata:,.0f}원<extra></extra>',
        customdata=median_channel_performance['total_spend']
    ))
    
    fig.add_hline(y=1.0, line_dash="dash", line_color="red", 
                  annotation_text="손익분기 ROAS (1.0)", annotation_position="bottom right")
    
    fig.update_layout(
        title_text="<b>비용 효율성 분석: 채널별 CPI vs ROAS</b>",
        xaxis_title="평균 CPI (원)",
        yaxis_title="최종 ROAS",
        showlegend=False,
        height=500
    )
    
    return fig

def create_performance_contribution_analysis(all_df: pd.DataFrame) -> go.Figure:
    """성과 기여도 분석: 채널별 매출 기여도와 지출 비중"""
    # 채널별 최종 누적값 계산 - 수정된 버전
    channel_final = all_df[all_df['type'] != 'Organic'].groupby(['os', 'country', 'name']).agg({
        'cum_revenue': lambda x: x.iloc[-1] if len(x) > 0 else 0,
        'cum_spend': lambda x: x.iloc[-1] if len(x) > 0 else 0,
        'installs': 'sum'
    }).reset_index()
    
    # 전체 대비 비중 계산
    total_revenue = channel_final['cum_revenue'].sum()
    total_spend = channel_final['cum_spend'].sum()
    
    if total_revenue == 0 or total_spend == 0:
        # 데이터가 없을 경우 빈 차트 반환
        fig = go.Figure()
        fig.add_annotation(
            text="분석할 데이터가 없습니다.",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(title_text="<b>성과 기여도 분석</b>")
        return fig
    
    channel_final['revenue_share'] = (channel_final['cum_revenue'] / total_revenue) * 100
    channel_final['spend_share'] = (channel_final['cum_spend'] / total_spend) * 100
    channel_final['efficiency_ratio'] = channel_final['revenue_share'] / channel_final['spend_share'].replace(0, np.nan)
    
    # NaN 값 제거
    channel_final = channel_final.dropna(subset=['efficiency_ratio'])
    
    # 채널명 생성
    channel_final['channel_label'] = channel_final['os'] + '_' + channel_final['country'] + '_' + channel_final['name']
    
    # 서브플롯 생성
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('매출 vs 지출 기여도', '채널별 효율성 지수'),
        specs=[[{"secondary_y": False}, {"type": "bar"}]],
        horizontal_spacing=0.15
    )
    
    # 첫 번째 차트: 매출 vs 지출 기여도 산점도
    fig.add_trace(
        go.Scatter(
            x=channel_final['spend_share'],
            y=channel_final['revenue_share'],
            mode='markers+text',
            marker=dict(
                size=15,
                color=channel_final['efficiency_ratio'],
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="효율성 지수", x=0.45),
                line=dict(width=2, color='white')
            ),
            text=channel_final['channel_label'],
            textposition="top center",
            textfont=dict(size=8),
            name='기여도',
            hovertemplate='<b>%{text}</b><br>지출 비중: %{x:.1f}%<br>매출 비중: %{y:.1f}%<br>효율성 지수: %{marker.color:.2f}<extra></extra>'
        ),
        row=1, col=1
    )
    
    # 대각선 (이상적인 균형선) 추가
    max_val = max(channel_final['spend_share'].max(), channel_final['revenue_share'].max())
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='균형선',
            hoverinfo='skip',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 두 번째 차트: 효율성 지수 막대 차트
    colors = ['red' if x < 1 else 'green' for x in channel_final['efficiency_ratio']]
    
    fig.add_trace(
        go.Bar(
            x=channel_final['channel_label'],
            y=channel_final['efficiency_ratio'],
            marker_color=colors,
            name='효율성 지수',
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>효율성 지수: %{y:.2f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 효율성 기준선 추가
    fig.add_hline(y=1.0, line_dash="dash", line_color="orange", row=1, col=2)
    
    # 레이아웃 업데이트
    fig.update_xaxes(title_text="지출 비중 (%)", row=1, col=1)
    fig.update_yaxes(title_text="매출 비중 (%)", row=1, col=1)
    fig.update_xaxes(title_text="채널", row=1, col=2, tickangle=45)
    fig.update_yaxes(title_text="효율성 지수", row=1, col=2)
    
    fig.update_layout(
        title_text="<b>성과 기여도 분석</b>",
        showlegend=False,
        height=500,
        annotations=[
            dict(
                x=0.25, y=-0.2, xref="paper", yref="paper",
                text="💡 균형선 위쪽은 매출 기여도가 지출 비중보다 높은 채널",
                showarrow=False, font=dict(size=10)
            ),
            dict(
                x=0.75, y=-0.2, xref="paper", yref="paper", 
                text="💡 1.0 이상은 효율적인 채널",
                showarrow=False, font=dict(size=10)
            )
        ]
    )
    
    return fig
