import numpy as np
import pandas as pd
import copy
from typing import Dict, Tuple, Callable, Optional
from scipy.stats import norm, beta, weibull_min, lognorm
from scipy.optimize import curve_fit

def weibull_survival_func(t, k, lam):
    return np.exp(-((t / lam) ** k))

def fit_retention_curve(retention_days, retention_values):
    try:
        popt, _ = curve_fit(weibull_survival_func, retention_days, retention_values, bounds=([0.1, 1], [2., 100.]))
        return popt
    except RuntimeError:
        return 0.5, 30.0

class AdvancedLaunchSimulator:
    def __init__(self, project_info: Dict, assumptions: Dict, num_simulations: int,
                 scenario_template: Dict, arppu_params: Dict):
        self.project_info = project_info
        self.assumptions = assumptions
        self.dist_assumptions = assumptions.get('distribution_assumptions', {})
        self.num_simulations = num_simulations
        self.arppu_params = arppu_params
        # [FIX] Do not modify the original scenario template. Uplift is applied dynamically.
        self.scenario_data = copy.deepcopy(scenario_template)
        self.simulation_days = self.project_info.get('ltv_duration_days', self.project_info['marketing_duration_days'])
        self.marketing_days = self.project_info['marketing_duration_days']
        self.daily_budget_dist = self._generate_budget_pacing_curve()
        self.daily_ltv_dist = self._generate_ltv_curve()

    def _sample_param(self, param_config: Dict, deterministic: bool = False) -> float:
        param_config = param_config.copy()
        if 'dist' not in param_config and 'param_name' in param_config:
            param_config['dist'] = self.dist_assumptions.get(param_config['param_name'], {}).get('dist')
        if deterministic:
            if param_config.get('dist') == 'beta':
                a, b = param_config.get('a', 0), param_config.get('b', 0)
                return a / (a + b) if (a + b) > 0 else 0
            return param_config.get('loc', 0)

        dist = param_config.get('dist')
        if not dist: raise ValueError(f"Distribution not defined for param: {param_config.get('param_name')}")
        
        loc = param_config.get('loc', 0)
        scale = param_config.get('scale', 0)
        a = param_config.get('a', 0)
        b = param_config.get('b', 0)

        if dist == 'norm': return max(0, norm.rvs(loc=loc, scale=scale))
        elif dist == 'lognorm':
            if loc <= 0: return 0.0
            sigma_sq = np.log((scale / loc)**2 + 1)
            mu = np.log(loc) - 0.5 * sigma_sq
            return lognorm.rvs(s=np.sqrt(sigma_sq), scale=np.exp(mu))
        elif dist == 'beta':
            if a > 0 and b > 0: return beta.rvs(a=a, b=b)
            else: return 0.0
        else: raise ValueError(f"Unsupported distribution: {dist}")

    def _run_single_simulation(self, deterministic: bool = False) -> pd.DataFrame:
        all_channel_data = []
        self._simulate_paid_channels(deterministic, all_channel_data)
        self._simulate_organic_channels(deterministic, all_channel_data)
        return all_channel_data

    def _simulate_paid_channels(self, deterministic: bool, results_list: list):
        adstock_cfg = self.assumptions['adstock']
        decay_rate, saturation_alpha = adstock_cfg['decay_rate'], adstock_cfg['saturation_alpha']
        daily_spend_total = self.project_info['target_budget'] * self.daily_budget_dist

        for os_mix in self.scenario_data['media_mix']:
            for country_mix in os_mix['channels']:
                for media_mix in country_mix['media']:
                    ratio = os_mix['budget_ratio'] * country_mix['budget_ratio'] * media_mix['budget_ratio']
                    channel_daily_spend = daily_spend_total * ratio
                    cpi = self._sample_param({**media_mix['cpi'], 'param_name': 'cpi'}, deterministic)
                    pcr = self._sample_param({**media_mix['payer_conversion_rate'], 'param_name': 'payer_conversion_rate'}, deterministic)
                    
                    # [FIX] Apply uplift rate dynamically to the sampled base ARPPU value
                    uplift_rate = self.arppu_params.get('uplift_rate', 1.0)
                    base_arppu = self._sample_param({**media_mix['arppu_d30'], 'param_name': 'arppu_d30'}, deterministic)
                    arppu = base_arppu * uplift_rate

                    adstock = np.zeros(self.simulation_days)
                    for day in range(1, self.marketing_days):
                        adstock[day] = channel_daily_spend[day-1] + adstock[day-1] * decay_rate
                    
                    # Ensure adstock has the same length as simulation_days
                    if len(adstock) < self.simulation_days:
                        adstock = np.pad(adstock, (0, self.simulation_days - len(adstock)), 'constant')

                    avg_spend = np.mean(channel_daily_spend[channel_daily_spend > 0]) if np.any(channel_daily_spend > 0) else 0
                    effective_spend = avg_spend * (adstock / avg_spend)**saturation_alpha if avg_spend > 0 else np.zeros_like(adstock)
                    cohort_installs = effective_spend / cpi if cpi > 0 else np.zeros(self.simulation_days)
                    daily_ltv_per_user = (pcr * arppu) * self.daily_ltv_dist
                    daily_revenues = np.convolve(cohort_installs, daily_ltv_per_user, mode='full')[:self.simulation_days]

                    results_list.append({
                        "os": os_mix['os'], "country": country_mix['country'], "name": media_mix['name'],
                        "product": media_mix['product'], "type": media_mix['type'], "spend": channel_daily_spend,
                        "installs": cohort_installs, "revenue": daily_revenues, "pcr": pcr
                    })

    def _simulate_organic_channels(self, deterministic: bool, results_list: list):
        for org_mix in self.scenario_data.get('organic_assumptions', []):
            installs_per_day = self._sample_param({**org_mix['daily_installs'], 'param_name': 'daily_installs'}, deterministic)
            pcr = self._sample_param({**org_mix['payer_conversion_rate'], 'param_name': 'payer_conversion_rate'}, deterministic)
            
            # [FIX] Apply uplift rate dynamically to the sampled base ARPPU value
            uplift_rate = self.arppu_params.get('uplift_rate', 1.0)
            base_arppu = self._sample_param({**org_mix['arppu_d30'], 'param_name': 'arppu_d30'}, deterministic)
            arppu = base_arppu * uplift_rate
            
            cohort_installs = np.zeros(self.simulation_days)
            cohort_installs[:self.marketing_days] = installs_per_day
            
            daily_ltv_per_user = (pcr * arppu) * self.daily_ltv_dist
            daily_revenues = np.convolve(cohort_installs, daily_ltv_per_user, mode='full')[:self.simulation_days]

            results_list.append({
                "os": "N/A", "country": org_mix['country'], "name": "Organic", "product": "N/A", "type": "Organic",
                "spend": np.zeros(self.simulation_days), "installs": cohort_installs, "revenue": daily_revenues, "pcr": pcr
            })

    def _generate_budget_pacing_curve(self) -> np.ndarray:
        pacing_cfg = self.assumptions.get('budget_pacing')
        burst_days = min(pacing_cfg['burst_days'], self.marketing_days)
        intensity = pacing_cfg['burst_intensity']
        normal_days = self.marketing_days - burst_days
        if (burst_days * intensity + normal_days) == 0: return np.zeros(self.simulation_days)
        normal_rate = 1.0 / (burst_days * intensity + normal_days)
        burst_rate = normal_rate * intensity
        
        budget_curve = np.zeros(self.simulation_days)
        budget_curve[:burst_days] = burst_rate
        budget_curve[burst_days:self.marketing_days] = normal_rate
        return budget_curve

    def _generate_ltv_curve(self) -> np.ndarray:
        ltv_cfg = self.assumptions['ltv_curve']
        days = np.arange(1, self.simulation_days + 1)
        if ltv_cfg['type'] == 'weibull':
            shape, scale = ltv_cfg['shape'], ltv_cfg['scale']
            cumulative_dist = weibull_min.cdf(days, c=shape, scale=scale)
            return np.diff(np.insert(cumulative_dist, 0, 0))
        raise ValueError(f"Unsupported LTV curve type: {ltv_cfg['type']}")

    def run_monte_carlo(self, progress_callback: Optional[Callable] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        all_results_list = []
        for i in range(self.num_simulations):
            result_list = self._run_single_simulation(deterministic=False)
            for channel_data in result_list:
                df = pd.DataFrame({
                    'day': np.arange(1, self.simulation_days + 1), 'spend': channel_data['spend'],
                    'installs': channel_data['installs'], 'revenue': channel_data['revenue'], 'pcr': channel_data['pcr']
                })
                df['sim_id'] = i
                df['os'], df['country'], df['name'], df['product'], df['type'] = \
                    channel_data['os'], channel_data['country'], channel_data['name'], channel_data['product'], channel_data['type']
                all_results_list.append(df)
        
        all_df = pd.concat(all_results_list, ignore_index=True)
        all_df.sort_values(by=['sim_id', 'os', 'country', 'name', 'day'], inplace=True)
        
        all_df['paying_users'] = all_df['installs'] * all_df['pcr']
        
        all_df['cum_spend'] = all_df.groupby(['sim_id', 'os', 'country', 'name'])['spend'].cumsum()
        all_df['cum_revenue'] = all_df.groupby(['sim_id', 'os', 'country', 'name'])['revenue'].cumsum()
        all_df['paid_roas'] = all_df['cum_revenue'] / all_df['cum_spend'].where(all_df['cum_spend'] > 0, np.nan)
        
        final_summary = all_df.groupby('sim_id').agg(
            total_revenue=('revenue', 'sum'),
            paid_spend=('spend', 'sum'),
            total_installs=('installs', 'sum'),
            total_paying_users=('paying_users', 'sum')
        ).reset_index()

        paid_metrics = all_df[all_df['type'] != 'Organic'].groupby('sim_id').agg(
            paid_revenue=('revenue', 'sum'),
            paid_installs=('installs', 'sum'),
            paid_paying_users=('paying_users', 'sum')
        ).reset_index()

        final_summary = pd.merge(final_summary, paid_metrics, on='sim_id', how='left').fillna(0)

        final_summary['total_profit'] = final_summary['total_revenue'] - final_summary['paid_spend']
        final_summary['paid_roas'] = final_summary['paid_revenue'] / final_summary['paid_spend'].where(final_summary['paid_spend'] > 0, np.nan)
        
        final_summary['blended_cpi'] = final_summary['paid_spend'] / final_summary['total_installs'].where(final_summary['total_installs'] > 0, np.nan)
        final_summary['blended_arpu'] = final_summary['total_revenue'] / final_summary['total_installs'].where(final_summary['total_installs'] > 0, np.nan)
        final_summary['blended_pcr'] = final_summary['total_paying_users'] / final_summary['total_installs'].where(final_summary['total_installs'] > 0, np.nan)
        final_summary['blended_arppu'] = final_summary['total_revenue'] / final_summary['total_paying_users'].where(final_summary['total_paying_users'] > 0, np.nan)

        final_summary['paid_cpi'] = final_summary['paid_spend'] / final_summary['paid_installs'].where(final_summary['paid_installs'] > 0, np.nan)
        final_summary['paid_arppu'] = final_summary['paid_revenue'] / final_summary['paid_paying_users'].where(final_summary['paid_paying_users'] > 0, np.nan)
        
        return all_df, final_summary

    def get_retention_model_details(self) -> Dict:
        if not self.scenario_data['media_mix'] or not self.scenario_data['media_mix'][0]['channels'] or not self.scenario_data['media_mix'][0]['channels'][0]['media']:
            return {"original_days": [], "original_values": [], "fitted_days": [], "fitted_curve": [], "k": 0, "lambda": 0}
        rep_channel = self.scenario_data['media_mix'][0]['channels'][0]['media'][0]
        original_days = np.array([1, 7, 14, 30])
        original_values = np.array([
            rep_channel['retention_d1']['loc'], rep_channel['retention_d7']['loc'],
            rep_channel['retention_d14']['loc'], rep_channel['retention_d30']['loc']
        ])
        k, lam = fit_retention_curve(original_days, original_values)
        fitted_days = np.arange(0, 91)
        fitted_curve = weibull_survival_func(fitted_days, k, lam)
        return {"original_days": original_days, "original_values": original_values, "fitted_days": fitted_days, "fitted_curve": fitted_curve, "k": k, "lambda": lam}

    def run_convergence_test(self) -> pd.DataFrame:
        convergence_data, all_final_roas = [], []
        checkpoints = np.unique(np.linspace(10, self.num_simulations, 20, dtype=int))
        for i in range(self.num_simulations):
            result_df_list = self._run_single_simulation(deterministic=False)
            paid_revenue = sum(ch['revenue'].sum() for ch in result_df_list if ch['type'] != 'Organic')
            paid_spend = sum(ch['spend'].sum() for ch in result_df_list if ch['type'] != 'Organic')
            final_roas = paid_revenue / paid_spend if paid_spend > 0 else 0
            all_final_roas.append(final_roas)
            if (i + 1) in checkpoints:
                convergence_data.append({
                    "sim_count": i + 1, "median_roas": np.median(all_final_roas),
                    "p10_roas": np.quantile(all_final_roas, 0.1), "p90_roas": np.quantile(all_final_roas, 0.9)
                })
        return pd.DataFrame(convergence_data)

