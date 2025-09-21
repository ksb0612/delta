import copy
import pandas as pd
from advanced_simulator import AdvancedLaunchSimulator

class StrategicAnalyzer:
    def __init__(self, config: dict):
        self.config = config

    def _run_deterministic_simulation(self, scenario_data: dict) -> float:
        """단일 결정론적 시뮬레이션을 실행하고 최종 ROAS를 반환합니다."""
        simulator = AdvancedLaunchSimulator(
            self.config['project_info'], self.config['assumptions'], 1,
            scenario_data, {'scenario': 'ARPPU D30 (기준)', 'uplift_rate': 1.0}
        )
        result_list = simulator._run_single_simulation(deterministic=True)
        
        paid_revenue = sum(ch['revenue'].sum() for ch in result_list if ch['type'] != 'Organic')
        paid_spend = sum(ch['spend'].sum() for ch in result_list if ch['type'] != 'Organic')
        
        final_roas = paid_revenue / paid_spend if paid_spend > 0 else 0
        return final_roas

    def run_sensitivity_analysis(self, scenario_data: dict) -> pd.DataFrame:
        """채널별 KPI 민감도 분석을 수행합니다."""
        base_roas = self._run_deterministic_simulation(scenario_data)
        if base_roas is None or base_roas == 0:
            return pd.DataFrame(columns=["parameter", "impact_on_roas_pct"])

        params_to_test = self.config['assumptions']['strategic_analysis']['parameters_to_test']
        change_pct = self.config['assumptions']['strategic_analysis']['change_percentage'] / 100.0
        results = []

        for os_idx, os_mix in enumerate(scenario_data.get('media_mix', [])):
            for country_idx, country_mix in enumerate(os_mix.get('channels', [])):
                for media_idx, media_mix in enumerate(country_mix.get('media', [])):
                    channel_identifier = f"{os_mix['os']}_{country_mix['country']}_{media_mix['name']}"

                    for p_info in params_to_test:
                        param, key = p_info['name'], p_info['key']
                        
                        if param in media_mix and key in media_mix[param]:
                            modified_scenario = copy.deepcopy(scenario_data)
                            target_media = modified_scenario['media_mix'][os_idx]['channels'][country_idx]['media'][media_idx]
                            
                            original_value = target_media[param][key]
                            target_media[param][key] *= (1 + change_pct)
                            
                            new_roas = self._run_deterministic_simulation(modified_scenario)
                            impact = (new_roas - base_roas) / base_roas
                            
                            results.append({
                                "parameter": f"{channel_identifier}_{param}", 
                                "impact_on_roas_pct": impact * 100
                            })
        
        if not results:
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        df['abs_impact'] = df['impact_on_roas_pct'].abs()
        df = df.sort_values('abs_impact', ascending=False).drop(columns=['abs_impact'])
        return df
