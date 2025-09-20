import streamlit as st
import pandas as pd
import numpy as np
import yaml
from typing import List, Dict, Any, Tuple

def load_config(filepath: str = "scenarios.yaml") -> Dict[str, Any]:
    """
    YAML 설정 파일을 로드합니다.

    Args:
        filepath (str): YAML 파일 경로.

    Returns:
        Dict[str, Any]: 로드된 설정 딕셔너리.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"`{filepath}` 설정 파일을 찾을 수 없습니다.")
        st.stop()
    except yaml.YAMLError as e:
        st.error(f"`{filepath}` 파일 형식에 오류가 있습니다: {e}")
        st.stop()


def format_number(num: float, is_currency: bool = False, decimals: int = 0) -> str:
    """
    숫자를 서식이 적용된 문자열로 변환합니다. (예: 1000 -> 1,000원)

    Args:
        num (float): 포맷할 숫자.
        is_currency (bool): 통화(원) 기호를 추가할지 여부.
        decimals (int): 표시할 소수점 자리수.

    Returns:
        str: 서식이 적용된 숫자 문자열.
    """
    if not isinstance(num, (int, float, np.number)) or pd.isna(num):
        return "-"
    if is_currency:
        return f"{num:,.{decimals}f}원"
    return f"{num:,.{decimals}f}"


def calculate_errors(actual: np.ndarray, predicted: np.ndarray) -> Tuple[float, float]:
    """
    실제값과 예측값 사이의 RMSE와 MAPE를 계산합니다.

    Args:
        actual (np.ndarray): 실제값 배열.
        predicted (np.ndarray): 예측값 배열.

    Returns:
        Tuple[float, float]: (RMSE, MAPE) 튜플.
    """
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    if np.sum(mask) == 0:
        return np.sqrt(np.mean((predicted - actual) ** 2)), np.inf

    rmse = np.sqrt(np.mean((predicted - actual) ** 2))
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return rmse, mape


def flatten_scenario_df(scenario_template: Dict[str, Any], part: str) -> pd.DataFrame:
    """
    YAML 시나리오의 중첩된 딕셔너리 구조를 Pandas DataFrame으로 변환합니다.

    Args:
        scenario_template (Dict[str, Any]): 원본 시나리오 템플릿.
        part (str): 변환할 부분 ('media_mix' 또는 'organic').

    Returns:
        pd.DataFrame: 평탄화된 데이터프레임.
    """
    flat_list = []
    if part == 'media_mix':
        for os_mix in scenario_template.get('media_mix', []):
            for country_mix in os_mix.get('channels', []):
                for media in country_mix.get('media', []):
                    total_ratio = os_mix.get('budget_ratio', 0) * country_mix.get('budget_ratio', 0) * media.get('budget_ratio', 0)
                    row = {
                        'os': os_mix.get('os'),
                        'country': country_mix.get('country'),
                        'name': media.get('name'),
                        'product': media.get('product'),
                        'type': media.get('type'),
                        'budget_ratio': total_ratio
                    }
                    for kpi, values in media.items():
                        if isinstance(values, dict):
                            for key, val in values.items():
                                row[f"{kpi}_{key}"] = val
                    flat_list.append(row)

    elif part == 'organic':
        for item in scenario_template.get('organic_assumptions', []):
            row = {'country': item.get('country')}
            for kpi, values in item.items():
                if isinstance(values, dict):
                    for key, val in values.items():
                        row[f"{kpi}_{key}"] = val
            flat_list.append(row)

    return pd.DataFrame(flat_list)


def reconstruct_scenario_from_df(df: pd.DataFrame, part: str) -> List[Dict[str, Any]]:
    """
    데이터 편집기(DataFrame)의 내용을 다시 YAML 시나리오 구조로 재구성합니다.

    Args:
        df (pd.DataFrame): st.data_editor에서 수정된 데이터프레임.
        part (str): 재구성할 부분 ('media_mix' 또는 'organic').

    Returns:
        List[Dict[str, Any]]: 재구성된 시나리오 리스트.
    """
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    string_cols = df.select_dtypes(include='object').columns.tolist()

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for col in string_cols:
        df[col] = df[col].astype(str).fillna('N/A')


    if part == 'media_mix':
        total_budget_sum = df['budget_ratio'].sum()
        if total_budget_sum > 0:
            df['budget_ratio_normalized'] = df['budget_ratio'] / total_budget_sum
        else:
            df['budget_ratio_normalized'] = 1 / len(df) if len(df) > 0 else 0

        media_mix = []
        for os, os_df in df.groupby('os'):
            os_channels = []
            os_budget_total = os_df['budget_ratio_normalized'].sum()
            for country, country_df in os_df.groupby('country'):
                country_budget_total = country_df['budget_ratio_normalized'].sum()
                media_list = []
                for _, row in country_df.iterrows():
                    media = {
                        'name': row['name'],
                        'product': row['product'],
                        'type': row['type'],
                        'budget_ratio': row['budget_ratio_normalized'] / country_budget_total if country_budget_total > 0 else 0
                    }
                    kpi_keys = ['cpi', 'payer_conversion_rate', 'arppu_d30', 'retention_d1', 'retention_d7', 'retention_d14', 'retention_d30']
                    for kpi in kpi_keys:
                        media[kpi] = {k.split('_')[-1]: v for k, v in row.to_dict().items() if k.startswith(kpi)}
                    media_list.append(media)

                os_channels.append({
                    'country': country,
                    'budget_ratio': country_budget_total / os_budget_total if os_budget_total > 0 else 0,
                    'media': media_list
                })

            media_mix.append({
                'os': os,
                'budget_ratio': os_budget_total,
                'channels': os_channels
            })
        return media_mix

    elif part == 'organic':
        reconstructed_list = []
        for _, row in df.iterrows():
            item = {'country': row['country']}
            kpi_keys = ['daily_installs', 'payer_conversion_rate', 'arppu_d30', 'retention_d1', 'retention_d7', 'retention_d14', 'retention_d30']
            for kpi in kpi_keys:
                item[kpi] = {k.split('_')[-1]: v for k, v in row.to_dict().items() if k.startswith(kpi)}
            reconstructed_list.append(item)
        return reconstructed_list

    return []
