from sklearn.preprocessing import StandardScaler
from config import logger
import numpy as np
import pandas as pd
import os
from eda import distribution, correlation
np.set_printoptions(precision=4, suppress=True, linewidth=120)

def read_data(data_path:str, daily_factors:list, monthly_factors:list, quarterly_factors:list):
    df = pd.DataFrame()
    for file_name in daily_factors+monthly_factors:
        temp_data = pd.read_csv(os.path.join(data_path, file_name+'.csv'), parse_dates=['observation_date'])
        temp_data.set_index("observation_date", inplace=True)
        temp_data.sort_index(inplace=True)
        # resample data to quarterly frequency for frequency alignment
        # shift each quarter-end index to the first day of the next quarter
        resampled_data = temp_data.resample('QE-DEC').mean()
        resampled_data.index = (resampled_data.index + pd.offsets.QuarterBegin(startingMonth=1))
        df = pd.concat([df,resampled_data], axis=1)
    
    for file_name in quarterly_factors:
        quarterly_data = pd.read_csv(os.path.join(data_path, file_name + '.csv'), parse_dates=['observation_date'])
        quarterly_data.set_index("observation_date", inplace=True)
        quarterly_data.sort_index(inplace=True)
        df = pd.concat([df, quarterly_data], axis=1)
    
    # drop initial nan values after resampling
    df.dropna(inplace=True)
    # check if all date differences are quarterly (whether skip quarter exists)
    date_diffs = df.index.to_series().diff().dt.days.dropna()
    if not date_diffs.isin([90, 91, 92]).all():
        raise ValueError("Date index is not consistently quarterly spaced.")
    return df

def log_difference(series: pd.Series) -> pd.Series:
    if (series <= 0).any():
        raise ValueError(f"Log difference requires all values to be positive. Found non-positive in {series.name}")
    return np.log(series).diff().dropna()

def recover_from_log_diff(log_diff: pd.Series, initial_value: float) -> pd.Series:
    if initial_value <= 0:
        raise ValueError("Initial value must be positive for log recovery.")
    recovered_log = np.log(initial_value) + log_diff.cumsum()
    recovered = np.exp(recovered_log)
    recovered.index = log_diff.index
    return recovered

def first_difference(series: pd.Series) -> pd.Series:
    return series.diff().dropna()

def recover_from_diff(diff_series: pd.Series, initial_value: float) -> pd.Series:
    recovered_values = np.r_[initial_value, diff_series.values].cumsum()
    recovered_index = [diff_series.index[0]] + list(diff_series.index)
    return pd.Series(recovered_values, index=recovered_index, name=diff_series.name)

def make_stationary(df, log_diff_cols, diff_cols):
    _df = df.copy(deep=True)
    for col in log_diff_cols:
        _df[col] = log_difference(_df[col])
    for col in diff_cols:
        _df[col] = first_difference(_df[col])
    return _df.dropna()

def z_score_standardize(df: pd.DataFrame):
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df),
        columns=df.columns,
        index=df.index
    )
    logger.info("Standardization complete.")
    return df_scaled, scaler

def invert_z_score(df_scaled: pd.DataFrame, scaler: StandardScaler):
    '''
    Invert Z-score standardization using the fitted scaler.
    '''
    df_original = pd.DataFrame(
        scaler.inverse_transform(df_scaled),
        columns=df_scaled.columns,
        index=df_scaled.index
    )
    logger.info("Inverted standardization.")
    return df_original

def clip_by_MAD(series: pd.Series, clip_n: float) -> pd.Series:
    '''
    Conditionally winsorize a series by MAD — only if clipping is needed.
    '''
    _series = series.copy(deep=True)
    median_window = _series.median()
    abs_deviation = np.abs(_series - median_window)
    mad = abs_deviation.median()

    lower = median_window - clip_n * mad
    upper = median_window + clip_n * mad

    # Check if any value exceeds bounds
    if (_series < lower).any() or (_series > upper).any():
        logger.info(f"Clipping outliers using MAD for {series.name} with clip_n = {clip_n}.")
        return _series.clip(lower, upper)
    else:
        return _series

def initialize_by_window(df: pd.DataFrame, clipping_factors: list = ['UNRATE'], plot: bool=False, window_size: int = 48, clip_n: float = 3.0):
    _df = df.iloc[:window_size].copy(deep=True)

    for col in clipping_factors:
        if col in _df.columns:
            _df[col] = clip_by_MAD(_df[col], clip_n=clip_n)

    _df, _scaler = z_score_standardize(_df)

    logger.info(f"Window initialized with size {window_size}. Features: {list(_df.columns)}")

    if plot: # # currently set to false for rolling window; better to save img
        distribution(_df)
        correlation(_df)

    return _df, _scaler