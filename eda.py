import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6, 3)
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import seaborn as sns
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=120)

def eda(df: pd.DataFrame, window=20, z_score=3):
    for col in df.columns:
        series = df[col]
        rolling_mean = series.rolling(window=window).mean()
        rolling_std = series.rolling(window=window).std()
        outliers = (abs(series - rolling_mean) > z_score * rolling_std)
        
        plt.plot(series, color='gray', alpha=0.5, label='Original')
        plt.plot(rolling_mean, label='Rolling Mean', linestyle='--')
        plt.plot(rolling_std, label='Rolling Std', linestyle=':')
        plt.scatter(series.index[outliers], series[outliers], 
                    color='red', label='Outliers', zorder=5)

        plt.title(f'Rolling Mean and Stdev + Outliers for {col}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
    return

def adf_test_all_features(df: pd.DataFrame, signif=0.05):
    """
    Perform the Augmented Dickey-Fuller test.
    """
    results = []
    
    for col in df.columns:
        # Run ADF test for the current feature
        result = adfuller(df[col].dropna())
        adf_stat = result[0]
        p_value = result[1]
        critical_values = result[4]
        is_stationary = p_value < signif
        
        results.append({
            'Feature': col,
            'ADF Statistic': adf_stat,
            'p-value': p_value,
            'Stationary': 'Yes' if is_stationary else 'No',
            'Critical Values': critical_values
        })
    
    results_df = pd.DataFrame(results)
    return results_df

def distribution(df: pd.DataFrame):
    for col in df.columns:
        series = df[col]
        sns.histplot(series, bins=50, edgecolor='white', kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
def correlation(df:pd.DataFrame):
    corr_matrix = df.corr()
    plt.figure(figsize=(6, 6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True, linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()