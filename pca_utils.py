import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from config import logger
plt.rcParams['figure.figsize'] = (6, 3)
np.set_printoptions(precision=4, suppress=True, linewidth=120)

def scree_plot(df:pd.DataFrame):
    pca = PCA()
    X_pca = pca.fit_transform(df)
    
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.show()

    loadings_df = pd.DataFrame(pca.components_, columns=df.columns)
    return loadings_df

def pca(df:pd.DataFrame, n_components=2, plot:bool=False):
    """
    Performs PCA on the input DataFrame and plots the time series of the top principal components.
    """
    model = PCA(n_components=n_components)
    X_pca = model.fit_transform(df)
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)], index=df.index)
    
    logger.info(f"PCA complete. Explained variance ratio: {model.explained_variance_ratio_[:n_components]}")

    if plot: # currently set to false for rolling window; better to save img 
        for col in pca_df.columns:
            plt.plot(pca_df.index, pca_df[col], label=col)
        plt.title(f'Top {n_components} Principal Components Over Time')
        plt.xlabel('Time')
        plt.ylabel('Component Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return pca_df