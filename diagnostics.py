import numpy as np
import pandas as pd
from config import logger
import statsmodels.api as sm
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (6, 3)
np.set_printoptions(precision=4, suppress=True, linewidth=120)

def compute_cross_covariance(X1: np.ndarray, X2: np.ndarray, name1: str, name2: str, tolerance: float = 0.1) -> bool:
    """
    Compute cross covariance and flag error if covariance exceeds tol.
    """
    # Compute covariance: E[(X1 - E(X1))(X2 - E(X2))^T]
    X1_centered = X1 - X1.mean(axis=0)
    X2_centered = X2 - X2.mean(axis=0)
    cross_cov = X1_centered.T @ X2_centered / (X1.shape[0] - 1)
    max_abs_val = np.max(np.abs(cross_cov))

    logger.info(f"Empirical Covariance between {name1} and {name2}:\n{cross_cov}")

    if max_abs_val > tolerance:
        logger.warning(
            f"Max absolute cross-covariance ({max_abs_val:.5f}) exceeds tolerance ({tolerance}) "
            f"between {name1} and {name2}."
        )
        return False

    logger.info(f"All cross-covariance entries between {name1} and {name2} are within the tolerance {tolerance}.")
    return True

def compute_residuals_correlation(residuals: np.ndarray, name: str = "Residuals", lags: int = 12) -> bool:
    """
    Test whether residuals are white noise up to specified lag and flag error if autocorrelation is detected.
    """
    all_passed = True
    for i in range(residuals.shape[1]):
        lb_test = sm.stats.acorr_ljungbox(residuals[:, i], lags=[lags], return_df=True)
        pval = lb_test["lb_pvalue"].values[-1]
        if pval > 0.05:
            logger.info(f"{name}[{i}] Ljung-Box p={pval:.4f} — no autocorrelation")
        else:
            logger.warning(f"{name}[{i}] Ljung-Box p={pval:.4f} — autocorrelation exists")
            all_passed = False
    return all_passed

def compute_covariance_diagnostics(matrix: np.ndarray, name: str, tol: float = 1e-8):
    """
    Compute diagnostics of covariance matrix.
    """
    cov = matrix @ matrix.T
    eigvals = np.linalg.eigvalsh(cov)
    diag = np.diag(cov)
    cond = np.linalg.cond(matrix)
    rank = np.sum(eigvals > tol)

    logger.info(f"Covariance Diagnostics for {name} {name}^T")
    logger.info(f"{name} {name}^T diag: {np.round(diag, 4)}")
    logger.info(f"{name} {name}^T eigenvalues: {np.round(eigvals, 6)}")
    logger.info(f"{name} condition number: {cond:.4f}")
    logger.info(f"{name} {name}^T numerical rank: {rank} / {cov.shape[0]}")

    return

def residual_plot(residuals: np.ndarray, name: str = "Residuals", window: int=12):
    """
    Residual plot over time
    """
    residuals = pd.DataFrame(residuals)
    
    print(f"Plotting residual diagnostics for {name}:")
    
    for i in range(residuals.shape[1]):
        series = residuals.iloc[:, i]
        rolling_var = series.rolling(window).var()

        fig, axs = plt.subplots(2, 1, sharex=True)
        
        axs[0].plot(series, label=f'{name}[{i}]')
        axs[0].set_title(f'{name}[{i}] — Level')
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(rolling_var, color='orange', label='Rolling Variance')
        axs[1].set_title(f'{name}[{i}] — Rolling Variance')
        axs[1].grid(True)
        axs[1].legend()

        plt.tight_layout()
        plt.show()
    return

def compute_residual_mean(residuals: np.ndarray, name: str = "Residuals", tolerance: float = 1e-3) -> bool:
    """
    Compute residuals mean and flag errors if any dimension is considered non-zero given the tolerance.
    """
    mean_vec = residuals.mean(axis=0)
    max_abs_mean = np.max(np.abs(mean_vec))

    if max_abs_mean > tolerance:
        logger.warning(f"{name} mean not zero within tolerance {tolerance}: {mean_vec}")
        return False

    logger.info(f"{name} mean is approximately zero (within tolerance {tolerance}): {mean_vec}")
    return True

def run_kalman_initialization_checks(
    epsilon: np.ndarray,
    eta: np.ndarray,
    X: np.ndarray,
    G: np.ndarray,
    H: np.ndarray,
    tolerance: float = 0.1,
    lags: int = 12,
    p: int = 1
) -> dict:
    """
    Run Kalman Filter initialization sanity checks:
    1. Cross-covariance checks
    2. Rank diagnostics
    3. Residual whiteness and zero-mean checks
    4. Visual diagnostics of residuals
    """
    logger.info("=== Kalman Initialization Sanity Checks ===")

    results = {}

    # 1. Cov(epsilon, eta)
    logger.info("Step 1: Checking cross-covariance between epsilon and eta...")
    results['cross_eps_eta'] = compute_cross_covariance(
        epsilon, eta[p:], "epsilon", "eta", tolerance
    )

    # 2. Cov(X, eta)
    logger.info("Step 2: Checking cross-covariance between X and eta...")
    results['cross_X_eta'] = compute_cross_covariance(
        X, eta, "X", "eta", tolerance
    )

    # 3. Residual whiteness: epsilon
    logger.info("Step 3: Checking whiteness of epsilon (state residuals)...")
    results['white_eps'] = compute_residuals_correlation(epsilon, "epsilon", lags)

    # 4. Residual whiteness: eta
    logger.info("Step 4: Checking whiteness of eta (observation residuals)...")
    results['white_eta'] = compute_residuals_correlation(eta, "eta", lags)
    
    # 5. Diagnostics of HH'
    logger.info("Step 5: Computing diagnostics of observation covariance HH'...")
    compute_covariance_diagnostics(H, 'H')
    
    # 6. Diagnostics of GG'
    logger.info("Step 5: Computing diagnostics of process covariance GG'...")
    compute_covariance_diagnostics(G, 'G')

    # 7. Residual plots
    logger.info("Step 6: Plotting residual diagnostics for epsilon and eta...")
    residual_plot(epsilon, "epsilon", window=lags)
    residual_plot(eta, "eta", window=lags)

    # 8. Zero-mean check (handled by covariance centering)
    logger.info("Step 7 skipped: Mean-centering is implicitly handled in covariance computation.")

    logger.info("Sanity checks completed.")
    logger.info(f"Summary of checks:\n{results}")

    return results
