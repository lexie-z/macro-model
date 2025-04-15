import numpy as np
import pandas as pd
from numpy.linalg import eigvals, qr, solve
from scipy.linalg import cholesky, LinAlgError
from config import logger
np.set_printoptions(precision=4, suppress=True, linewidth=120)
from statsmodels.tsa.api import VAR

def spectral_radius_projection(A: np.ndarray, target_radius: float = 0.98):
    eigs = eigvals(A)
    current_radius = np.max(np.abs(eigs))
    if current_radius >= 1:
        A = (target_radius / current_radius) * A
        logger.warning(f"Projected A to spectral radius {target_radius:.2f} to ensure stability.")
    else:
        logger.info("Matrix A is already stable (spectral radius < 1).")

    return A

def regularize_covariance_ridge(M: np.ndarray, epsilon=1e-2):
    logger.info(f"Applying ridge regularization to covariance matrix with ε = {epsilon}.")
    return M + epsilon * np.eye(M.shape[0])

def fit_state_space_model(X_pca: pd.DataFrame, p=1):
    """
    Fit VAR(p) on latent states to initialize A and Q.
    """
    logger.info(f"Fitting VAR({p}) model to latent state series with shape {X_pca.shape}.")
    model = VAR(X_pca)
    results = model.fit(p)

    A = results.coefs[0] # (k x k)                
    residuals = results.resid.values # (T-p x k)
    Q = np.cov(residuals.T) # (k x k)           

    # Stabilize A
    logger.info("VAR coefficients extracted. Projecting A to ensure stability.")
    A = spectral_radius_projection(A)

    # Check PD of Q and fix if needed
    try:
        G = cholesky(Q, lower=True)
        logger.info("Covariance matrix Q is positive definite. Cholesky decomposition successful.")
    except LinAlgError:
        logger.warning("Covariance matrix Q is not positive definite. Applying ridge regularization.")
        Q = regularize_covariance_ridge(Q)
        G = cholesky(Q, lower=True)
        logger.info("Cholesky decomposition after regularization successful.")
        
        # SVD check for residuals
        _, s_residual, _ = np.linalg.svd(residuals, full_matrices=False)
        logger.info(f"Singular values of residuals: {np.round(s_residual, 4)}")

    return A, G, residuals

def fit_observation_model(Y: pd.DataFrame, X: pd.DataFrame):
    """
    Fit observation model: Y_t = B X_t + H η_t
    where H is the Cholesky factor of residual covariance matrix R.
    """
    logger.info(f"Fitting OLS observation model with Y shape {Y.shape}, X shape {X.shape}.")

    Y_vals = Y.values  # (T x n)
    X_vals = X.values  # (T x k)

    # QR decomposition for stable least squares
    Q, R_qr = qr(X_vals)
    B = solve(R_qr, Q.T @ Y_vals).T  # (n x k)

    Y_pred = X_vals @ B.T  # (T x n)
    residuals = Y_vals - Y_pred  # (T x n)
    R = np.cov(residuals.T)  # (n x n)

    try:
        H = cholesky(R, lower=True)
        logger.info("Covariance matrix R is positive definite. Cholesky decomposition successful.")
    except LinAlgError:
        logger.warning("Covariance matrix R is not positive definite. Applying ridge regularization.")
        _, s_residual, _ = np.linalg.svd(residuals, full_matrices=False)
        logger.info(f"Singular values of residuals: {np.round(s_residual, 4)}")
        R = regularize_covariance_ridge(R)
        H = cholesky(R, lower=True)
        logger.info("Cholesky decomposition after regularization successful.")

    return B, H, residuals