import jax.numpy as jnp
from jax.numpy.linalg import eigvals, qr, solve
from scipy.linalg import cholesky, LinAlgError
from config import logger
import pandas as pd
from statsmodels.tsa.api import VAR
jnp.set_printoptions(precision=4, suppress=True, linewidth=120)

def spectral_radius_projection(A: jnp.ndarray, target_radius: float = 0.98) -> jnp.ndarray:
    eigs = jnp.linalg.eigvals(A)
    current_radius = jnp.max(jnp.abs(eigs))
    scale = jnp.minimum(1.0, target_radius / current_radius)
    return A * scale

def regularize_covariance_ridge(M: jnp.ndarray, epsilon=1e-2):
    logger.info(f"Applying ridge regularization to covariance matrix with ε = {epsilon}.")
    return M + epsilon * jnp.eye(M.shape[0])

def fit_transition_model(X_pca: pd.DataFrame, p=1):
    """
    Fit VAR(p) on latent states to initialize A and collect residuals (epsilon).
    """
    logger.info(f"Fitting VAR({p}) model to latent state series with shape {X_pca.shape}.")
    model = VAR(X_pca)
    results = model.fit(p)

    A = jnp.array(results.coefs[0])
    residuals_eps = jnp.array(results.resid)

    # Stabilize A
    logger.info("VAR coefficients extracted. Projecting A to ensure stability.")
    A = spectral_radius_projection(A)

    return A, residuals_eps

def fit_observation_model(Y: pd.DataFrame, X: pd.DataFrame):
    """
    Fit OLS observation model: Y_t = B X_t + H η_t (where H will later be constructed).
    """
    logger.info(f"Fitting OLS observation model with Y shape {Y.shape}, X shape {X.shape}.")

    Y_vals = jnp.array(Y)
    X_vals = jnp.array(X)

    # QR decomposition for stable least squares
    Q, R_qr = qr(X_vals)
    B = solve(R_qr, Q.T @ Y_vals).T

    Y_pred = X_vals @ B.T
    residuals_eta = Y_vals - Y_pred

    return B, residuals_eta

def compute_stacked_noise_params(residuals_eps: jnp.ndarray, residuals_eta: jnp.ndarray, ridge_epsilon=1e-2):
    """
    Stack epsilon and eta residuals into xi, compute full covariance matrix of xi, apply Cholesky regularization if needed,
    and construct G_bar and H_bar.
    """
    logger.info("Stacking epsilon and eta residuals for full noise covariance estimation.")

    # Align residuals: use eps as is, drop first row of eta
    xi = jnp.hstack([residuals_eps, residuals_eta[1:]]) # (T-1, k+n)
    cov_xi = jnp.cov(xi.T)

    try:
        L = cholesky(cov_xi, lower=True)
        logger.info("Full stacked covariance matrix is positive definite. Cholesky decomposition successful.")
    except LinAlgError:
        logger.warning("Full stacked covariance matrix is not positive definite. Applying ridge regularization.")
        cov_xi = regularize_covariance_ridge(cov_xi, epsilon=ridge_epsilon)
        L = cholesky(cov_xi, lower=True)
        logger.info("Cholesky decomposition after regularization successful.")

    k = residuals_eps.shape[1]
    n = residuals_eta.shape[1]

    G_pad = jnp.hstack([jnp.eye(k), jnp.zeros((k, n))])
    H_pad = jnp.hstack([jnp.zeros((n, k)), jnp.eye(n)])

    G_bar = G_pad @ L
    H_bar = H_pad @ L

    return G_bar, H_bar, xi, L
