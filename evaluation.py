import numpy as np
import matplotlib.pyplot as plt
from config import logger
plt.rcParams['figure.figsize'] = (6, 3)
np.set_printoptions(precision=4, suppress=True, linewidth=120)
from sklearn.metrics import mean_squared_error, r2_score
from diagnostics import residual_plot, compute_cross_covariance, compute_residual_mean, compute_residuals_correlation

def plot_state_consistency(X_filt: np.ndarray, X_pca: np.ndarray, P_all=None):
    """
    Plot filtered state & PCA state over time with confidence band;
    Plot state norm and trace of covariance matrix over time
    """
    logger.info("Plotting state consistency...")
    # should add save image options
    T, k = X_filt.shape
    time = np.arange(T)

    for i in range(k):
        plt.plot(time, X_filt[:, i], label='Filtered X', linewidth=2)
        plt.plot(time, X_pca.iloc[:, i], label='PCA Init X', linestyle='--')

        # plot 95% confidence interval 
        if P_all is not None:
            std = np.sqrt([P[i, i] for P in P_all])
            plt.fill_between(time, X_filt[:, i] - 1.96 * std, X_filt[:, i] + 1.96 * std,
                             color='gray', alpha=0.3, label='95% CI')

        plt.title(f'State Component {i}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    # Plot state norm and trace(P)
    state_norms = np.linalg.norm(X_filt, axis=1)
    plt.plot(state_norms, label='Norm(X_filtered)')
    plt.title('Filtered State Norm')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    if P_all is not None:
        traces = [np.trace(P) for P in P_all]
        plt.plot(traces, label='Trace(P_t)')
        plt.title('Filtered State Covariance Trace')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return

def evaluate_observation_fit(Y_target: np.ndarray, X_input: np.ndarray, B: np.ndarray, label: str = "t+1 prediction"):
    """
    Evaluate observation Y_hat = B @ X for either t+1 prediction or filtered fit.
    """
    logger.info(f"Evaluating {label} accuracy...")

    Y_pred = X_input @ B.T

    for i in range(Y_target.shape[1]):
        rmse = np.sqrt(mean_squared_error(Y_target[:, i], Y_pred[:, i]))
        r2 = r2_score(Y_target[:, i], Y_pred[:, i])
        logger.info(f"Y[{i}] ({label}) — RMSE: {rmse:.4f}, R^2: {r2:.4f}")

        plt.figure(figsize=(6, 3))
        plt.plot(Y_target[:, i], label='True Y', linewidth=2)
        plt.plot(Y_pred[:, i], label=f'Predicted Y ({label})', linestyle='--')
        plt.title(f'{label} — Y[{i}]')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    return


def compute_loglik_from_kf_terms(innovations: np.ndarray, S_list: np.ndarray):
    """
    Compute total log-likelihood L(\theta | Y_{1:t}) for all time steps t
    """
    logL = 0.0
    n = innovations[0].shape[0] 
    
    logger.info("Computing total log-likelihood from Kalman innovations...")

    for t, (innovation, S) in enumerate(zip(innovations, S_list)):
        # Compute log-determinant of S 
        sign, logdet = np.linalg.slogdet(S)

        if sign != 1:
            logger.warning(f"Innovation covariance S_t at time {t} is not positive definite (sign={sign})")
            logL += -1e6  # Penalize ill-conditioned S_t
            continue
        
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning(f"Inversion failed for S[{t}]. Using pseudo-inverse.")
            S_inv = np.linalg.pinv(S)

        # Gaussian log-density for multivariate normal: 
        # -0.5 * (log |S| + innovation.T @ S^{-1} @ innovation + n log(2π))
        logL += -0.5 * (logdet + innovation.T @ S_inv @ innovation + n * np.log(2 * np.pi))

    logger.info(f"Final computed log-likelihood: {logL:.4f}")
    return logL

def evaluate_kalman_pipeline(Y, X_filt, X_pca, B, innovations, S_list, P_all):
    """
    Evaluate Kalman Filter performance:
    1. State estimate consistency
    2. Residual diagnostics
    3. Observation reconstruction accuracy
    4. Log-likelihood computation
    """
    logger.info("=== Kalman Pipeline Evaluation ===")

    # 1. State estimate consistency
    logger.info("Step 1: Evaluating state estimate consistency...")
    plot_state_consistency(X_filt, X_pca, P_all)

    # 2. Residual analysis
    if isinstance(innovations, list):
        innovations = np.vstack(innovations)
    logger.info("Step 2: Performing residual diagnostics...")
    compute_cross_covariance(X_filt, innovations, name1="X_filt", name2="Innovation")
    compute_residuals_correlation(innovations, name="Innovation")
    residual_plot(innovations, name="Innovation")
    compute_residual_mean(innovations, name="Innovation")

    # Step 3: filtered reconstruction evaluation
    logger.info("Step 3: Evaluating observation reconstruction accuracy...")
    evaluate_observation_fit(Y, X_filt, B, label="filtered")
    
    # Step 4: t+1 prediction evaluation
    logger.info("Step 4: Evaluating one-step-ahead prediction...")
    evaluate_observation_fit(Y[1:], X_filt[:-1], B, label="t+1 prediction")

    # 5. Log-likelihood evaluation
    logger.info("Step 5: Computing log-likelihood...")
    logL = compute_loglik_from_kf_terms(innovations, S_list)


    return logL