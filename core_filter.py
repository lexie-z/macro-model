import numpy as np
import pandas as pd
from config import logger
np.set_printoptions(precision=4, suppress=True, linewidth=120)

def kalman_filter(Y: np.ndarray, A: np.ndarray, B: np.ndarray, G: np.ndarray, H: np.ndarray, X0: np.ndarray, P0: np.ndarray):
    """
    Run Kalman Filter for the linear Gaussian state-space model.
    """
    T, k = Y.shape[0], A.shape[0]
    
    logger.info(f"Running Kalman filter for {T} time steps.")

    # recover covariance matrix
    Q = G @ G.T  # process noise covariance
    R = H @ H.T  # observation noise covariance
    
    X_filt = np.zeros((T, k))  
    P = P0.copy()              
    x = X0.copy()              

    P_all = []                # store posterior covariances
    innovations = []          # store innovations 
    S_list = []               # store innovation covariances

    for t in range(T):
        # prediction phase
        x_pred = A @ x                      # priori state estimate
        P_pred = A @ P @ A.T + Q           # priori state covariance

        # update phase 
        y_pred = B @ x_pred                
        innovation = Y[t] - y_pred         
        S = B @ P_pred @ B.T + R           # innovation covariance
        
        try:
            K = P_pred @ B.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.error(f"Numerical instability at t={t}: innovation covariance not invertible.")
            raise
        
        if t < 5:
            logger.info(f"Kalman gain K[{t}]:\n{K}") # log kalman gain from first 5 time steps

        x = x_pred + K @ innovation
        P = P_pred - K @ B @ P_pred

        # store results
        X_filt[t] = x
        P_all.append(P.copy())
        innovations.append(innovation)
        S_list.append(S)

        logger.debug(f"t={t}: |innovation|={np.linalg.norm(innovation):.4f}, Kalman gain norm={np.linalg.norm(K):.4f}")

    logger.info("Kalman filter run completed.")
    return X_filt, P_all, innovations, S_list

def run_kalman_filter(Y: np.ndarray, A: np.ndarray, B: np.ndarray, G: np.ndarray, H: np.ndarray, use_pca_init: bool=False, X_pca=None, certainty_factor: float=1):
    """
    Main function to run Kalman Filter with default or PCA-based initialization.
    """
    k = A.shape[0]  
    logger.info("Starting Kalman Filter execution...")
    logger.info(f"State dimension k={k}, certainty factor for P0={certainty_factor}")

    # Initialize state
    if use_pca_init:
        if X_pca is None:
            raise ValueError("X_pca must be provided if use_pca_init=True")
        X0 = X_pca.iloc[0].values
        logger.info("Initializing latent state from first PCA value.")
    else:
        X0 = np.zeros(k)
        logger.info("Initializing latent state as zero vector.")

    P0 = np.eye(k) * certainty_factor
    
    # Run Kalman Filter
    X_filt, P_all, innovations, S_list = kalman_filter(
        Y=Y,
        A=A,
        B=B,
        G=G,
        H=H,
        X0=X0,
        P0=P0
    )

    logger.info("Kalman Filter execution completed.")
    return X_filt, P_all, innovations, S_list