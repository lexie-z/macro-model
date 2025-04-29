import jax.numpy as jnp
import jax
from jax import jit
import logging
from functools import partial

logger = logging.getLogger(__name__)

class KalmanFilter:
    def __init__(self, A: jnp.ndarray, B: jnp.ndarray, G: jnp.ndarray, H: jnp.ndarray, 
                 X0: jnp.ndarray = None, P0: jnp.ndarray = None, certainty_factor: float = 1.0):
        """
        Initialize Kalman Filter.
        Args:
            A (jnp.ndarray): State transition matrix.
            B (jnp.ndarray): Observation matrix.
            G (jnp.ndarray): Process noise input matrix.
            H (jnp.ndarray): Measurement noise input matrix.
            X0 (jnp.ndarray, optional): Initial latent state. Defaults to zeros.
            P0 (jnp.ndarray, optional): Initial state covariance. Defaults to scaled identity.
            certainty_factor (float, optional): Scaling for initial P0. Defaults to 1.0.
        """
        self.A = A
        self.B = B
        self.G = G
        self.H = H

        self.k = A.shape[0]

        if X0 is None:
            self.X0 = jnp.zeros(self.k)
        else:
            self.X0 = X0

        if P0 is None:
            self.P0 = jnp.eye(self.k) * certainty_factor
        else:
            self.P0 = P0

    @partial(jax.jit, static_argnums=0)
    def run_filter(self, Y: jnp.ndarray):
        """
        Run standard Kalman filter.
        Args:
            Y (jnp.ndarray): Observation matrix (T x n)

        Returns:
            X_filt (jnp.ndarray): Filtered latent states (T x k)
            P_all (jnp.ndarray): Posterior covariances (T x k x k)
            innovations (jnp.ndarray): Innovations (T x n)
            S_list (jnp.ndarray): Innovation covariances (T x n x n)
        """
        T, n = Y.shape

        # recover covariance matrix
        Q = self.G @ self.G.T # process noise covariance
        R = self.H @ self.H.T # observation noise covariance

        def step(carry, yt):
            x, P = carry
            # prediction phase
            x_pred = self.A @ x  # priori state estimate
            P_pred = self.A @ P @ self.A.T + Q  # priori state covariance

            # update phase 
            y_pred = self.B @ x_pred
            innovation = yt - y_pred
            S = self.B @ P_pred @ self.B.T + R # innovation covariance
            K = jnp.linalg.solve(S, (P_pred @ self.B.T).T).T

            x_new = x_pred + K @ innovation
            P_new = P_pred - K @ self.B @ P_pred

            return (x_new, P_new), (x_new, P_new, innovation, S)

        (xf, pf), (X_filt, P_all, innovations, S_list) = jax.lax.scan(
            step, (self.X0, self.P0), Y
        )

        return X_filt, P_all, innovations, S_list

    def forecast_one_step(self, X_filt: jnp.ndarray):
        """
        One-step-ahead forecast: \hat{Y}_{t+1|t} = B A X_t
        Args:
            X_filt (jnp.ndarray): Filtered latent states (T x k)

        Returns:
            Y_pred_tplus1 (jnp.ndarray): One-step ahead forecast (T-1 x n)
        """
        X_pred_tplus1 = (self.A @ X_filt[:-1].T).T
        Y_pred_tplus1 = (self.B @ X_pred_tplus1.T).T
        return Y_pred_tplus1

    def evaluate_forecast(self, Y: jnp.ndarray):
        """
        Full evaluation pipeline: run filter, forecast, compute RMSE loss.
        Args:
            Y (jnp.ndarray): Observation matrix (T x n)

        Returns:
            rmse (float): Root Mean Squared Error of one-step-ahead forecast.
        """
        logger.info("Evaluating Kalman filter forecasting performance...")

        X_filt, _, _, _ = self.run_filter(Y)
        Y_forecast = self.forecast_one_step(X_filt)
        Y_true = Y[1:]

        mse = jnp.mean((Y_true - Y_forecast) ** 2)

        logger.info(f"Forecast MSE (training loss): {mse:.6f}")

        return mse