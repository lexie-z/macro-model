import jax.numpy as jnp
import jax
import logging
from functools import partial
from dataclasses import dataclass
from typing import Optional, Any
jnp.set_printoptions(precision=4, suppress=True, linewidth=120)
logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class KalmanFilterConfig:
    certainty_factor: float = 1.0  
    use_pca_init: bool = False     
    log_first_steps: int = 5       # how many first steps to log Kalman gain
    
class KalmanFilter:
    def __init__(self, A: jnp.ndarray, B: jnp.ndarray, G: jnp.ndarray, H: jnp.ndarray, config: Optional[KalmanFilterConfig] = None):
        """
        Initialize Kalman Filter.
        Args:
            A (jnp.ndarray): State transition matrix.
            B (jnp.ndarray): Observation matrix.
            G (jnp.ndarray): Process noise input matrix.
            H (jnp.ndarray): Measurement noise input matrix.
            config (KalmanFilterConfig, optional): Initial configuration.
        """
        self.A = A
        self.B = B
        self.G = G
        self.H = H
        self.config = config if config else KalmanFilterConfig()
        self.k = A.shape[0]
        
        # State tracking
        self.X_filt: Optional[jnp.ndarray] = None
        self.P_all: Optional[Any] = None
        self.innovations: Optional[Any] = None
        self.S_list: Optional[Any] = None
        self.log_likelihood: Optional[float] = None

    @partial(jax.jit, static_argnames=["self"])
    def _filter_core(self, Y, X0, P0):
        """
        Construct standard kalman filter
        """
        # recover covariance matrix
        Q = self.G @ self.G.T
        R = self.H @ self.H.T

        def step(carry, yt):
            x, P = carry
            # Prediction phase
            x_pred = self.A @ x  # priori state estimate
            P_pred = self.A @ P @ self.A.T + Q  # priori state covariance

            # Update phase
            y_pred = self.B @ x_pred
            innovation = yt - y_pred
            S = self.B @ P_pred @ self.B.T + R # innovation covariance
            K = jnp.linalg.solve(S, (P_pred @ self.B.T).T).T

            x_new = x_pred + K @ innovation
            P_new = P_pred - K @ self.B @ P_pred

            return (x_new, P_new), (x_new, P_new, innovation, S)

        (_, _), (X_filt, P_all, innovations, S_list) = jax.lax.scan(
            step,
            (X0, P0),
            Y
        )

        return X_filt, P_all, innovations, S_list

    def run_filter(self, Y: jnp.ndarray, X_pca: Optional[jnp.ndarray] = None):
        """
        main execution function for kalman filter
        """
        k = self.A.shape[0]

        if self.config.use_pca_init:
            if X_pca is None:
                raise ValueError("PCA initialization requested but no X_pca provided.")
            X0 = jnp.array(X_pca[0])
        else:
            X0 = jnp.zeros(k)

        P0 = jnp.eye(k) * self.config.certainty_factor

        self.X_filt, self.P_all, self.innovations, self.S_list = self._filter_core(Y, X0, P0)
        return

    @partial(jax.jit, static_argnames=["self"])
    def predict_one_step(self, X_t):
        return self.B @ self.A @ X_t

    @partial(jax.jit, static_argnames=["self"])
    def compute_loss(self, Y_target: jnp.ndarray, X_pca: Optional[jnp.ndarray] = None):
        """
        Instance loss — for inspection/debugging only. Do not use with jax.grad - non differentiable.
        """
        # Run filter if needed
        if self.X_filt is None:
            self.run_filter(Y_target, X_pca)

        # One-step ahead prediction
        Y_pred_tplus1 = (self.B @ (self.A @ self.X_filt[:-1].T)).T
        Y_true_tplus1 = Y_target[1:]

        loss = jnp.mean(jnp.square(Y_pred_tplus1 - Y_true_tplus1))
        return loss
    
    @staticmethod
    @partial(jax.jit, static_argnames=["config"])
    def compute_loss_static(params, Y_target, X_pca=None, config=None):
        """Differentiable loss function."""
        kf = KalmanFilter(
            A=params["A"], B=params["B"], G=params["G"], H=params["H"], config=config
        )
        kf.run_filter(Y_target, X_pca)
        # One-step ahead prediction
        Y_pred_tplus1 = (kf.B @ (kf.A @ kf.X_filt[:-1].T)).T
        Y_true_tplus1 = Y_target[1:]
        
        loss = jnp.mean(jnp.square(Y_pred_tplus1 - Y_true_tplus1))
        return loss
    
    