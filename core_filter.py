import jax.numpy as jnp
import jax
import logging
from functools import partial
from dataclasses import dataclass
from typing import Optional, Any
import jax.lax as lax
jnp.set_printoptions(precision=4, suppress=True, linewidth=120)
logger = logging.getLogger(__name__)

# set to static for JIT compatibility
# when flexibility is required, create another callback config
@dataclass(frozen=True) 
class KalmanFilterConfig:
    certainty_factor: float = 1.0  # initial P = certainty_factor * I
    use_pca_init: bool = False     
    log_first_steps: int = 5  # how many first steps to log Kalman gain
    lambda_A: float = 0   # L2 penalty over ||A||
    lambda_G: float = 0   # L2 penalty over ||G||
    lambda_H: float = 0   # L2 penalty over ||H||
    cond_eps: float = 1e-12
    gamma: float = 0.7  # Decay factor for multi-step prediction
    max_horizon: int = 1  # default to short horizon
    
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
    
    @staticmethod
    def regularize_if_ill_conditioned(S: jnp.ndarray, threshold: float = 1e5, ridge_eps: float = 1e-4):
        """
        Conditionally apply ridge regularization to S if its condition number is too high.
        """
        eigvals = jnp.linalg.eigvalsh(S)
        cond_number = jnp.max(eigvals) / (jnp.min(eigvals) + KalmanFilterConfig.cond_eps)

        def regularize(S_):
            return S_ + ridge_eps * jnp.eye(S_.shape[0], dtype=S_.dtype)

        return lax.cond(cond_number > threshold, regularize, lambda S_: S, S)

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
            S = KalmanFilter.regularize_if_ill_conditioned(S)
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
    
    @staticmethod
    @partial(jax.jit, static_argnames=["config"])
    def compute_loss_static(params, Y_target, X_pca=None, config=None):
        """
        Differentiable loss with multi-step in-sample forecast.
        Supports up to 4-step ahead prediction with decaying weight.
        """
        kf = KalmanFilter(
            A=params["A"], B=params["B"], G=params["G"], H=params["H"], config=config
        )
        kf.run_filter(Y_target, X_pca)

        X_filt = kf.X_filt
        B, A = kf.B, kf.A
        loss = 0.0

        for n in range(1, config.max_horizon + 1):
            # Predict Y_{t+n} = B @ A^n @ X_t 
            A_n = jnp.linalg.matrix_power(A, n)
            X_t = X_filt[:-n]  # (T - h, k)
            Y_pred = (B @ (A_n @ X_t.T)).T  # (T - h, n)
            Y_true = Y_target[n:]

        decay_weight = config.gamma ** (n - 1)
        loss += decay_weight * jnp.mean(jnp.square(Y_pred - Y_true))

        # L2 Regularization for stability
        loss += config.lambda_A * jnp.sum(params["A"]**2)
        loss += config.lambda_G * jnp.sum(params["G"]**2)
        loss += config.lambda_H * jnp.sum(params["H"]**2)

        return loss

    
    