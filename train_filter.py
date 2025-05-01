import jax
import jax.numpy as jnp
import optax
import logging
from typing import Dict, Optional, List, Tuple
from core_filter import KalmanFilter, KalmanFilterConfig
from model_fitting import spectral_radius_projection

logger = logging.getLogger(__name__)

class KalmanTrainer:
    def __init__(
        self,
        Y: jnp.ndarray,
        params_init: Dict[str, jnp.ndarray],
        optimizer: optax.GradientTransformation,
        X_pca: Optional[jnp.ndarray] = None,
        config: Optional[KalmanFilterConfig] = None,
        project_spectral: bool = True,
        log_interval: int = 10,
    ):
        """
        Initialize Kalman Filter Trainer for reusability over different optimizers, hyperparameter tuning etc.
        """
        self.Y = Y
        self.X_pca = X_pca
        self.params = params_init
        self.config = config or KalmanFilterConfig()
        self.optimizer = optimizer
        self.opt_state = optimizer.init(self.params)
        self.project_spectral = project_spectral
        self.log_interval = log_interval

        self.loss_history: List[float] = []
        self.loss_grad_fn = jax.jit(jax.value_and_grad(KalmanTrainer._loss_wrapper),static_argnames=["config"])

    @staticmethod
    def _loss_wrapper(params, Y, X_pca, config):
        return KalmanFilter.compute_loss_static(params, Y, X_pca, config)

    def _step(self, params: Dict[str, jnp.ndarray], opt_state: optax.OptState) -> Tuple[Dict[str, jnp.ndarray], optax.OptState, float]:
        """
        Single optimization step: computes loss, gradients, and applies parameter updates.
        args: params : dict(A,B,G,H)
        opt_state: internal state of the optimizer
        """
        loss, grads = self.loss_grad_fn(params, self.Y, self.X_pca, self.config)

        if not jnp.isfinite(loss):
            raise FloatingPointError("Non-finite loss encountered.")

        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

         # apply constraint on A: keep spectral radius within unit circle
        if self.project_spectral:
            # copy to avoid mutating params directly
            params = params.copy()
            params["A"] = spectral_radius_projection(params["A"])

        return params, opt_state, float(loss)

    def train(self, num_steps: int = 100, fallback_optimizer_fn: Optional[callable] = None):
        """
        main training loop
        """
        params = self.params
        opt_state = self.opt_state

        for step in range(num_steps):
            try:
                params, opt_state, loss = self._step(params, opt_state)
            except FloatingPointError as e:
                logger.warning(f"Step {step:03d} failed: {e}. Switching optimizer...")

                "switch to a fallback optimizer in case current optimizer becomes unstable"
                if fallback_optimizer_fn is not None:
                    self.optimizer = fallback_optimizer_fn()
                    opt_state = self.optimizer.init(params)
                    logger.info(f"Switched to fallback optimizer at step {step}")
                    continue
                else:
                    logger.warning("No fallback optimizer provided. Stopping early.")
                    break

            self.loss_history.append(loss)

            if step % self.log_interval == 0:
                logger.info(f"[Step {step:03d}] Loss: {loss:.6f}")

        self.params = params
        self.opt_state = opt_state

        if num_steps > 0 and len(self.loss_history) > 0:
            logger.info(f"[Final Step {len(self.loss_history)-1:03d}] Loss: {self.loss_history[-1]:.6f}")

        return self.params

    def get_loss_history(self):
        return self.loss_history