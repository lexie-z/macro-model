import jax
import jax.numpy as jnp
import optax
import numpy as np
import logging
import matplotlib.pyplot as plt
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
        log_interval: int = 30
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
        self.param_history: List[Dict[str, jnp.ndarray]] = []
        self.grad_history: List[Dict[str, jnp.ndarray]] = []

        self.loss_grad_fn = jax.jit(jax.value_and_grad(KalmanTrainer._loss_wrapper),static_argnames=["config"])

    @staticmethod
    def _loss_wrapper(params, Y, X_pca, config):
        return KalmanFilter.compute_loss_static(params, Y, X_pca, config)

    def _step(self, params: Dict[str, jnp.ndarray], opt_state: optax.OptState) -> Tuple[Dict[str, jnp.ndarray], optax.OptState, float, Dict[str, jnp.ndarray]]:
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

        if self.project_spectral:
            params = params.copy()
            params["A"] = spectral_radius_projection(params["A"])

        return params, opt_state, float(loss), grads

    def train(self, num_steps: int = 100):
        """
        main training loop
        """
        params = self.params
        opt_state = self.opt_state

        for step in range(num_steps):
            try:
                params, opt_state, loss, grads = self._step(params, opt_state)
            except FloatingPointError as e:
                logger.warning(f"Step {step:03d} failed: {e}. Stopping early.")
                break

            self.loss_history.append(loss)
            self.param_history.append({k: v.copy() for k, v in params.items()})
            self.grad_history.append({k: v.copy() for k, v in grads.items()})
            
            if step % self.log_interval == 0:
                logger.info(f"[Step {step:03d}] Loss: {loss:.6f}")

        self.params = params
        self.opt_state = opt_state

        if num_steps > 0 and len(self.loss_history) > 0:
            logger.info(f"[Final Step {len(self.loss_history)-1:03d}] Loss: {self.loss_history[-1]:.6f}")

        return self.params
    
    def main_diagnostics(self):
        """
        Plot training diagnostics using stored param and grad histories.
        """

        steps = range(len(self.loss_history))

        # --- Plot 1: Loss Curve ---
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 2, 1)
        plt.plot(self.loss_history)
        plt.title("Loss Over Steps")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True)

        # --- Plot 2: Spectral Radius of A ---
        spectral_radii = [
            np.max(np.abs(np.linalg.eigvals(np.array(p["A"]))))
            for p in self.param_history
        ]
        plt.subplot(2, 2, 2)
        plt.plot(steps, spectral_radii)
        plt.axhline(0.98, color='red', linestyle='--', label="Unit circle")
        plt.title("Spectral Radius of A")
        plt.xlabel("Step")
        plt.ylabel("ρ(A)")
        plt.grid(True)
        plt.legend()

        # --- Plot 3: Gradient Norm of A ---
        grad_norms = [
            np.linalg.norm(np.array(g["A"]))
            for g in self.grad_history
        ]
        plt.subplot(2, 2, 3)
        plt.plot(steps, grad_norms)
        plt.title("Gradient Norm of A")
        plt.xlabel("Step")
        plt.ylabel("||∇A||")
        plt.grid(True)

        # --- Plot 4: Loss Delta (Log scale) ---
        loss_deltas = np.abs(np.diff(self.loss_history))
        plt.subplot(2, 2, 4)
        plt.plot(steps[:-1], np.log1p(loss_deltas))
        plt.title("Log Loss Delta")
        plt.xlabel("Step")
        plt.ylabel("log(ΔLoss + 1)")
        plt.grid(True)

        plt.tight_layout()
        plt.show()
