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
        config_callback = None,
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
        self.config_callback = config_callback
        self.optimizer = optimizer
        self.opt_state = optimizer.init(self.params)
        self.project_spectral = project_spectral
        self.log_interval = log_interval

        self.loss_history: List[float] = []
        self.param_history: List[Dict[str, jnp.ndarray]] = []
        self.grad_history: List[Dict[str, jnp.ndarray]] = []

        self.loss_grad_fn = jax.jit(jax.value_and_grad(KalmanTrainer._loss_wrapper),static_argnames=["config"])

    @staticmethod
    def _loss_wrapper(params, Y, X_pca,config):
        return KalmanFilter.compute_loss_static(params, Y, X_pca, config)

    def _step(self, params: Dict[str, jnp.ndarray], opt_state: optax.OptState) -> Tuple[Dict[str, jnp.ndarray], optax.OptState, float, Dict[str, jnp.ndarray]]:
        """
        Single optimization step: computes loss, gradients, and applies parameter updates.
        args: params : dict(A,B,G,H)
        opt_state: internal state of the optimizer
        """
        # Compute loss and gradient
        loss, grads = self.loss_grad_fn(params, self.Y, self.X_pca, self.config)

        if not jnp.isfinite(loss):
            raise FloatingPointError("Non-finite loss encountered.")

        # Update: params <- params - delta * gradient
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        # Project radius of A to lie within unit circle
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
            
            # Replace config when callback config is not None (i.e. penalty weight scheduling & curriculum training)
            if self.config_callback is not None:
                self.config = self.config_callback(step, self.config)
                
            # Early stop if loss contains NaN or Inf
            try:
                params, opt_state, loss, grads = self._step(params, opt_state)
            except FloatingPointError as e:
                logger.warning(f"Step {step:03d} failed: {e}. Stopping early.")
                break
            
            # Record loss, parameter, and gradient norm history for model diagnosis
            self.loss_history.append(loss)
            self.param_history.append({k: v.copy() for k, v in params.items()})
            self.grad_history.append({k: v.copy() for k, v in grads.items()})
            
            # Log loss every once in a while
            if step % self.log_interval == 0:
                logger.info(f"[Step {step:03d}] Loss: {loss:.6f}")

        self.params = params
        self.opt_state = opt_state

        if num_steps > 0 and len(self.loss_history) > 0:
            logger.info(f"[Final Step {len(self.loss_history)-1:03d}] Loss: {self.loss_history[-1]:.6f}")

        return self.params
    
    def main_diagnostics(self):
        """
        Plot training diagnostics.
        """

        steps = range(len(self.loss_history))

        fig, axs = plt.subplots(2, 2, figsize=(12,6))

        # --- Plot 1: Loss Curve ---
        axs[0, 0].plot(self.loss_history)
        axs[0, 0].set_title("Loss Over Steps")
        axs[0, 0].set_xlabel("Step")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].grid(True)

        # --- Plot 2: Spectral Radius of A ---
        spectral_radii = [
            np.max(np.abs(np.linalg.eigvals(np.array(p["A"]))))
            for p in self.param_history
        ]
        axs[0, 1].plot(steps, spectral_radii)
        axs[0, 1].axhline(0.98, color='red', linestyle='--', label="Unit circle")
        axs[0, 1].set_title("Spectral Radius of A")
        axs[0, 1].set_xlabel("Step")
        axs[0, 1].set_ylabel("ρ(A)")
        axs[0, 1].legend()
        axs[0, 1].grid(True)

        # --- Plot 3: Gradient Norms ---
        grad_norm_A = [np.linalg.norm(np.array(g["A"])) for g in self.grad_history]
        grad_norm_B = [np.linalg.norm(np.array(g["B"])) for g in self.grad_history]
        grad_norm_G = [np.linalg.norm(np.array(g["G"])) for g in self.grad_history]
        grad_norm_H = [np.linalg.norm(np.array(g["H"])) for g in self.grad_history]

        axs[1, 0].plot(steps, grad_norm_A, label="||∇A||")
        axs[1, 0].plot(steps, grad_norm_B, label="||∇B||")
        axs[1, 0].plot(steps, grad_norm_G, label="||∇G||")
        axs[1, 0].plot(steps, grad_norm_H, label="||∇H||")
        axs[1, 0].set_title("Gradient Norms")
        axs[1, 0].set_xlabel("Step")
        axs[1, 0].set_ylabel("Gradient Norm")
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        # --- Plot 4: Log Loss Delta ---
        loss_deltas = np.abs(np.diff(self.loss_history))
        axs[1, 1].plot(steps[:-1], np.log1p(loss_deltas))
        axs[1, 1].set_title("Log Loss Delta")
        axs[1, 1].set_xlabel("Step")
        axs[1, 1].set_ylabel("log(ΔLoss + 1)")
        axs[1, 1].grid(True)

        plt.tight_layout()
        plt.show()
