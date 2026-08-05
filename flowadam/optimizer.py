"""
FlowAdam: Adam with clipped gradient-flow integration and a scalar-preconditioner dial.

FlowAdam augments Adam with short, clipped ODE gradient-flow steps when the running gradient
statistics indicate a plateau or stiff region, blending the flow velocity into Adam's momentum
(soft injection) rather than replacing it. The precond_power dial softens Adam's per-coordinate
second moment toward a shared gauge-invariant scalar, so the injected gradient-flow direction
survives subsequent Adam steps.
"""

import math
import warnings

import torch
from torch.optim import Optimizer
from torchdiffeq import odeint


def sync():
    """
    Synchronize CUDA operations for accurate timing.
    No-op if CUDA is not available.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class FlowAdam(Optimizer):
    """
    FlowAdam: adaptive hybrid optimizer with soft momentum injection.

    Combines Adam with short, clipped ODE gradient-flow steps for navigating difficult
    loss-landscape regions (plateaus and stiff curvature).

    Args:
        params: model parameters in a single parameter group
        lr: learning rate (default: 1e-3)
        betas: Adam beta coefficients (default: (0.9, 0.999))
        eps: Adam epsilon (default: 1e-8)
        mode: preset 'A' or 'B' (default: 'B') for switch_sensitivity,
            curvature_sensitivity, and ode_t_scale; explicit values override the preset.
        ode_t_scale: ODE integration time scale (default: 1.0)
        ode_method: ODE solver method (default: 'dopri5')
        ode_tol: ODE solver tolerance (default: 1e-4)
        switch_sensitivity: plateau-detection threshold (default: 0.5)
        curvature_sensitivity: stiffness-detection threshold (default: 2.0)
        momentum_blend_gamma: blend factor for ODE velocity injection (default: 0.5),
            new_momentum = (1 - gamma) * old_momentum + gamma * ode_velocity.
        clip_mode: 'percoord' (elementwise clip to [-1, 1]) or 'globalnorm'
            (scale the gradient to norm <= clip_norm_c, preserving direction).
        clip_norm_c: norm bound used when clip_mode='globalnorm' (default: 1.0).
        precond_power: dial p for the Adam denominator, s_i^p * sbar^(1 - p);
            p=1 is standard Adam, p=0 is a shared-scalar (direction-preserving) denom.
        precond_scalar: shared-scalar convention for p<1, 'geomean' or 'rms'
            ('rms' is exactly gauge-invariant at p=0).

    Requires a closure that computes the loss and calls backward().
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        mode="B",
        ode_t_scale=None,
        ode_method='dopri5',
        ode_tol=1e-4,
        switch_sensitivity=None,
        curvature_sensitivity=None,
        momentum_blend_gamma=0.5,
        clip_mode="percoord",
        clip_norm_c=1.0,
        precond_power=1.0,
        precond_scalar="geomean",
    ):
        presets = {
            "A": {
                "switch_sensitivity": 0.40,
                "curvature_sensitivity": 3.0,
                "ode_t_scale": 2.0,
            },
            "B": {
                "switch_sensitivity": 0.5,
                "curvature_sensitivity": 2.0,
                "ode_t_scale": 1.0,
            },
        }
        if mode not in presets:
            raise ValueError("mode must be 'A' or 'B'.")

        if switch_sensitivity is None:
            switch_sensitivity = presets[mode]["switch_sensitivity"]
        if curvature_sensitivity is None:
            curvature_sensitivity = presets[mode]["curvature_sensitivity"]
        if ode_t_scale is None:
            ode_t_scale = presets[mode]["ode_t_scale"]

        defaults = dict(lr=lr, betas=betas, eps=eps,
                        mode=mode,
                        ode_t_scale=ode_t_scale,
                        ode_method=ode_method,
                        ode_tol=ode_tol,
                        switch_sensitivity=switch_sensitivity,
                        curvature_sensitivity=curvature_sensitivity,
                        momentum_blend_gamma=momentum_blend_gamma,
                        clip_mode=clip_mode,
                        clip_norm_c=clip_norm_c,
                        precond_power=precond_power,
                        precond_scalar=precond_scalar)
        if clip_mode not in ("percoord", "globalnorm"):
            raise ValueError("clip_mode must be 'percoord' or 'globalnorm'.")
        if precond_scalar not in ("geomean", "rms"):
            raise ValueError("precond_scalar must be 'geomean' or 'rms'.")
        super().__init__(params, defaults)
        if len(self.param_groups) != 1:
            raise ValueError(
                "FlowAdam currently supports exactly one parameter group. "
                "Pass a single iterable of parameters rather than per-group options."
            )

        self.state['global'] = {
            'avg_grad_norm': None,
            'avg_curvature': None,
            'step_count': 0,
            'history_ode': [],
            'grad_evals_total': 0,
            'ode_nfe_total': 0,
            'ode_nfe_per_trigger': [],
            '_current_ode_nfe': 0,
        }

    def _flatten(self, tensor_list):
        """Flatten list of tensors into single vector."""
        views = []
        for p in tensor_list:
            views.append(p.view(-1))
        return torch.cat(views) if views else torch.tensor([])

    def _unflatten_and_update(self, flat_params, target_params):
        """Update parameters from flattened vector."""
        offset = 0
        for p in target_params:
            numel = p.numel()
            p.data.copy_(flat_params[offset:offset+numel].view_as(p))
            offset += numel

    def get_ode_count(self):
        """Return number of ODE triggers so far."""
        return len(self.state['global']['history_ode'])

    def get_total_grad_evals(self):
        """
        Return total backward calls including ODE.
        grad_evals_total_including_ode = grad_evals_total + ode_nfe_total
        (Each ode_func call triggers closure() which calls backward())
        """
        stats = self.state['global']
        return stats['grad_evals_total'] + stats['ode_nfe_total']

    def get_total_ode_nfe(self):
        """Return total ODE function evaluations."""
        return self.state['global']['ode_nfe_total']

    def get_ode_nfe_stats(self):
        """
        Return statistics over ode_nfe_per_trigger.
        Returns dict with mean/median/min/max, or None values if empty.
        """
        nfe_list = self.state['global']['ode_nfe_per_trigger']
        if not nfe_list:
            return {
                'mean': None,
                'median': None,
                'min': None,
                'max': None,
                'count': 0
            }

        import statistics
        return {
            'mean': statistics.mean(nfe_list),
            'median': statistics.median(nfe_list),
            'min': min(nfe_list),
            'max': max(nfe_list),
            'count': len(nfe_list)
        }

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Args:
            closure: A callable that computes loss and calls backward().
                     Required for this optimizer.
        """
        if closure is None:
            raise RuntimeError("FlowAdam requires a closure that computes loss.")

        stats = self.state['global']

        loss = closure()
        stats['grad_evals_total'] += 1

        all_params = []
        all_grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    all_params.append(p)
                    all_grads.append(p.grad)

        if not all_params:
            return loss

        current_grad_vec = self._flatten(all_grads)
        current_norm = torch.norm(current_grad_vec).item()

        prev_grad_vec = stats.get('prev_grad_vec', torch.zeros_like(current_grad_vec))
        current_curvature = torch.norm(current_grad_vec - prev_grad_vec).item()

        beta_stat = 0.9

        if stats['avg_grad_norm'] is None:
            stats['avg_grad_norm'] = current_norm
            stats['avg_curvature'] = current_curvature
        else:
            stats['avg_grad_norm'] = beta_stat * stats['avg_grad_norm'] + (1-beta_stat) * current_norm
            stats['avg_curvature'] = beta_stat * stats['avg_curvature'] + (1-beta_stat) * current_curvature

        group = self.param_groups[0]
        is_plateau = current_norm < (stats['avg_grad_norm'] * group['switch_sensitivity'])
        is_stiff = current_curvature > (stats['avg_curvature'] * group['curvature_sensitivity'])

        use_ode = (is_plateau or is_stiff) and (stats['step_count'] > 10)

        if use_ode:
            stats['history_ode'].append(stats['step_count'])

            y0 = self._flatten(all_params)
            old_params_flat = y0.clone()

            nfe_before = stats['_current_ode_nfe']
            stats['_current_ode_nfe'] = 0

            clip_mode = group['clip_mode']

            def ode_func(t, y_flat):
                """Gradient flow ODE: dtheta/dt = -clip(nablaL(theta))

                clip_mode='percoord' (default): elementwise clip to [-1, 1] - bounds each
                    coordinate but distorts the gradient direction when coordinates differ in
                    scale (sign-like), which throttles the gradient-flow low-rank bias.
                clip_mode='globalnorm': scale the whole gradient vector to norm <= 1 - bounds
                    magnitude while preserving direction exactly, retaining the flow geometry.
                """
                stats['_current_ode_nfe'] += 1

                self._unflatten_and_update(y_flat, all_params)
                with torch.enable_grad():
                    self.zero_grad()
                    closure()
                new_grads = [p.grad.view(-1) for p in all_params]
                g = torch.cat(new_grads)
                if clip_mode == 'globalnorm':
                    c = group['clip_norm_c']
                    gn = g.norm()
                    if gn > c:
                        g = g * (c / (gn + 1e-12))
                    return -g
                return -torch.clamp(g, -1.0, 1.0)

            t_span = torch.tensor([0.0, group['lr'] * group['ode_t_scale']]).to(y0.device)

            try:
                solution = odeint(ode_func, y0, t_span,
                                  method=group['ode_method'],
                                  rtol=group['ode_tol'],
                                  atol=group['ode_tol'])

                y_final = solution[-1]
                self._unflatten_and_update(y_final, all_params)

                nfe_delta = stats['_current_ode_nfe']
                stats['ode_nfe_per_trigger'].append(nfe_delta)
                stats['ode_nfe_total'] += nfe_delta

                displacement = y_final - old_params_flat
                gamma = group['momentum_blend_gamma']
                offset = 0
                for p in all_params:
                    numel = p.numel()
                    p_disp = displacement[offset:offset+numel].view_as(p)

                    state = self.state[p]
                    if 'exp_avg' in state:
                        ode_velocity = -p_disp / group['lr']

                        current_mom_norm = state['exp_avg'].norm()
                        ode_vel_norm = ode_velocity.norm()
                        scaling_factor = 1.0
                        if ode_vel_norm > current_mom_norm * 5.0:
                            scaling_factor = (current_mom_norm * 5.0) / (ode_vel_norm + 1e-8)

                        state['exp_avg'].mul_(1.0 - gamma).add_(ode_velocity * scaling_factor, alpha=gamma)

                    offset += numel

            except (RuntimeError, FloatingPointError) as exc:
                nfe_delta = stats['_current_ode_nfe']
                if nfe_delta > 0:
                    stats['ode_nfe_per_trigger'].append(nfe_delta)
                    stats['ode_nfe_total'] += nfe_delta

                warnings.warn(
                    "FlowAdam ODE integration failed at optimizer step "
                    f"{stats['step_count']} ({type(exc).__name__}: {exc}). "
                    "Restoring the pre-ODE parameters and applying the Adam fallback for this step.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._unflatten_and_update(y0, all_params)
                # The last ODE evaluation left gradients from a trial state; recompute
                # them at the restored parameters before the fallback Adam step.
                with torch.enable_grad():
                    self.zero_grad()
                    closure()
                stats['grad_evals_total'] += 1
                self._adam_step(all_params, group)
        else:
            self._adam_step(all_params, group)

        stats['prev_grad_vec'] = current_grad_vec.detach()
        stats['step_count'] += 1
        return loss

    def _adam_step(self, params, group):
        """Adam update with the precond_power dial.

        precond_power (p) softens Adam's per-coordinate preconditioner toward a shared
        scalar (direction-preserving) one:
            denom_i = (s_i)^p * (sbar)^(1 - p),   s_i = sqrt(v_hat_i)
        p=1 is standard per-coordinate Adam; p=0 uses a shared scalar denominator, which
        preserves the injected gradient-flow direction so the low-rank bias survives.
        sbar follows precond_scalar ('geomean' or 'rms'; 'rms' is gauge-invariant).

        Epsilon convention: the p=1 branch folds bias correction into the step size and
        adds eps to sqrt(v) before that factor (TensorFlow-style Adam); it differs from
        torch.optim.Adam only through an effective epsilon of eps/sqrt(bias_corr2) when
        gradient variance is near zero.
        """
        beta1, beta2 = group['betas']
        p_pow = group['precond_power']

        if p_pow >= 1.0:
            for p in params:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['step'] += 1
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_corr1 = 1 - beta1 ** state['step']
                bias_corr2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_corr2) / bias_corr1

                denom = exp_avg_sq.sqrt().add_(group['eps'])
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
            return

        eps = group['eps']
        scalar_stat = group.get('precond_scalar', 'geomean')
        active = [p for p in params if p.grad is not None]
        for p in active:
            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
            state['step'] += 1
            state['exp_avg'].mul_(beta1).add_(p.grad.data, alpha=1 - beta1)
            state['exp_avg_sq'].mul_(beta2).addcmul_(p.grad.data, p.grad.data, value=1 - beta2)

        acc = torch.zeros((), device=active[0].data.device) if active else 0.0
        count = 0
        s_cache = {}
        for p in active:
            state = self.state[p]
            bias_corr2 = 1 - beta2 ** state['step']
            vh = state['exp_avg_sq'] / bias_corr2
            s = vh.sqrt()
            s_cache[p] = s
            acc = acc + (vh.sum() if scalar_stat == 'rms' else torch.log(s + eps).sum())
            count += s.numel()
        shared = (acc / max(count, 1)).sqrt() if scalar_stat == 'rms' \
            else torch.exp(acc / max(count, 1))

        for p in active:
            state = self.state[p]
            bias_corr1 = 1 - beta1 ** state['step']
            mhat = state['exp_avg'] / bias_corr1
            denom = (s_cache[p] + eps).pow(p_pow) * (shared + eps).pow(1.0 - p_pow)
            p.data.addcdiv_(mhat, denom + eps, value=-group['lr'])


AdaptiveHybridOptimizer = FlowAdam
