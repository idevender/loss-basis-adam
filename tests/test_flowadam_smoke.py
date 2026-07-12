import pytest


torch = pytest.importorskip("torch")
pytest.importorskip("torchdiffeq")

from flowadam import FlowAdam
import flowadam.optimizer as optimizer_module


def _quadratic_closure(optimizer, parameter):
    def closure():
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        return loss

    return closure


def test_rms_scalar_endpoint_smoke_step():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = FlowAdam(
        [parameter],
        lr=1e-2,
        precond_power=0.0,
        precond_scalar="rms",
    )

    loss = optimizer.step(_quadratic_closure(optimizer, parameter))

    assert torch.isfinite(loss)
    assert torch.isfinite(parameter).all()
    assert parameter.item() < 1.0


def test_multiple_parameter_groups_are_rejected():
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([1.0]))

    with pytest.raises(ValueError, match="exactly one parameter group"):
        FlowAdam([{"params": [first]}, {"params": [second]}])


def test_ode_runtime_error_warns_and_falls_back_to_adam(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = FlowAdam([parameter], lr=1e-2)
    stats = optimizer.state["global"]
    stats["step_count"] = 11
    stats["avg_grad_norm"] = 10.0
    stats["avg_curvature"] = 10.0

    def fail_odeint(*args, **kwargs):
        raise RuntimeError("forced solver failure")

    monkeypatch.setattr(optimizer_module, "odeint", fail_odeint)

    with pytest.warns(RuntimeWarning, match="ODE integration failed"):
        optimizer.step(_quadratic_closure(optimizer, parameter))

    assert torch.isfinite(parameter).all()
    assert parameter.item() < 1.0


def test_ode_failure_fallback_recomputes_gradient_at_restored_state(monkeypatch):
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = FlowAdam([parameter], lr=1e-2)
    stats = optimizer.state["global"]
    stats["step_count"] = 11
    stats["avg_grad_norm"] = 10.0
    stats["avg_curvature"] = 10.0

    def evaluate_then_fail(func, y0, t_span, **kwargs):
        # Evaluate once at a trial state on the far side of the optimum, leaving a
        # stale gradient of the opposite sign, then fail.
        func(t_span[0], torch.tensor([-2.0]))
        raise RuntimeError("forced solver failure after one evaluation")

    monkeypatch.setattr(optimizer_module, "odeint", evaluate_then_fail)

    with pytest.warns(RuntimeWarning, match="ODE integration failed"):
        optimizer.step(_quadratic_closure(optimizer, parameter))

    # The fallback Adam step must use the gradient at the restored parameters
    # (positive at 1.0), moving toward the optimum rather than away from it.
    assert parameter.item() < 1.0
