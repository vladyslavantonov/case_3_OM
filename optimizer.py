import numpy as np
from model import estimate_expected_revenue

def finite_difference_gradient(O, eps, *args, **kwargs):
    f_plus, _ = estimate_expected_revenue(O + eps, *args, **kwargs)
    f_minus, _ = estimate_expected_revenue(O - eps, *args, **kwargs)
    return (f_plus - f_minus) / (2 * eps)

def gradient_descent(initial_O, args_model, lr=1.0, eps_grad=1e-2, max_iters=50):
    """Simple gradient descent using finite-difference gradient estimates.
    args_model: tuple (N, price, compensation, p_cancel, lambda_base, days, runs, seed)
    Returns best_O, history list of tuples (O, est_revenue).
    """
    O = float(initial_O)
    history = []
    for k in range(max_iters):
        est_rev, _ = estimate_expected_revenue(int(round(O)), *args_model)
        history.append((int(round(O)), est_rev))
        grad = finite_difference_gradient(int(round(O)), 1.0, *args_model)
        # update (we maximize, so step in +grad direction)
        O_new = O + lr * grad
        # enforce bounds: O between 0 and N//2
        N = args_model[0]
        O_new = max(0, min(O_new, max(0, N//2)))
        if abs(O_new - O) < eps_grad:
            break
        O = O_new
    return int(round(O)), history
