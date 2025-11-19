import numpy as np
import matplotlib.pyplot as plt
from model import estimate_expected_revenue
from optimizer import gradient_descent

def main():
    # Model parameters (example)
    N = 200                # total rooms
    price = 140.0          # average price per night $
    compensation = 250.0   # cost per rejected guest $
    p_cancel = 0.1         # cancellation probability
    lambda_base = 120.0    # mean daily bookings (Poisson)
    days = 30
    runs = 300

    # Estimate revenue for a range of O
    Os = list(range(0, 41))
    revs = []
    for O in Os:
        est, std = estimate_expected_revenue(O, N, price, compensation, p_cancel, lambda_base, days, runs)
        revs.append(est)

    # Plot revenue vs O
    plt.figure(figsize=(8,4))
    plt.plot(Os, revs, marker='o')
    plt.xlabel('Overbooking level O (bookings)')
    plt.ylabel('Estimated revenue ($) over {} days'.format(days))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results.png', dpi=150)
    print('Saved results.png')

    # Run optimizer (gradient descent)
    args_model = (N, price, compensation, p_cancel, lambda_base, days, runs, None)
    best_O, history = gradient_descent(initial_O=10, args_model=args_model, lr=0.05, eps_grad=0.5, max_iters=10)
    print('Best O found (approx):', best_O)
    est_best, _ = estimate_expected_revenue(best_O, N, price, compensation, p_cancel, lambda_base, days, runs)
    print('Estimated revenue at best O: ${:,.2f}'.format(est_best))

if __name__ == '__main__':
    main()
