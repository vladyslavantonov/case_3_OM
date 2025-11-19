import numpy as np

def generate_demand(lambda_base, days=1, seed=None):
    """Generate daily demand (Poisson) for given lambda_base."
    rng = np.random.default_rng(seed)
    return rng.poisson(lam=lambda_base, size=days)

def simulate_one_run(O, N, price, compensation, p_cancel, demand_series):
    """Simulate one series of days and compute total revenue."
    # arrivals accepted = min(d + O, N)
    arrivals = np.minimum(demand_series + O, N)
    # expected cancellations reduce arrivals: simulate cancellations per booking
    # for simplicity apply cancellation probabilistically to accepted bookings
    rng = np.random.default_rng()
    cancellations = rng.binomial(arrivals.astype(int), p_cancel)
    actual_occupancy = arrivals - cancellations
    # rejections: bookings that exceed capacity before cancellations
    rejections = np.maximum(0, demand_series + O - N)
    revenue = price * actual_occupancy.sum() - compensation * rejections.sum()
    return revenue, actual_occupancy.sum(), rejections.sum()

def estimate_expected_revenue(O, N, price, compensation, p_cancel, lambda_base, days=30, runs=200, seed=None):
    """Estimate expected revenue by Monte-Carlo simulation over 'runs' trials."""
    rng = np.random.default_rng(seed)
    revenues = []
    for r in range(runs):
        # generate demand series for the period
        demand_series = rng.poisson(lam=lambda_base, size=days)
        rev, occ, rej = simulate_one_run(O, N, price, compensation, p_cancel, demand_series)
        revenues.append(rev)
    return np.mean(revenues), np.std(revenues)
