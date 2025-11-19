# hotel-overbooking-optimization

Minimal working implementation of overbooking optimization for a hotel room inventory.
This repository contains a simple stochastic model of bookings and cancellations and a gradient-descent optimizer that finds an approximate optimal overbooking level O.

## Structure
- `model.py` — model functions (demand generation, revenue calc).
- `optimizer.py` — simple gradient-descent optimizer with finite-difference gradient.
- `main.py` — example script to run simulation and plot results.
- `requirements.txt` — required Python packages.
- `LICENSE` — MIT license.

## Quick start
1. Create virtual environment and install requirements:
```
python -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Run example:
```
python main.py
```

Outputs:
- prints optimal O and revenue estimate.
- saves `results.png` (revenue vs O) to the repository root.

## Notes
- This is a minimal educational implementation. For production-quality revenue management, consider adding robust demand forecasting, advanced optimization (SLSQP/ILP), and rigorous validation.
