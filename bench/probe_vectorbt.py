"""VectorBT parameter-grid trap. Plan §3.1.

Asking for a 5x5 indicator grid returns 5 columns, not 25: `param_product`
applies WITHIN one indicator's parameters, not ACROSS two separately-run
indicators, so you get the diagonal. There is no warning — the results table
looks right and every row is correctly labelled.

This is why VectorBT is the exploration and analysis layer here rather than the
sweep engine, alongside the measured 89.7ms/combination against our engine's
2.2ms.

    python3.12 -m venv .vbt && .vbt/bin/pip install vectorbt "plotly<6"
    .vbt/bin/python bench/probe_vectorbt.py

(`plotly<6` is required: current VectorBT fails to import against plotly 6,
which removed `scattermapbox`.)
"""
import numpy as np, pandas as pd, vectorbt as vbt
N=5000
close = pd.Series(1900+np.cumsum(np.random.default_rng(1).normal(0,0.9,N)),
                  index=pd.date_range("2021-01-04", periods=N, freq="15min"))
fasts=[10,15,20,25,30]; slows=[40,50,60,70,80]
f = vbt.MA.run(close, fasts, short_name="fast", param_product=True)
s = vbt.MA.run(close, slows, short_name="slow", param_product=True)
pf = vbt.Portfolio.from_signals(close, f.ma_crossed_above(s), f.ma_crossed_below(s),
                                init_cash=10_000, freq="15min")
print(f"asked for  : {len(fasts)} x {len(slows)} = {len(fasts)*len(slows)} combinations")
print(f"got        : {pf.wrapper.shape_2d[1]} columns")
print(f"pairs      : {list(pf.total_return().index)}")
print("\n-> param_product applies WITHIN one indicator's params, not ACROSS two")
print("   separately-run indicators. No warning. The results table just has")
print("   fewer rows than you think, each correctly labelled.")
