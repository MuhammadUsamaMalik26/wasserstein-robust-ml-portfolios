# Does Wasserstein Robustness Rescue Machine-Learning Portfolios?

Replication code and LaTeX manuscript for a study of Wasserstein distributionally robust optimisation (WDRO), machine-learning portfolio allocation, regime uncertainty and transaction costs over 41 years of out-of-sample industry returns.

## Contents
- Python replication scripts for out-of-sample forecasts, portfolio backtests, diagnostics and statistical tests.
- `data/`: Kenneth R. French 49 Industry Portfolios and Fama–French five-factor series (source: [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html); source files retain their original metadata).
- `paper/`: manuscript LaTeX source, bibliography, research tables and figures.

## Replication
Use Python 3.11 and install dependencies:

```bash
python -m pip install -r requirements.txt
python forecasts.py signals
python forecasts.py lasso
python forecasts.py rf
python forecasts.py gb
python backtest.py bench 5
python backtest.py lasso 5
python backtest.py rf 5
python backtest.py gb 5
python analysis.py 5
python tests.py
python mechanism.py
python extras.py
python make_outputs.py
```

The accompanying archive's original instructions also recommend repeating backtests with `2` and `10` for robustness. To compile the manuscript, run `latexmk -pdf main.tex` from the `paper/` directory. Output artifacts are written to the ignored `out/` directory. Computational execution has not been independently verified in this publication copy.

## Research focus
The analysis investigates when — and whether — Wasserstein robustness improves ML-driven allocations, including the use of two-state HMM regime entropy to adapt the robustness radius. The work examines forecast deterioration, turnover, transaction-cost sensitivity, and portfolio-level outcomes.

## Author
Muhammad Usama Malik — MSc Financial Technology (Distinction), University of Exeter.
