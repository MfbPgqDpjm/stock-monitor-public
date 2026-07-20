# Stock Monitor Project Notes

This is a public-safe overview of the Stock Monitor codebase.

The repository contains a Streamlit dashboard, scheduled scan entrypoints,
notification plumbing, runtime-state helpers, market-data helpers, and modules
for several monitoring views.

The public copy intentionally omits private operating parameters, including:

- Numeric strategy windows
- Numeric thresholds
- Ranking formulas
- Default ticker universes
- Position-sizing rules
- Notification credentials
- Runtime state and cached market data

Runtime configuration should be supplied outside the repository. The checked-in
defaults are placeholders only and are not intended to represent a usable
trading system.

## Runtime Data

Runtime JSON files, logs, holdings, execution records, cache files, and secrets
must stay outside Git. Use environment-specific data directories for private
deployments.

## Modules

| Module | Public description |
| --- | --- |
| `app.py` | Streamlit dashboard and manual controls |
| `main.py` | Scheduler entrypoint |
| `scheduler.py` | Scan orchestration |
| `strategy.py` | Market-signal helpers |
| `momentum_scorer.py` | Momentum workflow skeleton |
| `risk_scoring.py` | Risk-display helper skeleton |
| `dip_zone.py` | Dip-zone display helper skeleton |
| `state_manager.py` | Config and runtime-state IO helpers |
| `notifier.py` | Notification formatting and delivery plumbing |

Private parameters and formulas are intentionally not documented here.
