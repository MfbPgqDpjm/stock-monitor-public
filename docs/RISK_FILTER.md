# Risk Filter

This public document describes the boundary of the risk-filter module without
publishing private scoring parameters.

The module is designed to produce a UI-facing risk assessment from price,
valuation, and trend context. Private deployments provide their own thresholds,
lookback windows, score weights, and action levels through private
configuration.

The public copy does not include:

- Numeric score thresholds
- Numeric valuation thresholds
- Numeric lookback windows
- Formula details
- Example ticker data

Runtime data and valuation inputs must remain private.
