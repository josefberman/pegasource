# Path estimation

Reconstruct continuous trajectories from mixed observations (GPS, cell sectors, etc.) using graph map-matching, probabilistic filters, and optional neural models. Install full dependencies with `pip install -e ".[path_estimation]"`.

The package root re-exports `run_evaluation` lazily; submodule imports such as `pegasource.path_estimation.metrics` work without **torch** / **filterpy**.

## Reference

::: pegasource.path_estimation.evaluate
