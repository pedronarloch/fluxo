# Fluxo

A Python toolkit for demand forecasting with MLflow integration and visualization utilities.

## Features

- **Synthetic Data Generation**: Generate realistic sales data with multiple influencing factors (seasonality, pricing, promotions, competition)
- **MLflow Integration**: Automatic experiment tracking, model logging, and artifact management
- **Visualization Suite**: Pre-built plots for exploratory data analysis and model diagnostics
- **Production-Ready**: Built with scikit-learn, LightGBM, and statsmodels for robust forecasting

## Installation

```bash
uv sync
```

## Quick Start

```python
from fluxo.fluxo_core.data.synthetic_generator import generate_apple_sales_data_with_promo_adjustment
from fluxo.fluxo_core.plots.time_series_demand import plot_time_series_demand

# Generate synthetic demand data
data = generate_apple_sales_data_with_promo_adjustment(
    base_demand=1000,
    n_rows=10_000
)

# Visualize demand patterns
fig = plot_time_series_demand(data, window_size=28)
```

See `fluxo/examples/apple_demand_forecasting.py` for a complete forecasting workflow with MLflow tracking.

## MLflow Setup

For a production-ready MLflow server with MinIO artifact storage, check out the compatible Docker stack:

[mlflow-stack](https://github.com/pedronarloch/mlflow-stack) - Containerized MLflow server with MinIO backend

## Requirements

- Python >= 3.13
- MLflow, LightGBM, scikit-learn, statsmodels, matplotlib, seaborn

## Development

```bash
uv sync --group dev
```
