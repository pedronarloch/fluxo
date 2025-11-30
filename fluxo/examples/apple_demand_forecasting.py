# %% 
import math
import mlflow
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from fluxo.fluxo_core.data.synthetic_generator import generate_apple_sales_data_with_promo_adjustment
import os

from fluxo.fluxo_core.plots.time_series_demand import plot_box_weekend, plot_coefficients, plot_correlation_matrix_and_save, plot_density_weekday_weekend, plot_prediction_error, plot_qq, plot_residuals, plot_scatter_demand_price, plot_time_series_demand

# %%
my_data = generate_apple_sales_data_with_promo_adjustment(
    base_demand=1000, n_rows=10_000, competitor_price_effect=-25.0
)
my_data.head(5)
# %%
os.environ["MLFLOW_LOCK_MODEL_DEPENDENCIES"] = "true"
mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("team-forecast")
# %%
X = my_data.drop(columns=["demand", "date"])
y = my_data["demand"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate error metrics
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
msle = mean_squared_log_error(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)

# %%
fig1 = plot_time_series_demand(my_data, window_size=28, style="seaborn-v0_8-dark")
fig2 = plot_box_weekend(my_data, style="seaborn-v0_8-dark")
fig3 = plot_scatter_demand_price(my_data, style="seaborn-v0_8-dark")
fig4 = plot_density_weekday_weekend(my_data, style="seaborn-v0_8-dark")

# Execute the correlation plot, saving the plot to a local temporary directory
plot_correlation_matrix_and_save(my_data, style="seaborn-v0_8-dark")

# Generate prediction-dependent plots
fig5 = plot_residuals(y_test, y_pred, style="seaborn-v0_8-dark")
fig6 = plot_coefficients(model, X_test.columns, style="seaborn-v0_8-dark")
fig7 = plot_prediction_error(y_test, y_pred, style="seaborn-v0_8-dark")
fig8 = plot_qq(y_test, y_pred, style="seaborn-v0_8-dark")

# %%
# Start an MLflow run for logging metrics, parameters, the model, and our figures
with mlflow.start_run() as run:
    # # Log the model
    mlflow.sklearn.log_model(sk_model=model, input_example=X_test, name="ridge-model", registered_model_name="my-registered-model")

    # Log the metrics
    mlflow.log_metrics(
        {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "msle": msle, "medae": medae}
    )

    # Log the hyperparameter
    mlflow.log_param("alpha", 1.0)

    # Log plots
    mlflow.log_figure(fig1, "time_series_demand.png")
    mlflow.log_figure(fig2, "box_weekend.png")
    mlflow.log_figure(fig3, "scatter_demand_price.png")
    mlflow.log_figure(fig4, "density_weekday_weekend.png")
    mlflow.log_figure(fig5, "residuals_plot.png")
    mlflow.log_figure(fig6, "coefficients_plot.png")
    mlflow.log_figure(fig7, "prediction_errors.png")
    mlflow.log_figure(fig8, "qq_plot.png")

    # Log the saved correlation matrix plot by referring to the local file system location
    mlflow.log_artifact("/tmp/corr_plot.png")
# %%
model_registered_name = "my-registered-model"
model_version = "latest"

model_uri = f"models:/{model_registered_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)
# %%
model
# %%
model.predict(X_test)
# %%
