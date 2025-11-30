# %%
import io
import math
import os
from datetime import datetime

import mlflow
import pandas as pd
from dotenv import load_dotenv
from minio import Minio
from sklearn.linear_model import Lasso, Ridge

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from fluxo.fluxo_core.data.synthetic_generator import (
    generate_apple_sales_data_with_promo_adjustment,
)
from fluxo.fluxo_core.plots.time_series_demand import (
    plot_box_weekend,
    plot_coefficients,
    plot_correlation_matrix_and_save,
    plot_density_weekday_weekend,
    plot_prediction_error,
    plot_qq,
    plot_residuals,
    plot_scatter_demand_price,
    plot_time_series_demand,
)

# %%
# Load environment variables
load_dotenv(override=True)

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_DATA_BUCKET = os.getenv("MINIO_DATA_BUCKET")
MINIO_MLFLOW_BUCKET = os.getenv("MINIO_MLFLOW_BUCKET")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

required_vars = {
    "MINIO_ENDPOINT": MINIO_ENDPOINT,
    "MINIO_ACCESS_KEY": MINIO_ACCESS_KEY,
    "MINIO_SECRET_KEY": MINIO_SECRET_KEY,
    "MINIO_DATA_BUCKET": MINIO_DATA_BUCKET,
    "MINIO_MLFLOW_BUCKET": MINIO_MLFLOW_BUCKET,
    "MLFLOW_TRACKING_URI": MLFLOW_TRACKING_URI,

}

missing_vars = [var for var, value in required_vars.items() if not value]
if missing_vars:
    raise ValueError(
        f"Missing required environment variables: {', '.join(missing_vars)}\n"
        "Please create a .env file based on .env.example"
    )

# %%
# Initialize MinIO client
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )

    if not minio_client.bucket_exists(MINIO_MLFLOW_BUCKET):
        minio_client.make_bucket(MINIO_MLFLOW_BUCKET)
        print(f"Created bucket: {MINIO_MLFLOW_BUCKET}")
    else:
        print(f"Using existing bucket: {MINIO_MLFLOW_BUCKET}")

    # Create bucket if it doesn't exist
    if not minio_client.bucket_exists(MINIO_DATA_BUCKET):
        minio_client.make_bucket(MINIO_DATA_BUCKET)
        print(f"Created bucket: {MINIO_DATA_BUCKET}")
    else:
        print(f"Using existing bucket: {MINIO_DATA_BUCKET}")
except Exception as e:
    print(f"Error connecting to MinIO: {e}")
    print("Make sure MinIO server is running at {MINIO_ENDPOINT}")
    raise

# %%
# Generate and save synthetic data to MinIO as Parquet
my_data = generate_apple_sales_data_with_promo_adjustment(
    base_demand=1000, n_rows=10_000, competitor_price_effect=-25.0
)

data_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
data_filename = f"apple_sales_data_{data_timestamp}.parquet"

parquet_buffer = io.BytesIO()
my_data.to_parquet(parquet_buffer, index=False)
parquet_buffer.seek(0)

try:
    minio_client.put_object(
        MINIO_DATA_BUCKET,
        f"dataset/{data_filename}",
        parquet_buffer,
        length=parquet_buffer.getbuffer().nbytes,
        content_type="application/octet-stream",
    )
    print(f"Data saved to MinIO: {MINIO_DATA_BUCKET}/{data_filename}")
except Exception as e:
    print(f"Error uploading data to MinIO: {e}")
    raise

# %%
# Fetch data from MinIO for training
try:
    response = minio_client.get_object(MINIO_DATA_BUCKET, f"dataset/{data_filename}",)
    training_data = pd.read_parquet(io.BytesIO(response.read()))
    response.close()
    response.release_conn()

    print(f"Data fetched from MinIO: {training_data.shape}")
    training_data.head(5)
except Exception as e:
    print(f"Error fetching data from MinIO: {e}")
    raise

# %%
# Configure MLflow
try:
    os.environ["MLFLOW_LOCK_MODEL_DEPENDENCIES"] = "true"
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("apple-demand-forecast")
    print(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
except Exception as e:
    print(f"Error configuring MLflow: {e}")
    print("Make sure MLflow server is running at {MLFLOW_TRACKING_URI}")
    raise

# %%
# Prepare training data
X = training_data.drop(columns=["demand", "date"])
y = training_data["demand"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = Ridge()
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
# Generate plots
fig1 = plot_time_series_demand(training_data, window_size=28, style="seaborn-v0_8-dark")
fig2 = plot_box_weekend(training_data, style="seaborn-v0_8-dark")
fig3 = plot_scatter_demand_price(training_data, style="seaborn-v0_8-dark")
fig4 = plot_density_weekday_weekend(training_data, style="seaborn-v0_8-dark")

# Execute the correlation plot, saving the plot to a local temporary directory
plot_correlation_matrix_and_save(training_data, style="seaborn-v0_8-dark")

# Generate prediction-dependent plots
fig5 = plot_residuals(y_test, y_pred, style="seaborn-v0_8-dark")
fig6 = plot_coefficients(model, X_test.columns, style="seaborn-v0_8-dark")
fig7 = plot_prediction_error(y_test, y_pred, style="seaborn-v0_8-dark")
fig8 = plot_qq(y_test, y_pred, style="seaborn-v0_8-dark")

# %%
# Start an MLflow run for logging metrics, parameters, the model, and our figures
with mlflow.start_run() as run:
    # Log the model
    mlflow.sklearn.log_model(
        sk_model=model,
        input_example=X_test,
        artifact_path="apple-demand-model",
        registered_model_name="apple-demand-forecast",
    )

    # Log the metrics
    mlflow.log_metrics(
        {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "msle": msle, "medae": medae}
    )

    # Log model hyperparameters and training config
    mlflow.log_params(
        {
            "test_size": 0.2,
            "data_source": f"minio://{MINIO_MLFLOW_BUCKET}/{data_filename}",
            "data_timestamp": data_timestamp,
            "n_training_samples": len(X_train),
            "n_test_samples": len(X_test),
        }
    )

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

    print(f"MLflow run completed. Run ID: {run.info.run_id}")

# %%
# Load model from MLflow registry for inference
model_name = "apple-demand-forecast"
model_version = "latest"

model_uri = f"models:/{model_name}/{model_version}"
loaded_model = mlflow.sklearn.load_model(model_uri)

# %%
# Check the loaded model
loaded_model

# %%
# Make predictions with loaded model
predictions = loaded_model.predict(X_test)
predictions_df = pd.DataFrame({"predictions": predictions})

# %%
df_bytes  = predictions_df.to_csv().encode()
df_buffer = io.BytesIO(df_bytes)
# %%
pred_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pred_filename = f"predictions_apple_sales_data_{pred_timestamp}.csv"
minio_client.put_object(
    MINIO_DATA_BUCKET,
    f"predictions/{pred_filename}",
    data=df_buffer,
    length=len(df_bytes),
    content_type="application/csv"
)
# %%
