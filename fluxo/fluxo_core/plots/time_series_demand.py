import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats



def plot_time_series_demand(data, window_size=7, style="seaborn", plot_size=(16, 12)):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    df = data.copy()

    df["date"] = pd.to_datetime(df["date"])

    # Calculate the rolling average
    df["rolling_avg"] = df["demand"].rolling(window=window_size).mean()

    with plt.style.context(style=style):
        fig, ax = plt.subplots(figsize=plot_size)
        # Plot the original time series data with low alpha (transparency)
        ax.plot(df["date"], df["demand"], "b-o", label="Original Demand", alpha=0.15)
        # Plot the rolling average
        ax.plot(
            df["date"],
            df["rolling_avg"],
            "r",
            label=f"{window_size}-Day Rolling Average",
        )

        # Set labels and title
        ax.set_title(
            f"Time Series Plot of Demand with {window_size} day Rolling Average",
            fontsize=14,
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Demand", fontsize=12)

        # Add legend to explain the lines
        ax.legend()
        plt.tight_layout()

    plt.close(fig)
    return fig

def plot_box_weekend(df, style="seaborn", plot_size=(10, 8)):
    with plt.style.context(style=style):
        fig, ax = plt.subplots(figsize=plot_size)
        sns.boxplot(data=df, x="weekend", y="demand", ax=ax, color="lightgray")
        sns.stripplot(
            data=df,
            x="weekend",
            y="demand",
            ax=ax,
            hue="weekend",
            palette={0: "blue", 1: "green"},
            alpha=0.15,
            jitter=0.3,
            size=5,
        )

        ax.set_title("Box Plot of Demand on Weekends vs. Weekdays", fontsize=14)
        ax.set_xlabel("Weekend (0: No, 1: Yes)", fontsize=12)
        ax.set_ylabel("Demand", fontsize=12)
        for i in ax.get_xticklabels() + ax.get_yticklabels():
            i.set_fontsize(10)
        ax.legend_.remove()
        plt.tight_layout()
    plt.close(fig)
    return fig


def plot_scatter_demand_price(df, style="seaborn", plot_size=(10, 8)):
    with plt.style.context(style=style):
        fig, ax = plt.subplots(figsize=plot_size)
        # Scatter plot with jitter, transparency, and color-coded based on weekend
        sns.scatterplot(
            data=df,
            x="price_per_kg",
            y="demand",
            hue="weekend",
            palette={0: "blue", 1: "green"},
            alpha=0.15,
            ax=ax,
        )
        # Fit a simple regression line for each subgroup
        sns.regplot(
            data=df[df["weekend"] == 0],
            x="price_per_kg",
            y="demand",
            scatter=False,
            color="blue",
            ax=ax,
        )
        sns.regplot(
            data=df[df["weekend"] == 1],
            x="price_per_kg",
            y="demand",
            scatter=False,
            color="green",
            ax=ax,
        )

        ax.set_title("Scatter Plot of Demand vs Price per kg with Regression Line", fontsize=14)
        ax.set_xlabel("Price per kg", fontsize=12)
        ax.set_ylabel("Demand", fontsize=12)
        for i in ax.get_xticklabels() + ax.get_yticklabels():
            i.set_fontsize(10)
        plt.tight_layout()
    plt.close(fig)
    return fig

def plot_density_weekday_weekend(df, style="seaborn", plot_size=(10, 8)):
    with plt.style.context(style=style):
        fig, ax = plt.subplots(figsize=plot_size)

        # Plot density for weekdays
        sns.kdeplot(
            df[df["weekend"] == 0]["demand"],
            color="blue",
            label="Weekday",
            ax=ax,
            fill=True,
            alpha=0.15,
        )

        # Plot density for weekends
        sns.kdeplot(
            df[df["weekend"] == 1]["demand"],
            color="green",
            label="Weekend",
            ax=ax,
            fill=True,
            alpha=0.15,
        )

        ax.set_title("Density Plot of Demand by Weekday/Weekend", fontsize=14)
        ax.set_xlabel("Demand", fontsize=12)
        ax.legend(fontsize=12)
        for i in ax.get_xticklabels() + ax.get_yticklabels():
            i.set_fontsize(10)

        plt.tight_layout()
    plt.close(fig)
    return fig

def plot_coefficients(model, feature_names, style="seaborn", plot_size=(10, 8)):
    try:
        with plt.style.context(style=style):
            fig, ax = plt.subplots(figsize=plot_size)
            ax.barh(feature_names, model.coef_)
            ax.set_title("Coefficient Plot", fontsize=14)
            ax.set_xlabel("Coefficient Value", fontsize=12)
            ax.set_ylabel("Features", fontsize=12)
            plt.tight_layout()
        plt.close(fig)
        return fig
    except Exception:
        return None

def plot_residuals(y_test, y_pred, style="seaborn", plot_size=(10, 8)):
    residuals = y_test - y_pred

    with plt.style.context(style=style):
        fig, ax = plt.subplots(figsize=plot_size)
        sns.residplot(
            x=y_pred,
            y=residuals,
            lowess=True,
            ax=ax,
            line_kws={"color": "red", "lw": 1},
        )

        ax.axhline(y=0, color="black", linestyle="--")
        ax.set_title("Residual Plot", fontsize=14)
        ax.set_xlabel("Predicted values", fontsize=12)
        ax.set_ylabel("Residuals", fontsize=12)

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontsize(10)

        plt.tight_layout()

    plt.close(fig)
    return fig

def plot_prediction_error(y_test, y_pred, style="seaborn", plot_size=(10, 8)):
    with plt.style.context(style=style):
        fig, ax = plt.subplots(figsize=plot_size)
        ax.scatter(y_pred, y_test - y_pred)
        ax.axhline(y=0, color="red", linestyle="--")
        ax.set_title("Prediction Error Plot", fontsize=14)
        ax.set_xlabel("Predicted Values", fontsize=12)
        ax.set_ylabel("Errors", fontsize=12)
        plt.tight_layout()
    plt.close(fig)
    return fig

def plot_qq(y_test, y_pred, style="seaborn", plot_size=(10, 8)):
    residuals = y_test - y_pred
    with plt.style.context(style=style):
        fig, ax = plt.subplots(figsize=plot_size)
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title("QQ Plot", fontsize=14)
        plt.tight_layout()
    plt.close(fig)
    return fig

def plot_correlation_matrix_and_save(
    df, style="seaborn", plot_size=(10, 8), path="/tmp/corr_plot.png"
):
    with plt.style.context(style=style):
        fig, ax = plt.subplots(figsize=plot_size)

        # Calculate the correlation matrix
        corr = df.corr()

        # Generate a mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(
            corr,
            mask=mask,
            cmap="coolwarm",
            vmax=0.3,
            center=0,
            square=True,
            linewidths=0.5,
            annot=True,
            fmt=".2f",
        )

        ax.set_title("Feature Correlation Matrix", fontsize=14)
        plt.tight_layout()

    plt.close(fig)
    # convert to filesystem path spec for os compatibility
    save_path = pathlib.Path(path)
    fig.savefig(save_path)