"""Train first ML models on the YAQS noise twin dataset.

Pipeline:
YAQS dataset
-> train ML model
-> test whether ML predicts noisy observables
-> test whether ML can infer noise parameters
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASET_CSV = Path("examples/noise_twin/output/noise_twin_dataset.csv")


def train_forward_model(df: pd.DataFrame) -> None:
    """Predict final observable from Hamiltonian + noise parameters.

    This corresponds to:
        noise parameters -> noisy experiment output
    """

    feature_cols = [
        "length",
        "J",
        "g",
        "elapsed_time",
        "dt",
        "num_traj",
        "max_bond_dim",
        "mean_gamma_lowering",
        "std_gamma_lowering",
        "mean_gamma_dephasing",
        "std_gamma_dephasing",
    ]

    # Also include qubit-level hardware-like noise columns.
    feature_cols += [
        col for col in df.columns
        if col.startswith("gamma_lowering_q") or col.startswith("gamma_dephasing_q")
    ]

    target_col = "final_mean_x"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    models = {
        "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2,
        ),
    }

    print("\nForward prediction:")
    print("Goal: predict final observable from noise profile")

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        print(f"{name:20s} MAE = {mae:.6f} | R2 = {r2:.4f}")


def train_inverse_model(df: pd.DataFrame) -> None:
    """Infer noise strength from the simulation output.

    This corresponds to:
        observed output -> noise parameter estimate

    For the first simple version, we infer mean dephasing strength.
    """

    feature_cols = [
        "final_mean_x",
        "length",
        "J",
        "g",
        "elapsed_time",
        "dt",
    ]

    target_col = "mean_gamma_dephasing"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    models = {
        "linear_regression": make_pipeline(StandardScaler(), LinearRegression()),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2,
        ),
    }

    print("\nInverse prediction:")
    print("Goal: infer mean dephasing noise from observed output")

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)

        print(f"{name:20s} MAE = {mae:.6e} | R2 = {r2:.4f}")


def main() -> None:
    if not DATASET_CSV.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_CSV}\n"
            "Run generate_noise_twin_dataset.py first."
        )

    df = pd.read_csv(DATASET_CSV)

    print(f"Loaded dataset: {DATASET_CSV}")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    train_forward_model(df)
    train_inverse_model(df)


if __name__ == "__main__":
    main()
