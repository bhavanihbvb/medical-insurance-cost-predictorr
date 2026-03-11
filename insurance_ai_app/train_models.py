import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def train_and_save(csv_path="insurance.csv", output_path="models.pkl"):
    df = pd.read_csv(csv_path)
    df["age_bmi"] = df["age"] * df["bmi"]

    X = df.drop("charges", axis=1)
    y = df["charges"]

    num_features = ["age", "bmi", "children", "age_bmi"]
    cat_features = ["sex", "smoker", "region"]

    numeric_pipeline = Pipeline([("scaler", StandardScaler())])
    categorical_pipeline = Pipeline([
        ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline,     num_features),
        ("cat", categorical_pipeline, cat_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model_defs = {
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42
        ),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Linear Regression": LinearRegression(),
    }

    trained_models = {}
    results = []

    for name, model in model_defs.items():
        pipe = Pipeline([("preprocessing", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)

        preds = pipe.predict(X_test)
        rmse  = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae   = float(mean_absolute_error(y_test, preds))
        r2    = float(r2_score(y_test, preds))
        cv    = float(cross_val_score(pipe, X, y, cv=5, scoring="r2").mean())

        trained_models[name] = pipe
        results.append({
            "Model": name,
            "R2":    round(r2,   4),
            "RMSE":  round(rmse, 2),
            "MAE":   round(mae,  2),
            "CV_R2": round(cv,   4),
        })
        print(f"  checkmark {name:22s}  R2={r2:.4f}  RMSE=${rmse:,.0f}")

    best_name = max(results, key=lambda x: x["R2"])["Model"]

    payload = {
        "models":   trained_models,
        "results":  results,
        "best":     best_name,
        "num_feat": num_features,
        "cat_feat": cat_features,
    }

    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved to {output_path}  (best: {best_name})")
    return payload


if __name__ == "__main__":
    train_and_save()