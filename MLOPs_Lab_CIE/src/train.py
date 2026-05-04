import pandas as pd
import numpy as np
import json
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data/training_data.csv")

X = df.drop("outbreak_risk_score", axis=1)
y = df["outbreak_risk_score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "SVR": SVR(),
    "RandomForest": RandomForestRegressor()
}

results = []
mlflow.set_experiment("commuhealth-outbreak-risk-score")

for name, model in models.items():
    with mlflow.start_run():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_param("model", name)
        mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
        mlflow.set_tag("priority", "high")

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })

best = min(results, key=lambda x: x["rmse"])

output = {
    "experiment_name": "commuhealth-outbreak-risk-score",
    "models": results,
    "best_model": best["name"],
    "best_metric_name": "rmse",
    "best_metric_value": best["rmse"]
}

with open("results/step1_s1.json", "w") as f:
    json.dump(output, f, indent=4)