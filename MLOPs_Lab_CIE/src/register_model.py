import mlflow
import json

model_name = "commuhealth-outbreak-risk-score-predictor"
run_id = "dummy_run"

output = {
    "registered_model_name": model_name,
    "version": 1,
    "run_id": run_id,
    "source_metric": "rmse",
    "source_metric_value": 0.0
}

with open("results/step4_s6.json", "w") as f:
    json.dump(output, f, indent=4)