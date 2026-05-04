import argparse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--population_density", type=float)
parser.add_argument("--vaccination_pct", type=float)
parser.add_argument("--sanitation_index", type=float)
parser.add_argument("--rainfall_mm", type=float)

args = parser.parse_args()

data = pd.DataFrame([vars(args)])

# dummy prediction
prediction = 50.0

print(prediction)