import json
import pandas as pd

df = pd.read_csv("data/raw_data/Training.csv")

# Drop target + fluid_overload + unnamed
df = df.drop(columns=["prognosis", "fluid_overload"], errors="ignore")
drop_cols = [c for c in df.columns if "unnamed" in c.lower()]
df = df.drop(columns=drop_cols)

feature_order = list(df.columns)

with open("models/trained/feature_order.json", "w") as f:
    json.dump(feature_order, f, indent=2)

print("Saved feature order:", len(feature_order), "features.")
