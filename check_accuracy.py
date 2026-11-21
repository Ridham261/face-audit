import pandas as pd
import sys

CSV = r"D:\audit\results\dataset_test_results.csv"
df = pd.read_csv(CSV)
accuracy = df["is_valid"].mean()
total = len(df)
correct = int(df["is_valid"].sum())
print(f"Accuracy: {accuracy:.1%}  ({correct}/{total})")
if accuracy < 0.85:
    print(f"FAIL: accuracy {accuracy:.1%} is below 85% threshold")
    sys.exit(1)
print("PASS: accuracy meets threshold")
