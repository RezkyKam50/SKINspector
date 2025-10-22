import pandas as pd, numpy as np
import matplotlib.pyplot as plt, matplotlib, torch
matplotlib.use("Qt5Agg")

path_to_dataset = f"./datasets/SkinCAP/skincap_v240623.csv"

df = pd.read_csv(path_to_dataset)

class_counts = df["disease"].value_counts().sort_index()
plt.figure(figsize=(12, 8))
class_counts.plot(kind="bar")

num_samples = len(df)
weights = num_samples / (len(class_counts) * class_counts)
class_weights = torch.tensor(weights.values, dtype=torch.float32)

print("Class weights:", class_weights)

plt.title("Disease Class Distribution")
plt.xlabel("Disease")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
