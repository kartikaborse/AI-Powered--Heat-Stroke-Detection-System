# =============================================================================
# FILE 2 — DATA SPLITTING & TRAINING SET PREPARATION
# Heat Stroke Risk Prediction | Random Forest Pipeline
# =============================================================================
# Run   :  python file2_split_training.py
# Input :  heat_stroke_dataset.csv   (raw — all preprocessing done inside)
# Output:  train_set.csv, test_set.csv  (saved in current directory)
# =============================================================================
import matplotlib
matplotlib.use('Agg')   # <-- IMPORTANT FIX (no GUI needed)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# CONSTANTS                                                                     #
# --------------------------------------------------------------------------- #
CSV_PATH    = "heatstrokedataset.csv"   # <-- change path if needed
TEST_SIZE   = 0.20     # 80 % train / 20 % test
RANDOM_SEED = 42

LABEL_MAP = {0: "No Risk", 1: "Low Risk", 2: "Medium Risk", 3: "High Risk"}

print("=" * 65)
print("  HEAT STROKE RISK — DATA SPLITTING & TRAINING PREP")
print("=" * 65)

# =========================================================================== #
# STEP 1 — RELOAD & RE-PREPROCESS (this file is self-contained)               #
# =========================================================================== #
print("\n[1] Loading raw dataset …")
df = pd.read_csv(CSV_PATH)
df.drop_duplicates(inplace=True)
print(f"    Shape after dedup : {df.shape}")

feature_cols = ["ambient_temp", "humidity", "body_temp",
                "duration", "hour", "heat_index"]

# --- Feature engineering (mirrors File 1) ---
df["temp_humidity_stress"] = df["ambient_temp"] * df["humidity"] / 100
df["body_temp_deviation"]  = df["body_temp"] - 37.0

def time_category(hour):
    if 5 <= hour < 12:    return 0
    elif 12 <= hour < 17: return 1
    elif 17 <= hour < 21: return 2
    else:                 return 3

df["time_of_day"] = df["hour"].apply(time_category)

all_features = feature_cols + ["temp_humidity_stress",
                                "body_temp_deviation", "time_of_day"]

X = df[all_features].copy()
y = df["risk_label"].copy()

# --- Scaling ---
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X),
                         columns=all_features, index=X.index)
print(f"    Features           : {len(all_features)}")
print(f"    Feature scaling    : StandardScaler applied")

# =========================================================================== #
# STEP 2 — STRATIFIED TRAIN / TEST SPLIT                                       #
# =========================================================================== #
print(f"\n[2] Splitting  →  {int((1-TEST_SIZE)*100)}% train / "
      f"{int(TEST_SIZE*100)}% test  (stratified, seed={RANDOM_SEED})")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y          # preserve class proportions
)

print(f"    Training samples   : {len(X_train)}")
print(f"    Test samples       : {len(X_test)}")
print(f"    Total              : {len(X_train) + len(X_test)}")

# =========================================================================== #
# STEP 3 — CLASS DISTRIBUTION VERIFICATION                                     #
# =========================================================================== #
print("\n[3] Class Distribution Check")
print("─" * 55)

train_dist = Counter(y_train)
test_dist  = Counter(y_test)
full_dist  = Counter(y)

print(f"  {'Class':<6} {'Label':<14} {'Full':>7} {'Train':>7} {'Test':>6}")
print("  " + "─" * 44)
for cls in sorted(full_dist.keys()):
    print(f"  {cls:<6} {LABEL_MAP[cls]:<14} "
          f"{full_dist[cls]:>6}  "
          f"{train_dist[cls]:>6}  "
          f"{test_dist[cls]:>6}")

# =========================================================================== #
# STEP 4 — VISUALISE SPLIT                                                     #
# =========================================================================== #
print("\n[4] Generating split visualisation  →  split_analysis.png")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Train / Test Split Analysis", fontsize=14, fontweight="bold")
colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]

# 4a — Full dataset class distribution
labels_clean = [f"{k}\n{LABEL_MAP[k]}" for k in sorted(full_dist.keys())]
axes[0].bar(labels_clean,
            [full_dist[k] for k in sorted(full_dist.keys())],
            color=colors, edgecolor="black")
axes[0].set_title("Full Dataset")
axes[0].set_ylabel("Count")

# 4b — Train set class distribution
axes[1].bar(labels_clean,
            [train_dist[k] for k in sorted(full_dist.keys())],
            color=colors, edgecolor="black")
axes[1].set_title(f"Training Set ({len(X_train)} samples)")
axes[1].set_ylabel("Count")

# 4c — Test set class distribution
axes[2].bar(labels_clean,
            [test_dist[k] for k in sorted(full_dist.keys())],
            color=colors, edgecolor="black")
axes[2].set_title(f"Test Set ({len(X_test)} samples)")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.savefig("split_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("    Saved: split_analysis.png")

# =========================================================================== #
# STEP 5 — FEATURE CORRELATION IN TRAINING SET                                 #
# =========================================================================== #
print("\n[5] Feature Correlation (Training Set)")
print("─" * 55)

train_df_corr = X_train.copy()
train_df_corr["risk_label"] = y_train.values
corr_with_target = train_df_corr.corr()["risk_label"].drop("risk_label") \
                                                       .abs().sort_values(ascending=False)
print("\n  Feature Importance (correlation with target):")
for feat, val in corr_with_target.items():
    bar = "█" * int(val * 30)
    print(f"  {feat:<25} {val:.4f}  {bar}")

# =========================================================================== #
# STEP 6 — SAVE SPLIT DATASETS                                                 #
# =========================================================================== #
print("\n[6] Saving split datasets …")

train_out = X_train.copy()
train_out["risk_label"] = y_train.values
test_out  = X_test.copy()
test_out["risk_label"]  = y_test.values

train_out.to_csv("train_set.csv", index=False)
test_out.to_csv("test_set.csv",  index=False)

print("    ✔  train_set.csv  saved  →  shape:", train_out.shape)
print("    ✔  test_set.csv   saved  →  shape:", test_out.shape)

# =========================================================================== #
# STEP 7 — BASIC STATISTICS COMPARISON                                         #
# =========================================================================== #
print("\n[7] Train vs Test Feature Statistics")
print("─" * 55)

for col in all_features:
    tr_mean = X_train[col].mean()
    te_mean = X_test[col].mean()
    print(f"  {col:<25}  train μ={tr_mean:6.3f}   test μ={te_mean:6.3f}")

# =========================================================================== #
# SUMMARY                                                                      #
# =========================================================================== #
print("\n" + "=" * 65)
print("  SPLITTING COMPLETE")
print("=" * 65)
print(f"  ✔  Total samples     : {len(df)}")
print(f"  ✔  Train samples     : {len(X_train)}  ({len(X_train)/len(df)*100:.1f}%)")
print(f"  ✔  Test samples      : {len(X_test)}  ({len(X_test)/len(df)*100:.1f}%)")
print(f"  ✔  Stratified split  : Yes")
print(f"  ✔  Random seed       : {RANDOM_SEED}")
print(f"  ✔  train_set.csv     : saved")
print(f"  ✔  test_set.csv      : saved")
print(f"  ✔  split_analysis.png: saved")
print("=" * 65)
print("\n  NEXT STEP → Run  file3_random_forest_model.py\n")
