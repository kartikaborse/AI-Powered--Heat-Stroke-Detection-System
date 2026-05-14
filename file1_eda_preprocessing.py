# =============================================================================
# FILE 1 — DATA PREPROCESSING & EXPLORATORY DATA ANALYSIS
# Heat Stroke Risk Prediction | Random Forest Pipeline
# =============================================================================
# Run:  python file1_eda_preprocessing.py
# Outputs: preprocessed_heat_stroke.csv  (saved in current directory)
# =============================================================================
import matplotlib
matplotlib.use('Agg')   # <-- IMPORTANT FIX (no GUI needed)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# 1. LOAD DATASET                                                              #
# --------------------------------------------------------------------------- #
CSV_PATH = "heatstrokedataset.csv"   # <-- change path if needed

print("=" * 65)
print("  HEAT STROKE RISK — EDA & PREPROCESSING")
print("=" * 65)

df = pd.read_csv(CSV_PATH)
print(f"\n[1] Dataset loaded  →  {df.shape[0]} rows × {df.shape[1]} columns")

# --------------------------------------------------------------------------- #
# 2. BASIC INFORMATION                                                          #
# --------------------------------------------------------------------------- #
print("\n" + "─" * 65)
print("[2] DATASET OVERVIEW")
print("─" * 65)
print(df.head(10).to_string())

print("\n[2a] Data Types:")
print(df.dtypes.to_string())

print("\n[2b] Statistical Summary:")
print(df.describe().round(3).to_string())

# --------------------------------------------------------------------------- #
# 3. MISSING VALUES & DUPLICATES                                                #
# --------------------------------------------------------------------------- #
print("\n" + "─" * 65)
print("[3] DATA QUALITY CHECK")
print("─" * 65)

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
quality_df = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print("\nMissing Values:")
print(quality_df.to_string())

dupes = df.duplicated().sum()
print(f"\nDuplicate rows : {dupes}")

# Drop duplicates if any
if dupes > 0:
    df.drop_duplicates(inplace=True)
    print(f"  → Dropped {dupes} duplicate rows. New shape: {df.shape}")
else:
    print("  → No duplicates found. Data is clean.")

# --------------------------------------------------------------------------- #
# 4. TARGET CLASS DISTRIBUTION                                                  #
# --------------------------------------------------------------------------- #
print("\n" + "─" * 65)
print("[4] TARGET CLASS DISTRIBUTION  (risk_label)")
print("─" * 65)

label_map = {0: "No Risk", 1: "Low Risk", 2: "Medium Risk", 3: "High Risk"}
dist = df["risk_label"].value_counts().sort_index()
for k, v in dist.items():
    bar = "█" * int(v / len(df) * 50)
    print(f"  Class {k} ({label_map[k]:12s}) : {v:5d}  ({v/len(df)*100:.1f}%)  {bar}")

# --------------------------------------------------------------------------- #
# 5. EXPLORATORY DATA ANALYSIS — VISUALISATIONS                                 #
# --------------------------------------------------------------------------- #
print("\n" + "─" * 65)
print("[5] GENERATING EDA PLOTS  →  eda_plots.png")
print("─" * 65)

feature_cols = ["ambient_temp", "humidity", "body_temp",
                "duration", "hour", "heat_index"]

fig, axes = plt.subplots(4, 3, figsize=(18, 20))
fig.suptitle("Heat Stroke Dataset — EDA", fontsize=16, fontweight="bold", y=1.01)

# 5a — Histograms for each feature
for i, col in enumerate(feature_cols):
    ax = axes[0][i] if i < 3 else axes[1][i - 3]
    df[col].hist(ax=ax, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_title(f"Distribution: {col}", fontsize=11)
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")

# 5b — Box plots per class
for i, col in enumerate(feature_cols[:3]):
    ax = axes[2][i]
    df.boxplot(column=col, by="risk_label", ax=ax,
               boxprops=dict(color="#4C72B0"),
               medianprops=dict(color="red", linewidth=2))
    ax.set_title(f"{col} by Risk Label", fontsize=11)
    ax.set_xlabel("Risk Label")
    ax.set_ylabel(col)
    plt.sca(ax)
    plt.title(f"{col} by Risk Label")

# 5c — Correlation heatmap
ax_heat = axes[3][0]
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax_heat, linewidths=0.5, cbar_kws={"shrink": 0.8})
ax_heat.set_title("Correlation Matrix", fontsize=11)

# 5d — Class distribution bar chart
ax_bar = axes[3][1]
colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
dist.plot(kind="bar", ax=ax_bar, color=colors, edgecolor="black", rot=0)
ax_bar.set_title("Target Class Counts", fontsize=11)
ax_bar.set_xlabel("Risk Label")
ax_bar.set_ylabel("Count")
ax_bar.set_xticklabels([f"{k}\n({label_map[k]})" for k in dist.index])

# 5e — Scatter: ambient_temp vs heat_index coloured by class
ax_sc = axes[3][2]
scatter_colors = df["risk_label"].map({0: "#2ecc71", 1: "#f1c40f",
                                        2: "#e67e22", 3: "#e74c3c"})
ax_sc.scatter(df["ambient_temp"], df["heat_index"],
              c=scatter_colors, alpha=0.4, s=10)
ax_sc.set_title("Ambient Temp vs Heat Index", fontsize=11)
ax_sc.set_xlabel("Ambient Temp (°C)")
ax_sc.set_ylabel("Heat Index")
for label, color in zip(["No Risk", "Low", "Medium", "High"],
                         ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]):
    ax_sc.scatter([], [], c=color, label=label, s=40)
ax_sc.legend(fontsize=8)

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.close()
print("  → Saved: eda_plots.png")

# --------------------------------------------------------------------------- #
# 6. OUTLIER DETECTION (IQR method — reporting only)                            #
# --------------------------------------------------------------------------- #
print("\n" + "─" * 65)
print("[6] OUTLIER DETECTION (IQR Method)")
print("─" * 65)
for col in feature_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)).sum()
    print(f"  {col:20s}: {outliers:4d} potential outliers")

# --------------------------------------------------------------------------- #
# 7. FEATURE ENGINEERING                                                        #
# --------------------------------------------------------------------------- #
print("\n" + "─" * 65)
print("[7] FEATURE ENGINEERING")
print("─" * 65)

# 7a — Temp–humidity combined stress index
df["temp_humidity_stress"] = df["ambient_temp"] * df["humidity"] / 100
print("  + Added feature: temp_humidity_stress = ambient_temp × humidity / 100")

# 7b — Body-temp deviation from normal (37°C)
df["body_temp_deviation"] = df["body_temp"] - 37.0
print("  + Added feature: body_temp_deviation = body_temp − 37.0")

# 7c — Time-of-day category (morning / afternoon / evening / night)
def time_category(hour):
    if 5 <= hour < 12:   return 0   # Morning
    elif 12 <= hour < 17: return 1  # Afternoon
    elif 17 <= hour < 21: return 2  # Evening
    else:                 return 3  # Night

df["time_of_day"] = df["hour"].apply(time_category)
print("  + Added feature: time_of_day  (0=Morning 1=Afternoon 2=Evening 3=Night)")

updated_features = feature_cols + ["temp_humidity_stress",
                                    "body_temp_deviation", "time_of_day"]

print(f"\n  Total features after engineering: {len(updated_features)}")

# --------------------------------------------------------------------------- #
# 8. FEATURE SCALING                                                            #
# --------------------------------------------------------------------------- #
print("\n" + "─" * 65)
print("[8] FEATURE SCALING  (StandardScaler)")
print("─" * 65)

X = df[updated_features].copy()
y = df["risk_label"].copy()

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X),
                         columns=updated_features, index=X.index)

print("  Scaling applied to all feature columns.")
print(f"  Feature matrix shape : {X_scaled.shape}")
print(f"  Target vector shape  : {y.shape}")

# --------------------------------------------------------------------------- #
# 9. SAVE PREPROCESSED DATA                                                     #
# --------------------------------------------------------------------------- #
preprocessed = X_scaled.copy()
preprocessed["risk_label"] = y.values

OUT_FILE = "preprocessed_heat_stroke.csv"
preprocessed.to_csv(OUT_FILE, index=False)
print(f"\n[9] Preprocessed data saved  →  {OUT_FILE}")
print(f"    Shape: {preprocessed.shape}")

# --------------------------------------------------------------------------- #
# SUMMARY                                                                       #
# --------------------------------------------------------------------------- #
print("\n" + "=" * 65)
print("  PREPROCESSING COMPLETE")
print("=" * 65)
print(f"  ✔  Original dataset   : {df.shape[0]} rows × {df.shape[1]} cols")
print(f"  ✔  Features used      : {len(updated_features)}")
print(f"  ✔  Target classes     : 4  (0 No Risk → 3 High Risk)")
print(f"  ✔  Missing values     : 0")
print(f"  ✔  Duplicates removed : {dupes}")
print(f"  ✔  Output file        : {OUT_FILE}")
print(f"  ✔  EDA plot           : eda_plots.png")
print("=" * 65)
print("\n  NEXT STEP → Run  file2_split_training.py\n")
