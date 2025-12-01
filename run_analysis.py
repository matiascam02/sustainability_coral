import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, roc_auc_score, precision_recall_fscore_support

# --- Configuration ---
DATA_PATH = Path("data/SSP_Data.xlsx")
OUTPUT_DIR = Path("analysis_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
ID_COLUMN = "ID"
TARGET_COLUMN = "Val_Score"
BIC_THRESHOLD = 0.75
MAX_MISSING_RATE = 0.95

# --- 1. Load Data ---
print("Loading data...")
raw_df = pd.read_excel(DATA_PATH, sheet_name="Data")
raw_df.set_index("Index", inplace=True)

# Create Classification Target
raw_df["is_bic"] = (raw_df[TARGET_COLUMN] >= BIC_THRESHOLD).astype(int)

# --- 2. Feature Selection ---
GI_RANGE_START = "Activities - Raw material extraction"
GI_RANGE_END = "Number of workers - Indirect employed"
gi_block = list(raw_df.loc[:, GI_RANGE_START:GI_RANGE_END].columns)
GI_CATEGORICAL = ["Country", "Assessment type", "Assessment Year"]
GI_FEATURES = GI_CATEGORICAL + gi_block
SAQ_FEATURES = [col for col in raw_df.columns if col.startswith("Q")]

FEATURE_SETS = {
    "gi_only": {"features": GI_FEATURES},
    "gi_plus_saq": {"features": GI_FEATURES + SAQ_FEATURES},
}

# --- 3. Preprocessing Helpers ---
def build_preprocessor(df: pd.DataFrame, features: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    numeric_cols = df[features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df[features].select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)) # sparse_output=False for SHAP compatibility
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols

# --- 4. Split Data (Supplier Stratified) ---
print("Splitting data...")
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(splitter.split(raw_df, groups=raw_df[ID_COLUMN]))
train_df = raw_df.iloc[train_idx].copy()
test_df = raw_df.iloc[test_idx].copy()

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# --- 5. Prepare Feature Bundles ---
feature_bundles = {}
for key, meta in FEATURE_SETS.items():
    feature_list = meta["features"]
    
    # Filter high missing
    missing_rates = train_df[feature_list].isna().mean()
    filtered_features = missing_rates[missing_rates <= MAX_MISSING_RATE].index.tolist()
    
    preprocessor, num_cols, cat_cols = build_preprocessor(train_df, filtered_features)
    
    feature_bundles[key] = {
        "X_train": train_df[filtered_features],
        "X_test": test_df[filtered_features],
        "y_train_reg": train_df[TARGET_COLUMN],
        "y_test_reg": test_df[TARGET_COLUMN],
        "y_train_clf": train_df["is_bic"],
        "y_test_clf": test_df["is_bic"],
        "preprocessor": preprocessor,
        "features": filtered_features
    }

# --- 6. Model Training & Evaluation ---
print("Training models...")
results = []
model_registry = {}

for key, bundle in feature_bundles.items():
    # Regression
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg_pipeline = Pipeline([("prep", bundle["preprocessor"]), ("model", reg_model)])
    reg_pipeline.fit(bundle["X_train"], bundle["y_train_reg"])
    y_pred_reg = reg_pipeline.predict(bundle["X_test"])
    
    rmse = np.sqrt(mean_squared_error(bundle["y_test_reg"], y_pred_reg))
    r2 = r2_score(bundle["y_test_reg"], y_pred_reg)
    
    results.append({
        "feature_set": key,
        "task": "regression",
        "rmse": rmse,
        "r2": r2
    })
    model_registry[f"{key}_reg"] = reg_pipeline
    
    # Classification
    clf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_pipeline = Pipeline([("prep", bundle["preprocessor"]), ("model", clf_model)])
    clf_pipeline.fit(bundle["X_train"], bundle["y_train_clf"])
    y_pred_clf = clf_pipeline.predict(bundle["X_test"])
    y_prob_clf = clf_pipeline.predict_proba(bundle["X_test"])[:, 1]
    
    acc = accuracy_score(bundle["y_test_clf"], y_pred_clf)
    auc = roc_auc_score(bundle["y_test_clf"], y_prob_clf)
    
    results.append({
        "feature_set": key,
        "task": "classification",
        "accuracy": acc,
        "auc": auc
    })
    model_registry[f"{key}_clf"] = clf_pipeline

# Save metrics
with open(OUTPUT_DIR / "metrics.json", "w") as f:
    json.dump(results, f, indent=2)

# --- 7. Feature Importance (SHAP) ---
print("Generating SHAP plots...")
# Load column metadata for mapping
column_meta = pd.read_excel(DATA_PATH, sheet_name="Column Explanation")
column_meta["Column header"] = column_meta["Column header"].astype(str)
# Fix potential duplicates in lookup by taking first
topic_lookup = column_meta.drop_duplicates("Column header").set_index("Column header")["Topic"].fillna("-").to_dict()

for key in FEATURE_SETS.keys():
    pipeline = model_registry[f"{key}_reg"]
    bundle = feature_bundles[key]
    
    # Sample for SHAP
    X_sample = bundle["X_train"].sample(n=min(200, len(bundle["X_train"])), random_state=42)
    
    # Transform
    transformer = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    X_transformed = transformer.transform(X_sample)
    feature_names = transformer.get_feature_names_out()
    
    # Explain
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_transformed)
    
    # Summary Plot
    plt.figure(figsize=(10, 6))
    # We manually create a bar plot of mean abs shap values for better control/saving
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feature_names, "importance": mean_abs_shap})
    
    # Clean feature names for display (remove "num__", "cat__")
    shap_df["clean_feature"] = shap_df["feature"].str.replace(r"^(num__|cat__)", "", regex=True)
    # Map topics
    shap_df["topic"] = shap_df["clean_feature"].map(topic_lookup)
    
    top_10 = shap_df.sort_values("importance", ascending=False).head(10)
    
    sns.barplot(data=top_10, x="importance", y="clean_feature", hue="topic", dodge=False)
    plt.title(f"Top Feature Importance ({key})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"shap_{key}.png")
    plt.close()

# --- 8. Risk Segmentation ---
print("Generating Risk Segmentation...")
# Use GI+SAQ Regression model for risk scoring
best_reg_pipeline = model_registry["gi_plus_saq_reg"]
bundle = feature_bundles["gi_plus_saq"]
y_pred_final = best_reg_pipeline.predict(bundle["X_test"])

def assign_risk_tier(score):
    if score >= 0.80: return "Low Risk"
    if score >= 0.70: return "Moderate Risk"
    if score >= 0.60: return "Medium Risk"
    if score >= 0.50: return "High Risk"
    return "Critical Risk"

risk_df = test_df.copy()
risk_df["predicted_score"] = y_pred_final
risk_df["risk_tier"] = risk_df["predicted_score"].apply(assign_risk_tier)

plt.figure(figsize=(8, 5))
order = ["Critical Risk", "High Risk", "Medium Risk", "Moderate Risk", "Low Risk"]
sns.countplot(data=risk_df, x="risk_tier", order=order, palette="flare")
plt.title("Supplier Risk Segmentation (Test Set)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "risk_segmentation.png")
plt.close()

# --- 9. Fairness Analysis ---
print("Generating Fairness Analysis...")
# Use GI+SAQ Classification model
best_clf_pipeline = model_registry["gi_plus_saq_clf"]
y_pred_class = best_clf_pipeline.predict(bundle["X_test"])
risk_df["pred_class"] = y_pred_class

# Define regions
risk_df["region_group"] = np.where(risk_df["Country"] == "China", "China", "Rest of World")

# Calculate FNR per region
def compute_fnr(df):
    y_true = df["is_bic"]
    y_pred = df["pred_class"]
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    return fn / (fn + tp) if (fn + tp) > 0 else 0

fnr_data = risk_df.groupby("region_group").apply(compute_fnr).reset_index(name="fnr")

plt.figure(figsize=(6, 5))
sns.barplot(data=fnr_data, x="region_group", y="fnr", palette="crest")
plt.ylim(0, 1)
plt.title("False Negative Rate by Region (Fairness Check)")
plt.ylabel("False Negative Rate (Missed BIC)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fairness_check.png")
plt.close()

print("Analysis complete. Outputs saved to", OUTPUT_DIR)
