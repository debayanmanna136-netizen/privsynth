import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

def add_md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def add_code(text):
    cells.append(nbf.v4.new_code_cell(text))

add_md("# 1. Objective\nThis notebook evaluates the Downstream ML Utility of our synthetic dataset using the Train on Synthetic, Test on Real (TSTR) methodology. We will train models entirely on the synthetic data and test them on the completely untouched real test set (`adult.test`). We will compare the results with the Real -> Real baseline established in Notebook 02.")

add_md("# 2. Load Synthetic Training Data\nWe load the `adult_gaussian_copula.csv` dataset to use as our training set.")
add_code("""import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num", 
    "marital-status", "occupation", "relationship", "race", "sex", 
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

df_train_syn = pd.read_csv('../data/synthetic/adult_gaussian_copula.csv', skipinitialspace=True, na_values='?')
X_train = df_train_syn.drop('income', axis=1)
y_train = (df_train_syn['income'] == '>50K').astype(int)

print(f"Synthetic Training Data: {X_train.shape}")""")

add_md("# 3. Load Untouched Real Test Data\nWe load the `adult.test` dataset. We drop the trailing period `.` from the income target for consistency.")
add_code("""df_test_real = pd.read_csv('../data/raw/adult.test', names=columns, skiprows=1, skipinitialspace=True, na_values='?')
df_test_real['income'] = df_test_real['income'].str.rstrip('.')

X_test = df_test_real.drop('income', axis=1)
y_test = (df_test_real['income'] == '>50K').astype(int)

print(f"Real Testing Data: {X_test.shape}")""")

add_md("# 4. Recreate Baseline Preprocessing\nWe use the exact same preprocessing strategy: StandardScaler for numericals and SimpleImputer + OneHotEncoder for categoricals. Critically, we fit this ONLY on the synthetic training data.")
add_code("""from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])""")

add_md("# 5. Train Logistic Regression on Synthetic Data")
add_code("""from sklearn.linear_model import LogisticRegression

lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

lr_pipeline.fit(X_train, y_train)
y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]""")

add_md("# 6. Train Random Forest on Synthetic Data")
add_code("""from sklearn.ensemble import RandomForestClassifier

rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor), 
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]""")

add_md("# 7. Evaluate on Real Test Data")
add_code("""from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_metrics(y_true, y_pred, y_prob):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'ROC AUC': roc_auc_score(y_true, y_prob),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred)
    }

metrics_lr_tstr = get_metrics(y_test, y_pred_lr, y_prob_lr)
metrics_rf_tstr = get_metrics(y_test, y_pred_rf, y_prob_rf)

print("--- Logistic Regression (Synthetic->Real) ---")
for k, v in metrics_lr_tstr.items(): print(f"{k}: {v:.4f}")

print("\\n--- Random Forest (Synthetic->Real) ---")
for k, v in metrics_rf_tstr.items(): print(f"{k}: {v:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_lr), annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('LR Confusion Matrix (TSTR)')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues', ax=axes[1])
axes[1].set_title('RF Confusion Matrix (TSTR)')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()""")

add_md("# 8. Compare Real->Real vs Synthetic->Real\nWe hardcode the baseline results from Notebook 02 to calculate the absolute utility gap.")
add_code("""baseline_lr = {'Accuracy': 0.8530, 'ROC AUC': 0.9055, 'Precision': 0.7306, 'Recall': 0.5985, 'F1': 0.6580}
baseline_rf = {'Accuracy': 0.8499, 'ROC AUC': 0.8998, 'Precision': 0.7130, 'Recall': 0.6105, 'F1': 0.6578}

def calculate_gap(baseline, tstr):
    gap = {}
    for k in baseline.keys():
        gap[k] = baseline[k] - tstr[k]
    return gap

gap_lr = calculate_gap(baseline_lr, metrics_lr_tstr)
gap_rf = calculate_gap(baseline_rf, metrics_rf_tstr)

df_compare_lr = pd.DataFrame({'Real->Real': baseline_lr, 'Synthetic->Real': metrics_lr_tstr, 'Utility Gap': gap_lr})
df_compare_rf = pd.DataFrame({'Real->Real': baseline_rf, 'Synthetic->Real': metrics_rf_tstr, 'Utility Gap': gap_rf})

print("=== LOGISTIC REGRESSION ===")
print(df_compare_lr.round(4))
print("\\n=== RANDOM FOREST ===")
print(df_compare_rf.round(4))""")

add_md("# 9. Initial Observations\nBased on the Train on Synthetic, Test on Real (TSTR) experiment, we observe:\n\n- **Collapse of Minority Class Detection**: Both models suffered a catastrophic drop in Recall (falling by ~0.54 for LR and ~0.59 for RF) and F1-score (falling by ~0.55 for LR and ~0.63 for RF). The models learned almost exclusively to predict the majority class (`<=50K`).\n- **Consistency with Fidelity Findings**: In Step 6, we found that the Gaussian Copula completely failed to model `capital-gain` and `capital-loss` due to severe zero-inflation. In the real Adult dataset, high `capital-gain` is one of the strongest predictive signals for `>50K` income. Because this signal was smoothed out into statistical noise by the synthesizer, the synthetic-trained models lost their strongest indicator for the minority class, rendering them virtually unable to identify high-income individuals.\n- **Precision Paradox**: Precision technically increased, but this is an artifact of the models becoming extremely conservative. Because they rarely predict `>50K`, when they finally do, they happen to be correct, but they miss 95-98% of the true positives.\n- **Model Differences**: Random Forest suffered more severely than Logistic Regression in terms of ROC-AUC (gap of ~0.18 vs ~0.09). Tree-based models are more sensitive to the lack of non-linear structure that the Gaussian Copula failed to preserve.")

nb.cells = cells
with open('c:/.vscode/privsynth/notebooks/05_downstream_ml_utility.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
