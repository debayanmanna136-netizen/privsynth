import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

def evaluate_downstream_utility(
    synth_train_X: pd.DataFrame, synth_train_y: pd.Series,
    real_test_X: pd.DataFrame, real_test_y: pd.Series,
    num_cols: list, cat_cols_features: list
) -> dict:
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols_features)
        ])
    
    X_train_prep = preprocessor.fit_transform(synth_train_X)
    X_test_prep = preprocessor.transform(real_test_X)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_prep, synth_train_y)
    rf_pred = rf.predict(X_test_prep)
    rf_prob = rf.predict_proba(X_test_prep)[:, 1]
    
    lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_prep, synth_train_y)
    lr_pred = lr.predict(X_test_prep)
    lr_prob = lr.predict_proba(X_test_prep)[:, 1]
    
    return {
        "rf_roc_auc": float(roc_auc_score(real_test_y, rf_prob)),
        "rf_f1": float(f1_score(real_test_y, rf_pred)),
        "lr_roc_auc": float(roc_auc_score(real_test_y, lr_prob)),
        "lr_f1": float(f1_score(real_test_y, lr_pred))
    }
