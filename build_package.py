import os

base_dir = "c:/.vscode/privsynth/privsynth"
os.makedirs(base_dir, exist_ok=True)
os.makedirs(os.path.join(base_dir, "profiling"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "generators"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "evaluation"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "pipeline"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "utils"), exist_ok=True)

# Create __init__.py
for d in [base_dir, os.path.join(base_dir, "profiling"), os.path.join(base_dir, "generators"), os.path.join(base_dir, "evaluation"), os.path.join(base_dir, "pipeline"), os.path.join(base_dir, "utils")]:
    with open(os.path.join(d, "__init__.py"), "w") as f:
        f.write("")

# config.py
with open(os.path.join(base_dir, "config.py"), "w") as f:
    f.write('''import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_SYNTHETIC_DIR = os.path.join(BASE_DIR, 'data', 'synthetic')

CTGAN_DEFAULT_CONFIG = {
    "epochs": 25,
    "batch_size": 500,
    "embedding_dim": 256,
    "seed": 42
}
''')

# profiler.py
with open(os.path.join(base_dir, "profiling", "profiler.py"), "w") as f:
    f.write('''import pandas as pd

def profile_dataset(df: pd.DataFrame) -> dict:
    profile = {
        "num_rows": len(df),
        "num_cols": len(df.columns),
        "columns": list(df.columns),
        "numerical_cols": list(df.select_dtypes(include=['number']).columns),
        "categorical_cols": list(df.select_dtypes(exclude=['number']).columns),
        "missing_values_total": int(df.isna().sum().sum()),
        "missing_values_by_col": df.isna().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "unique_values_count": {col: df[col].nunique() for col in df.columns}
    }
    return profile
''')

# ctgan_generator.py
with open(os.path.join(base_dir, "generators", "ctgan_generator.py"), "w") as f:
    f.write('''import pandas as pd
import torch
import numpy as np
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer
from privsynth.config import CTGAN_DEFAULT_CONFIG

def generate_ctgan(df: pd.DataFrame, config=None) -> pd.DataFrame:
    cfg = config or CTGAN_DEFAULT_CONFIG
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    
    torch.manual_seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    
    synthesizer = CTGANSynthesizer(
        metadata,
        enforce_rounding=True,
        epochs=cfg['epochs'],
        batch_size=cfg['batch_size'],
        embedding_dim=cfg['embedding_dim'],
        verbose=False
    )
    
    synthesizer.fit(df)
    return synthesizer.sample(num_rows=len(df))
''')

# gaussian_copula_generator.py
with open(os.path.join(base_dir, "generators", "gaussian_copula_generator.py"), "w") as f:
    f.write('''import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
import numpy as np

def generate_gaussian_copula(df: pd.DataFrame, seed=42) -> pd.DataFrame:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)
    np.random.seed(seed)
    
    synthesizer = GaussianCopulaSynthesizer(metadata, enforce_rounding=True)
    synthesizer.fit(df)
    return synthesizer.sample(num_rows=len(df))
''')

# fidelity.py
with open(os.path.join(base_dir, "evaluation", "fidelity.py"), "w") as f:
    f.write('''import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from scipy.spatial.distance import jensenshannon

def evaluate_fidelity(real_df: pd.DataFrame, synth_df: pd.DataFrame, num_cols: list, cat_cols: list) -> dict:
    results = {}
    
    if num_cols:
        real_corr = real_df[num_cols].corr(numeric_only=True).fillna(0)
        synth_corr = synth_df[num_cols].corr(numeric_only=True).fillna(0)
        results['correlation_mae'] = float(np.abs(real_corr - synth_corr).mean().mean())
    
    ks_stats = {}
    for col in num_cols:
        stat, _ = ks_2samp(real_df[col].dropna(), synth_df[col].dropna())
        ks_stats[col] = float(stat)
    results['ks_statistic'] = ks_stats
    
    jsd_stats = {}
    for col in cat_cols:
        rc = real_df[col].value_counts(normalize=True).sort_index()
        sc = synth_df[col].value_counts(normalize=True).sort_index()
        all_cats = list(set(rc.index).union(set(sc.index)))
        rc = rc.reindex(all_cats, fill_value=0)
        sc = sc.reindex(all_cats, fill_value=0)
        jsd_stats[col] = float(jensenshannon(rc, sc))
    results['js_divergence'] = jsd_stats
    
    return results
''')

# utility.py
with open(os.path.join(base_dir, "evaluation", "utility.py"), "w") as f:
    f.write('''import pandas as pd
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
    num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
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
''')

# privacy.py
with open(os.path.join(base_dir, "evaluation", "privacy.py"), "w") as f:
    f.write('''import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def evaluate_privacy_proximity(
    real_df: pd.DataFrame, synth_df: pd.DataFrame, 
    num_cols: list, cat_cols: list
) -> dict:
    """
    Evaluates empirical privacy risk using Nearest Neighbor distance.
    This is an empirical proximity heuristic and DOES NOT guarantee Differential Privacy.
    """
    num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', cat_transformer, cat_cols)
        ])
    
    X_real = preprocessor.fit_transform(real_df)
    X_synth = preprocessor.transform(synth_df)
    
    nn = NearestNeighbors(n_neighbors=1, metric='euclidean', n_jobs=-1).fit(X_real)
    distances = nn.kneighbors(X_synth)[0].flatten()
    
    return {
        "syn_to_real_mean_distance": float(np.mean(distances)),
        "syn_to_real_median_distance": float(np.median(distances)),
        "syn_to_real_min_distance": float(np.min(distances)),
        "syn_to_real_p5_distance": float(np.percentile(distances, 5))
    }
''')

# data_utils.py
with open(os.path.join(base_dir, "utils", "data_utils.py"), "w") as f:
    f.write('''import pandas as pd

def structural_validation(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    return {
        "row_count_match": len(real_df) == len(synth_df),
        "column_count_match": len(real_df.columns) == len(synth_df.columns),
        "columns_match": list(real_df.columns) == list(synth_df.columns),
        "missing_values_synth": int(synth_df.isna().sum().sum()),
        "duplicate_rows_synth": int(synth_df.duplicated().sum()),
        "valid": len(real_df.columns) == len(synth_df.columns) and list(real_df.columns) == list(synth_df.columns)
    }
''')

# pipeline.py
with open(os.path.join(base_dir, "pipeline", "pipeline.py"), "w") as f:
    f.write('''import pandas as pd
from privsynth.profiling.profiler import profile_dataset
from privsynth.generators.ctgan_generator import generate_ctgan
from privsynth.generators.gaussian_copula_generator import generate_gaussian_copula
from privsynth.utils.data_utils import structural_validation

def run_pipeline(input_data: pd.DataFrame, generator="ctgan", config=None) -> dict:
    print("Profiling dataset...")
    profile = profile_dataset(input_data)
    
    print(f"Generating synthetic data using {generator}...")
    if generator == "ctgan":
        synth_data = generate_ctgan(input_data, config=config)
    elif generator == "gaussian_copula":
        synth_data = generate_gaussian_copula(input_data)
    else:
        raise ValueError(f"Unknown generator: {generator}")
        
    print("Validating output...")
    validation = structural_validation(input_data, synth_data)
    
    return {
        "profile": profile,
        "synthetic_data": synth_data,
        "validation": validation
    }
''')
