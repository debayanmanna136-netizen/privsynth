import pandas as pd
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
