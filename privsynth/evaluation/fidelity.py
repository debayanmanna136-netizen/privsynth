import pandas as pd
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
        r_col = real_df[col].dropna()
        s_col = synth_df[col].dropna()
        if len(r_col) > 0 and len(s_col) > 0:
            stat, _ = ks_2samp(r_col, s_col)
            ks_stats[col] = float(stat)
        else:
            ks_stats[col] = None
    results['ks_statistic'] = ks_stats
    
    jsd_stats = {}
    for col in cat_cols:
        rc = real_df[col].dropna().value_counts(normalize=True).sort_index()
        sc = synth_df[col].dropna().value_counts(normalize=True).sort_index()
        all_cats = list(set(rc.index).union(set(sc.index)))
        if all_cats:
            rc = rc.reindex(all_cats, fill_value=0)
            sc = sc.reindex(all_cats, fill_value=0)
            jsd_stats[col] = float(jensenshannon(rc, sc))
        else:
            jsd_stats[col] = None
    results['js_divergence'] = jsd_stats
    
    return results
