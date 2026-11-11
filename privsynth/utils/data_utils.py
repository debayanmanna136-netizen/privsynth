import pandas as pd

def structural_validation(real_df: pd.DataFrame, synth_df: pd.DataFrame) -> dict:
    return {
        "row_count_match": len(real_df) == len(synth_df),
        "column_count_match": len(real_df.columns) == len(synth_df.columns),
        "columns_match": list(real_df.columns) == list(synth_df.columns),
        "missing_values_synth": int(synth_df.isna().sum().sum()),
        "duplicate_rows_synth": int(synth_df.duplicated().sum()),
        "valid": len(real_df.columns) == len(synth_df.columns) and list(real_df.columns) == list(synth_df.columns)
    }
