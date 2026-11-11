import pandas as pd
import numpy as np

def generate_gaussian_copula(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic tabular data using Gaussian Copula.
    
    Uses lightweight standalone copulas and rdt (avoiding PyTorch/CTGAN imports),
    with automatic fallback to SDV GaussianCopulaSynthesizer if available.
    """
    np.random.seed(seed)
    try:
        from copulas.multivariate import GaussianMultivariate
        from copulas.univariate import BetaUnivariate
        from rdt import HyperTransformer

        ht = HyperTransformer()
        ht.detect_initial_config(df)
        transformed = ht.fit_transform(df)

        model = GaussianMultivariate(distribution=BetaUnivariate)
        model.fit(transformed)
        sampled = model.sample(len(df))

        synth_df = ht.reverse_transform(sampled)
        return synth_df[df.columns]
    except Exception:
        # Fallback to SDV synthesizer if standalone copulas/rdt pipeline raises
        from sdv.metadata import SingleTableMetadata
        from sdv.single_table import GaussianCopulaSynthesizer

        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)
        synthesizer = GaussianCopulaSynthesizer(metadata, enforce_rounding=True)
        synthesizer.fit(df)
        return synthesizer.sample(num_rows=len(df))
