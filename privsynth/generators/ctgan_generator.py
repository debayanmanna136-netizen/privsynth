import pandas as pd
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
