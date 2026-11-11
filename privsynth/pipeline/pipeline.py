import pandas as pd
from privsynth.profiling.profiler import profile_dataset
from privsynth.generators.gaussian_copula_generator import generate_gaussian_copula
from privsynth.utils.data_utils import structural_validation

def run_pipeline(input_data: pd.DataFrame, generator="ctgan", config=None) -> dict:
    print("Profiling dataset...")
    profile = profile_dataset(input_data)
    
    print(f"Generating synthetic data using {generator}...")
    if generator == "ctgan":
        try:
            from privsynth.generators.ctgan_generator import generate_ctgan
        except ImportError as e:
            raise RuntimeError(
                "CTGAN generator is not available in this environment because PyTorch/CTGAN dependencies are not installed. "
                "Please use 'gaussian_copula' or install the full requirements.txt."
            ) from e
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
