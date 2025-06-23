from ctgan import CTGAN
import pandas as pd
from scipy.stats import ks_2samp
import numpy as np

def generate_synthetic_patients(df, target_column=None, n_samples=100, discrete_columns=None, model=None, return_model=False, evaluate_quality=True):
    """
    Generate synthetic patient data using CTGAN, with optional quality assessment via KS test.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_column (str, optional): Column to exclude from generation.
        n_samples (int): Number of synthetic samples to generate.
        discrete_columns (list, optional): List of discrete column names.
        model (CTGAN, optional): Pre-trained CTGAN model.
        return_model (bool): Whether to return the model.
        evaluate_quality (bool): If True, computes KS statistics between real and synthetic data.

    Returns:
        dict: {
            'synthetic': pd.DataFrame,
            'model': CTGAN or None,
            'quality': dict with KS statistics per column and global score (if evaluate_quality is True)
        }
    """

    if not isinstance(df, pd.DataFrame):
        raise ValueError("`df` must be a pandas DataFrame.")

    X = df.copy()
    if target_column is not None:
        X = X.drop(columns=[target_column], errors='ignore')

    if discrete_columns is None:
        discrete_columns = X.select_dtypes(include='object').columns.tolist()

    if model is None:
        model = CTGAN(epochs=600)
        model.fit(X, discrete_columns)

    synthetic_df = model.sample(n_samples)

    quality_report = {}

    if evaluate_quality:
        ks_results = {}
        for col in X.columns:
            if col in discrete_columns:
                continue
            try:
                stat, _ = ks_2samp(X[col].dropna(), synthetic_df[col].dropna())
                ks_results[col] = round(stat, 4)
            except:
                ks_results[col] = None
        # Global reliability score: mean KS distance (lower = better)
        valid_scores = [v for v in ks_results.values() if v is not None]
        mean_ks = np.mean(valid_scores) if valid_scores else None
        quality_report = {"KS per column": ks_results, "Global KS score": round(mean_ks, 4) if mean_ks else None}

    return {
        "synthetic": synthetic_df,
        "model": model if return_model else None,
        "quality": quality_report if evaluate_quality else None
    }