"""
Custom preprocessing steps for the SECOM fail-detection project.

They live in this .py file (not inside the notebook) so that the saved
joblib pipeline can be loaded later by the Streamlit app.
Every step learns ONLY from the training data in fit().
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class HighMissingDropper(BaseEstimator, TransformerMixin):
    """Drop sensors whose missing rate in the TRAINING data is above `threshold`."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    def fit(self, X, y=None):
        missing_rate = X.isna().mean()
        self.columns_out_ = list(missing_rate.index[missing_rate <= self.threshold])
        return self

    def transform(self, X):
        return X[self.columns_out_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_out_, dtype=object)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each sensor to its training [lower, upper] quantiles (outlier control)."""

    def __init__(self, lower=0.005, upper=0.995):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        values = X.to_numpy(dtype=float)
        self.lo_ = np.nanquantile(values, self.lower, axis=0)
        self.hi_ = np.nanquantile(values, self.upper, axis=0)
        self.columns_out_ = list(X.columns)
        return self

    def transform(self, X):
        clipped = np.clip(X.to_numpy(dtype=float), self.lo_, self.hi_)
        return pd.DataFrame(clipped, columns=self.columns_out_, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_out_, dtype=object)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop a sensor if it has |corr| > threshold with an earlier sensor (training data)."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = np.abs(np.corrcoef(X.to_numpy(dtype=float), rowvar=False))
        upper = np.triu(corr, k=1)
        drop = upper.max(axis=0) > self.threshold
        self.columns_out_ = [c for c, d in zip(X.columns, drop) if not d]
        return self

    def transform(self, X):
        return X[self.columns_out_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.columns_out_, dtype=object)
    

# ---------------------------------------------------------------------------
# Used by the Streamlit app
# ---------------------------------------------------------------------------
N_SENSORS = 590
SENSOR_COLS = [f"sensor_{i:03d}" for i in range(1, N_SENSORS + 1)]


def validate_upload(df):
    """Check an uploaded table of production runs and return (X, warnings).

    Raises ValueError with a readable message when the file cannot be used.
    """
    warnings = []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # A raw file with 590 unnamed columns (same layout as secom.data) → name them by position
    if not set(SENSOR_COLS) & set(df.columns):
        other = [c for c in df.columns if c not in ("run_id", "label", "timestamp")]
        if len(other) == N_SENSORS:
            df = df.rename(columns=dict(zip(other, SENSOR_COLS)))
            warnings.append("No sensor names found — columns were mapped by position.")
        else:
            raise ValueError(f"Expected columns sensor_001 … sensor_590 (or exactly {N_SENSORS} "
                             f"unnamed numeric columns); found {df.shape[1]} columns.")

    if len(df) == 0:
        raise ValueError("The file has no rows.")

    missing = [c for c in SENSOR_COLS if c not in df.columns]
    if len(missing) > N_SENSORS // 2:
        raise ValueError(f"{len(missing)} of {N_SENSORS} sensor columns are missing.")
    if missing:
        warnings.append(f"{len(missing)} sensor columns were missing and will be filled with typical values.")

    raw = df.reindex(columns=SENSOR_COLS)
    X = raw.apply(pd.to_numeric, errors="coerce")
    bad = int(X.isna().sum().sum() - raw.isna().sum().sum())
    if bad:
        warnings.append(f"{bad} non-numeric values were treated as missing.")
    return X, warnings