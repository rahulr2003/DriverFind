import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer


class DataPreprocessor:
    """
    1. Multi-format data loading (.csv, .xlsx, or DataFrame)
    2. Custom missing value markers
    3. Categorical-to-numeric encoding (Label Encoding preserving NaNs)
    4. Flexible imputation methods (mean, median, mode, knn, or None)
    5. Feature scaling and target type auto-detection
    """
    def __init__(
        self,
        target_col: str,
        id_cols: Optional[List[str]] = None, # list of any strings
        custom_missing_values: Optional[List[Any]] = None, # any list of alpha-numeric characters
        imputation_method: Optional[str] = "median",  # "mean", "median", "mode", "knn", or None
        knn_neighbors: int = 5, # any integer < number of samples
        encode_categorical: bool = True,
        scale_features: bool = True
    ):
        self.target_col = target_col
        self.id_cols = id_cols or []
        self.custom_missing_values = custom_missing_values or ["?", "N/A", "n/a", "null", "NULL", " ", ""]
        self.imputation_method = imputation_method.lower() if imputation_method else None
        self.knn_neighbors = knn_neighbors
        self.encode_categorical = encode_categorical
        self.scale_features = scale_features
        
        # State tracking
        self.target_type: Optional[str] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.imputer = None
        self.scaler = StandardScaler() if scale_features else None

    def load_data(self, data_input: Union[str, pd.DataFrame]) -> pd.DataFrame:
        """Loads dataset from CSV, Excel file path, or Pandas DataFrame."""
        if isinstance(data_input, pd.DataFrame):
            df = data_input.copy()
        elif isinstance(data_input, str):
            if not os.path.exists(data_input):
                raise FileNotFoundError(f"File path does not exist: {data_input}")
            
            ext = os.path.splitext(data_input)[-1].lower()
            if ext == ".csv":
                df = pd.read_csv(data_input, na_values=self.custom_missing_values)
            elif ext in [".xlsx", ".xls"]:
                df = pd.read_excel(data_input, na_values=self.custom_missing_values)
            else:
                raise ValueError(f"Unsupported file format '{ext}'. Supported: .csv, .xlsx, .xls")
        else:
            raise TypeError("Input data must be a file path string (.csv, .xlsx) or a pandas DataFrame.")
        
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts non-numeric string/categorical columns to numeric using LabelEncoding while preserving NaNs."""
        df_encoded = df.copy()
        
        for col in df_encoded.columns:
            if col == self.target_col:
                continue
                
            if df_encoded[col].dtype == 'object' or isinstance(df_encoded[col].dtype, pd.CategoricalDtype):
                # Mask NaNs so LabelEncoder doesn't fail on them
                non_null_mask = df_encoded[col].notna()
                if non_null_mask.any():
                    le = LabelEncoder()
                    df_encoded.loc[non_null_mask, col] = le.fit_transform(df_encoded[col][non_null_mask].astype(str))
                    self.label_encoders[col] = le
                
                df_encoded[col] = pd.to_numeric(df_encoded[col], errors='coerce')
                
        return df_encoded

    def _build_imputer(self):
        """Instantiates requested imputer engine."""
        if self.imputation_method == "mean":
            return SimpleImputer(strategy="mean")
        elif self.imputation_method == "median":
            return SimpleImputer(strategy="median")
        elif self.imputation_method in ["mode", "most_frequent"]:
            return SimpleImputer(strategy="most_frequent")
        elif self.imputation_method == "knn":
            return KNNImputer(n_neighbors=self.knn_neighbors)
        elif self.imputation_method is None or self.imputation_method == "none":
            return None
        else:
            raise ValueError(
                f"Invalid imputation method '{self.imputation_method}'. "
                "Choose from: 'mean', 'median', 'mode', 'knn', or None."
            )

    def fit_transform(self, data_input: Union[str, pd.DataFrame]) -> Tuple[pd.DataFrame, str]:
        """
        Executes full preprocessing pipeline.
        Returns:
            - Processed Feature Matrix + Target (DataFrame)
            - Detected Target Type ('continuous' or 'categorical')
        """
        # 1. Load Data
        df = self.load_data(data_input)

        # 2. Map custom missing markers to NaN
        df = df.replace(self.custom_missing_values, np.nan)

        # 3. Validate Target Column
        if self.target_col not in df.columns:
            raise KeyError(f"Target column '{self.target_col}' not found in input data.")

        # 4. Drop ID Columns
        drop_list = [col for col in self.id_cols if col in df.columns]
        if drop_list:
            df = df.drop(columns=drop_list)

        # 5. Detect Target Type
        target_series = df[self.target_col].dropna()
        is_numeric = pd.api.types.is_numeric_dtype(target_series)
        unique_count = target_series.nunique()
        self.target_type = "categorical" if (not is_numeric or unique_count <= 10) else "continuous"

        # 6. Separate X and y
        y = df[self.target_col].copy()
        X = df.drop(columns=[self.target_col])

        # 7. Convert Non-Numeric Categorical Features
        if self.encode_categorical:
            X = self._encode_categoricals(X)

        # Ensure all feature columns are cast to float for numeric processing
        feature_names = X.columns.tolist()
        X_mat = X.values.astype(np.float64)

        # 8. Missing Value Imputation
        self.imputer = self._build_imputer()
        if self.imputer is not None and np.isnan(X_mat).any():
            X_mat = self.imputer.fit_transform(X_mat)

        # 9. Optional Feature Scaling
        if self.scale_features and self.imputer is not None:
            # Note: Scaling requires imputed matrix (no NaNs)
            X_mat = self.scaler.fit_transform(X_mat)

        # Reconstruct DataFrame
        X_processed = pd.DataFrame(X_mat, columns=feature_names, index=df.index)
        df_final = pd.concat([X_processed, y.rename(self.target_col)], axis=1)

        return df_final, self.target_type