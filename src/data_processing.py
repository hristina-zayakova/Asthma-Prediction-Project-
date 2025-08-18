"""
Data Processing Module for Asthma Prediction Project

This module contains all data cleaning, preprocessing, and feature engineering
functions extracted from our exploratory notebooks. It provides a complete
pipeline for transforming raw asthma data into model-ready features.

Author: Your Name
Project: Asthma Diagnosis Prediction
Created: [Date]
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import logging
from typing import Tuple, List, Dict, Optional, Union
import warnings

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AsthmaDataProcessor:
    """
    A comprehensive data processing pipeline for asthma prediction.

    This class handles the complete data transformation pipeline from raw data
    to model-ready features, including cleaning, feature engineering, encoding,
    scaling, and class balancing.

    Attributes:
        scaler (StandardScaler): Fitted scaler for numeric features
        target_column (str): Name of the target variable
        feature_columns (List[str]): List of final feature column names
        class_imbalance_threshold (float): Threshold for applying SMOTE
    """

    def __init__(self, target_column: str = 'Diagnosis',
                 class_imbalance_threshold: float = 3.0):
        """
        Initialize the data processor.

        Args:
            target_column: Name of the target variable column
            class_imbalance_threshold: Ratio threshold for applying SMOTE
        """
        self.target_column = target_column
        self.class_imbalance_threshold = class_imbalance_threshold
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False

        logger.info(f"Initialized AsthmaDataProcessor with target: {target_column}")

    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """
        Load raw data from CSV file with validation.

        Args:
            file_path: Path to the CSV file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If required columns are missing
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

            # Validate required columns
            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in data")

            return df

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the raw data by handling missing values, outliers, and data types.

        Args:
            df: Raw DataFrame

        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        logger.info("Starting data cleaning...")

        # Handle missing values
        initial_missing = df_clean.isnull().sum().sum()
        if initial_missing > 0:
            logger.info(f"Found {initial_missing} missing values")
            df_clean = self._handle_missing_values(df_clean)

        # Fix data types
        df_clean = self._fix_data_types(df_clean)

        # Remove constant columns
        df_clean = self._remove_constant_columns(df_clean)

        # Handle outliers (conservative approach)
        df_clean = self._handle_outliers(df_clean)

        logger.info(f"Data cleaning complete: {df_clean.shape[0]} rows, {df_clean.shape[1]} columns")
        return df_clean

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate strategies."""
        df_filled = df.copy()

        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if df[col].dtype in ['int64', 'float64']:
                    # Numeric: use median
                    fill_value = df[col].median()
                    df_filled[col].fillna(fill_value, inplace=True)
                    logger.info(f"Filled {col} missing values with median: {fill_value}")
                else:
                    # Categorical: use mode
                    fill_value = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
                    df_filled[col].fillna(fill_value, inplace=True)
                    logger.info(f"Filled {col} missing values with mode: {fill_value}")

        return df_filled

    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types for optimal processing."""
        df_fixed = df.copy()

        # Convert object columns that should be numeric
        for col in df_fixed.select_dtypes(include=['object']).columns:
            if col != self.target_column:
                try:
                    # Try to convert to numeric
                    pd.to_numeric(df_fixed[col], errors='raise')
                    df_fixed[col] = pd.to_numeric(df_fixed[col])
                    logger.info(f"Converted {col} to numeric")
                except:
                    # Keep as categorical
                    pass

        return df_fixed

    def _remove_constant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove columns with only one unique value."""
        df_clean = df.copy()
        constant_cols = []

        for col in df.columns:
            if col != self.target_column and df[col].nunique() == 1:
                constant_cols.append(col)

        if constant_cols:
            df_clean = df_clean.drop(columns=constant_cols)
            logger.info(f"Removed constant columns: {constant_cols}")

        return df_clean

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers using IQR method (conservative - cap rather than remove).
        """
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col != self.target_column:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap outliers instead of removing them
                outliers_count = len(df_clean[(df_clean[col] < lower_bound) |
                                              (df_clean[col] > upper_bound)])
                if outliers_count > 0:
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                    logger.info(f"Capped {outliers_count} outliers in {col}")

        return df_clean

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features based on domain knowledge and EDA insights.

        Args:
            df: Cleaned DataFrame

        Returns:
            DataFrame with new features
        """
        df_features = df.copy()
        logger.info("Creating engineered features...")

        # Respiratory symptom score
        respiratory_features = ['Wheezing', 'ShortnessOfBreath', 'ChestTightness',
                                'Coughing', 'NighttimeSymptoms']
        existing_respiratory = [col for col in respiratory_features if col in df.columns]

        if existing_respiratory:
            df_features['respiratory_score'] = df_features[existing_respiratory].sum(axis=1)
            logger.info(f"Created respiratory_score from: {existing_respiratory}")

        # Allergy score
        allergy_features = ['PetAllergy', 'HistoryOfAllergies', 'Eczema', 'HayFever']
        existing_allergies = [col for col in allergy_features if col in df.columns]

        if existing_allergies:
            df_features['allergy_score'] = df_features[existing_allergies].sum(axis=1)
            logger.info(f"Created allergy_score from: {existing_allergies}")

        # Risk factor score (weighted)
        risk_weights = {
            'FamilyHistoryAsthma': 3,
            'Smoking': 2,
            'Age': 1
        }

        risk_score = 0
        used_factors = []
        for factor, weight in risk_weights.items():
            if factor in df.columns:
                if factor == 'Age':
                    # Normalize age first
                    age_norm = ((df_features[factor] - df_features[factor].min()) /
                                (df_features[factor].max() - df_features[factor].min()))
                    risk_score += age_norm * weight
                else:
                    risk_score += df_features[factor] * weight
                used_factors.append(factor)

        if len(used_factors) > 0:
            df_features['risk_score'] = risk_score
            logger.info(f"Created risk_score from: {used_factors}")

        # Interaction features
        if 'ExerciseInduced' in df.columns and 'respiratory_score' in df_features.columns:
            df_features['exercise_respiratory_interaction'] = (
                    df_features['ExerciseInduced'] * df_features['respiratory_score']
            )
            logger.info("Created exercise_respiratory_interaction")

        logger.info(
            f"Feature engineering complete. Added {len([c for c in df_features.columns if c not in df.columns])} new features")
        return df_features

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features for machine learning.

        Args:
            df: DataFrame with features to encode

        Returns:
            DataFrame with encoded features
        """
        df_encoded = df.copy()
        logger.info("Encoding categorical features...")

        categorical_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()

        # Remove target column from encoding if it's categorical
        if self.target_column in categorical_cols:
            categorical_cols.remove(self.target_column)

        for col in categorical_cols:
            unique_count = df_encoded[col].nunique()

            if unique_count == 2:
                # Binary encoding
                if not set(df_encoded[col].unique()).issubset({0, 1, '0', '1'}):
                    le = LabelEncoder()
                    df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col])
                    df_encoded = df_encoded.drop(columns=[col])
                    logger.info(f"Binary encoded: {col}")
                else:
                    df_encoded[col] = df_encoded[col].astype(int)

            elif unique_count > 2 and unique_count <= 10:
                # One-hot encoding for reasonable number of categories
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(columns=[col])
                logger.info(f"One-hot encoded: {col} ({len(dummies.columns)} dummies)")

            else:
                # Too many categories - consider grouping or dropping
                logger.warning(f"Column {col} has {unique_count} categories - consider manual handling")

        return df_encoded

    def prepare_features_target(self, df: pd.DataFrame,
                                feature_selection: Optional[List[str]] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features and target, with optional feature selection.

        Args:
            df: Encoded DataFrame
            feature_selection: Optional list of features to keep

        Returns:
            Tuple of (features_df, target_series)
        """
        # Separate target
        y = df[self.target_column].copy()

        # Get feature columns
        feature_cols = [col for col in df.columns if col != self.target_column]

        # Apply feature selection if provided
        if feature_selection:
            available_features = [col for col in feature_selection if col in feature_cols]
            feature_cols = available_features
            logger.info(f"Applied feature selection: {len(feature_cols)} features selected")

        X = df[feature_cols].copy()
        self.feature_columns = list(X.columns)

        logger.info(f"Prepared features: {X.shape[1]} columns, target: {len(y)} samples")
        return X, y

    def scale_features(self, X_train: pd.DataFrame, X_test: Optional[pd.DataFrame] = None) -> Union[
        pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Scale numeric features using StandardScaler.

        Args:
            X_train: Training features
            X_test: Optional test features

        Returns:
            Scaled features (train only or train+test)
        """
        numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_features:
            logger.info("No numeric features to scale")
            return (X_train, X_test) if X_test is not None else X_train

        X_train_scaled = X_train.copy()

        # Fit and transform training data
        X_train_scaled[numeric_features] = self.scaler.fit_transform(X_train[numeric_features])
        self.is_fitted = True

        logger.info(f"Scaled {len(numeric_features)} numeric features")

        if X_test is not None:
            X_test_scaled = X_test.copy()
            X_test_scaled[numeric_features] = self.scaler.transform(X_test[numeric_features])
            return X_train_scaled, X_test_scaled

        return X_train_scaled

    def balance_classes(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance classes using SMOTE if imbalance is significant.

        Args:
            X: Features
            y: Target

        Returns:
            Balanced features and target
        """
        class_counts = y.value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()

        logger.info(f"Class distribution: {dict(class_counts)}")
        logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

        if imbalance_ratio > self.class_imbalance_threshold:
            logger.info("Applying SMOTE for class balancing...")

            # Ensure we have enough samples for SMOTE
            min_samples = class_counts.min()
            k_neighbors = min(5, min_samples - 1)

            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_balanced, y_balanced = smote.fit_resample(X, y)

            new_counts = pd.Series(y_balanced).value_counts()
            logger.info(f"Balanced distribution: {dict(new_counts)}")

            return pd.DataFrame(X_balanced, columns=X.columns), pd.Series(y_balanced)

        else:
            logger.info("Classes reasonably balanced - no SMOTE applied")
            return X, y

    def create_train_test_split(self, X: pd.DataFrame, y: pd.Series,
                                test_size: float = 0.2,
                                random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Create stratified train/test split.

        Args:
            X: Features
            y: Target
            test_size: Proportion for test set
            random_state: Random seed

        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y
        )

        logger.info(f"Train/test split created:")
        logger.info(f"  Training: {X_train.shape[0]} samples")
        logger.info(f"  Test: {X_test.shape[0]} samples")

        return X_train, X_test, y_train, y_test

    def process_pipeline(self, raw_data_path: str,
                         output_dir: str = '../data/processed',
                         feature_selection: Optional[List[str]] = None,
                         apply_smote: bool = True,
                         test_size: float = 0.2) -> Dict[str, str]:
        """
        Complete data processing pipeline from raw data to model-ready datasets.

        Args:
            raw_data_path: Path to raw CSV data
            output_dir: Directory to save processed data
            feature_selection: Optional list of features to keep
            apply_smote: Whether to apply SMOTE for class balancing
            test_size: Test set proportion

        Returns:
            Dictionary with paths to saved files
        """
        logger.info("Starting complete data processing pipeline...")

        # Step 1: Load and clean data
        df_raw = self.load_raw_data(raw_data_path)
        df_clean = self.clean_data(df_raw)

        # Step 2: Feature engineering
        df_features = self.create_features(df_clean)

        # Step 3: Encode categorical features
        df_encoded = self.encode_categorical_features(df_features)

        # Step 4: Prepare features and target
        X, y = self.prepare_features_target(df_encoded, feature_selection)

        # Step 5: Train/test split (before balancing to keep test set realistic)
        X_train, X_test, y_train, y_test = self.create_train_test_split(X, y, test_size)

        # Step 6: Scale features
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        # Step 7: Balance training classes (only training set)
        if apply_smote:
            X_train_balanced, y_train_balanced = self.balance_classes(X_train_scaled, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train_scaled, y_train

        # Step 8: Save processed datasets
        import os
        os.makedirs(output_dir, exist_ok=True)

        file_paths = {}
        file_paths['X_train'] = f"{output_dir}/X_train.csv"
        file_paths['X_test'] = f"{output_dir}/X_test.csv"
        file_paths['y_train'] = f"{output_dir}/y_train.csv"
        file_paths['y_test'] = f"{output_dir}/y_test.csv"

        X_train_balanced.to_csv(file_paths['X_train'], index=False)
        X_test_scaled.to_csv(file_paths['X_test'], index=False)
        pd.Series(y_train_balanced).to_csv(file_paths['y_train'], index=False, header=['target'])
        pd.Series(y_test).to_csv(file_paths['y_test'], index=False, header=['target'])

        # Save scaler
        scaler_path = f"{output_dir.replace('processed', 'models')}/feature_scaler.pkl"
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        joblib.dump(self.scaler, scaler_path)
        file_paths['scaler'] = scaler_path

        logger.info("Pipeline complete! Files saved:")
        for name, path in file_paths.items():
            logger.info(f"  {name}: {path}")

        return file_paths

    def save_processor(self, filepath: str):
        """Save the fitted processor for future use."""
        joblib.dump(self, filepath)
        logger.info(f"Processor saved to: {filepath}")

    @classmethod
    def load_processor(cls, filepath: str):
        """Load a fitted processor."""
        processor = joblib.load(filepath)
        logger.info(f"Processor loaded from: {filepath}")
        return processor