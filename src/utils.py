"""
Utility Functions for Asthma Prediction Project

This module contains helper functions for data validation, file operations,
and other common tasks used throughout the project.
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame,
                       required_columns: List[str] = None,
                       min_rows: int = 100) -> bool:
    """
    Validate a DataFrame meets basic requirements.

    Args:
        df: DataFrame to validate
        required_columns: List of columns that must be present
        min_rows: Minimum number of rows required

    Returns:
        True if valid, raises ValueError if not
    """
    if df.empty:
        raise ValueError("DataFrame is empty")

    if len(df) < min_rows:
        raise ValueError(f"DataFrame has {len(df)} rows, minimum {min_rows} required")

    if required_columns:
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    logger.info(f"DataFrame validation passed: {df.shape[0]} rows, {df.shape[1]} columns")
    return True


def save_json(data: Dict[str, Any], filepath: str):
    """
    Save dictionary to JSON file with proper serialization.

    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert numpy types to Python native types
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return obj

    # Recursively convert all values
    def recursive_convert(data):
        if isinstance(data, dict):
            return {k: recursive_convert(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [recursive_convert(item) for item in data]
        else:
            return convert_types(data)

    converted_data = recursive_convert(data)

    with open(filepath, 'w') as f:
        json.dump(converted_data, f, indent=2)

    logger.info(f"JSON saved to: {filepath}")


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    logger.info(f"JSON loaded from: {filepath}")
    return data


def get_memory_usage(df: pd.DataFrame) -> Dict[str, str]:
    """
    Get memory usage information for a DataFrame.

    Args:
        df: DataFrame to analyze

    Returns:
        Dictionary with memory usage info
    """
    memory_usage = df.memory_usage(deep=True)
    total_mb = memory_usage.sum() / 1024 ** 2

    usage_info = {
        'total_mb': f"{total_mb:.2f} MB",
        'shape': f"{df.shape[0]} rows × {df.shape[1]} columns",
        'dtypes': df.dtypes.value_counts().to_dict()
    }

    return usage_info


def create_directory_structure(base_path: str):
    """
    Create the standard project directory structure.

    Args:
        base_path: Base project path
    """
    directories = [
        'data/raw',
        'data/interim',
        'data/processed',
        'models',
        'docs',
        'notebooks',
        'src',
        'tests'
    ]

    for directory in directories:
        full_path = os.path.join(base_path, directory)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"Created directory: {full_path}")


def log_data_info(df: pd.DataFrame, stage: str):
    """
    Log comprehensive information about a DataFrame.

    Args:
        df: DataFrame to analyze
        stage: Processing stage name
    """
    logger.info(f"=== DATA INFO: {stage} ===")
    logger.info(f"Shape: {df.shape}")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
    logger.info(f"Missing values: {df.isnull().sum().sum()}")
    logger.info(f"Duplicate rows: {df.duplicated().sum()}")

    # Data types summary
    dtype_counts = df.dtypes.value_counts()
    logger.info(f"Data types: {dict(dtype_counts)}")


# File: src/config.py
"""
Configuration file for Asthma Prediction Project

This module contains all configuration parameters, file paths, and settings
used throughout the project. Modify this file to change project behavior.
"""

import os

# Project Configuration
PROJECT_NAME = "Asthma Diagnosis Prediction"
PROJECT_VERSION = "1.0.0"
RANDOM_SEED = 42

# File Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
INTERIM_DATA_DIR = os.path.join(DATA_DIR, 'interim')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
DOCS_DIR = os.path.join(BASE_DIR, 'docs')

# Data Processing Parameters
TARGET_COLUMN = 'Diagnosis'
CLASS_IMBALANCE_THRESHOLD = 3.0
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Feature Engineering Parameters
RESPIRATORY_SYMPTOMS = [
    'Wheezing', 'ShortnessOfBreath', 'ChestTightness',
    'Coughing', 'NighttimeSymptoms'
]

ALLERGY_FEATURES = [
    'PetAllergy', 'HistoryOfAllergies', 'Eczema', 'HayFever'
]

RISK_FACTOR_WEIGHTS = {
    'FamilyHistoryAsthma': 3.0,
    'Smoking': 2.0,
    'Age': 1.0
}

INTERACTION_PAIRS = [
    ('ExerciseInduced', 'respiratory_score'),
    ('Smoking', 'FamilyHistoryAsthma'),
    ('Age', 'allergy_score')
]

# Model Parameters
MODEL_CONFIGS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': RANDOM_SEED
    },
    'logistic_regression': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'liblinear',
        'random_state': RANDOM_SEED
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': RANDOM_SEED
    }
}

# Data Validation Rules
VALIDATION_RULES = {
    'min_rows': 100,
    'max_missing_percentage': 20,
    'required_columns': [TARGET_COLUMN],
    'numeric_columns_min': 5,
    'categorical_columns_max': 10
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(DOCS_DIR, 'processing.log')
}

# File: src/pipeline.py
"""
Complete Data Processing Pipeline Script

This script demonstrates how to use the AsthmaDataProcessor class to process
data from raw CSV to model-ready datasets. This is the main entry point for
data processing operations.

Usage:
    python src/pipeline.py --input data/raw/asthma_disease_data.csv --output data/processed
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from data_processing import AsthmaDataProcessor
from config import *
import utils


def setup_logging():
    """Setup logging configuration."""
    os.makedirs(DOCS_DIR, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
        handlers=[
            logging.FileHandler(LOGGING_CONFIG['file']),
            logging.StreamHandler(sys.stdout)
        ]
    )


def main():
    """Main pipeline execution function."""
    parser = argparse.ArgumentParser(description='Process asthma data for machine learning')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to raw CSV data file')
    parser.add_argument('--output', type=str, default=PROCESSED_DATA_DIR,
                        help='Output directory for processed data')
    parser.add_argument('--no-smote', action='store_true',
                        help='Skip SMOTE class balancing')
    parser.add_argument('--test-size', type=float, default=TEST_SIZE,
                        help='Test set proportion')
    parser.add_argument('--features', type=str, nargs='*',
                        help='Specific features to use (optional)')

    args = parser.parse_args()

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(f"Starting {PROJECT_NAME} v{PROJECT_VERSION}")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output}")

    try:
        # Initialize processor
        processor = AsthmaDataProcessor(
            target_column=TARGET_COLUMN,
            class_imbalance_threshold=CLASS_IMBALANCE_THRESHOLD
        )

        # Run complete pipeline
        file_paths = processor.process_pipeline(
            raw_data_path=args.input,
            output_dir=args.output,
            feature_selection=args.features,
            apply_smote=not args.no_smote,
            test_size=args.test_size
        )

        # Save processor for future use
        processor_path = os.path.join(MODELS_DIR, 'data_processor.pkl')
        processor.save_processor(processor_path)

        # Create summary report
        summary = {
            'project': PROJECT_NAME,
            'version': PROJECT_VERSION,
            'input_file': args.input,
            'output_files': file_paths,
            'features_count': len(processor.feature_columns),
            'feature_names': processor.feature_columns,
            'target_column': processor.target_column,
            'smote_applied': not args.no_smote,
            'test_size': args.test_size
        }

        summary_path = os.path.join(DOCS_DIR, 'pipeline_summary.json')
        utils.save_json(summary, summary_path)

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info(f"Processed {len(processor.feature_columns)} features")
        logger.info(f"Files saved to: {args.output}")
        logger.info(f"Processor saved to: {processor_path}")
        logger.info(f"Summary saved to: {summary_path}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()