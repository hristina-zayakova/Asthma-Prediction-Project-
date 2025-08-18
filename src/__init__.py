# File: src/__init__.py
"""
Asthma Prediction Data Processing Package

This package provides a complete data processing pipeline for asthma diagnosis
prediction, including data cleaning, feature engineering, and preprocessing
for machine learning models.

Modules:
- data_processing: Main processing classes and functions
- feature_engineering: Specialized feature creation utilities
- utils: Helper functions and utilities
- config: Configuration parameters and settings
- pipeline: Command-line pipeline execution script

Example usage:
    from src.data_processing import AsthmaDataProcessor

    processor = AsthmaDataProcessor()
    file_paths = processor.process_pipeline('data/raw/asthma_data.csv')
"""

from .data_processing import AsthmaDataProcessor
from .feature_engineering import (
    create_respiratory_score,
    create_allergy_score,
    create_risk_score,
    create_interaction_features,
    select_top_features
)
from .utils import (
    validate_dataframe,
    save_json,
    load_json,
    get_memory_usage,
    create_directory_structure,
    log_data_info
)
from .config import *

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    'AsthmaDataProcessor',
    'create_respiratory_score',
    'create_allergy_score',
    'create_risk_score',
    'create_interaction_features',
    'select_top_features',
    'validate_dataframe',
    'save_json',
    'load_json',
    'get_memory_usage',
    'create_directory_structure',
    'log_data_info'
]