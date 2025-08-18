"""
Configuration file for Asthma Prediction Project
Enhanced version with comprehensive model and evaluation settings
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

ENVIRONMENTAL_FEATURES = [
    'DustMiteAllergy', 'PollenAllergy', 'ExerciseInduced'
]

RISK_FACTOR_WEIGHTS = {
    'FamilyHistoryAsthma': 3.0,
    'Smoking': 2.0,
    'Age': 1.0
}

# Enhanced Feature Engineering Configuration
FEATURE_ENGINEERING = {
    'respiratory_symptoms': RESPIRATORY_SYMPTOMS,
    'allergy_features': ALLERGY_FEATURES,
    'environmental_features': ENVIRONMENTAL_FEATURES,
    'risk_factor_weights': RISK_FACTOR_WEIGHTS,
    'composite_features': [
        'respiratory_score',
        'allergy_score',
        'environmental_score',
        'risk_score'
    ],
    'smote_sampling_strategy': 'auto',
    'smote_random_state': RANDOM_SEED,
    'scaler_type': 'StandardScaler',
    'scaling_features': ['Age', 'BMI'],  # Features that need scaling
    'categorical_features': ['Gender', 'Ethnicity'],  # Categorical features
    'binary_features': RESPIRATORY_SYMPTOMS + ALLERGY_FEATURES + ENVIRONMENTAL_FEATURES
}

# Enhanced Model Parameters
MODEL_CONFIGS = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'n_jobs': -1,
        'random_state': RANDOM_SEED
    },
    'logistic_regression': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'liblinear',
        'class_weight': 'balanced',
        'max_iter': 1000,
        'random_state': RANDOM_SEED
    },
    'xgboost': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'random_state': RANDOM_SEED,
        'use_label_encoder': False
        # Note: scale_pos_weight will be calculated dynamically based on class distribution
    }
}

# Cross-Validation Parameters
CV_CONFIG = {
    'n_splits': 5,
    'shuffle': True,
    'random_state': RANDOM_SEED,
    'scoring_metrics': ['roc_auc', 'f1', 'precision', 'recall']
}

# Model Evaluation Parameters
EVALUATION_CONFIG = {
    'classification_target_names': ['No Asthma', 'Has Asthma'],
    'positive_class_label': 'Has Asthma',
    'top_features_to_show': 10,
    'roc_curve_figsize': (10, 8),
    'confusion_matrix_figsize': (15, 4),
    'feature_importance_figsize': (12, 8),
    'feature_importance_decimals': 4,
    'performance_decimals': 4,
    'probability_threshold': 0.5
}

# File Naming Conventions
FILE_PATTERNS = {
    'model_files': {
        'random_forest': 'random_forest_corrected.pkl',
        'xgboost': 'xgboost_corrected.pkl',
        'logistic_regression': 'logistic_regression_corrected.pkl'
    },
    'results_files': {
        'model_results': 'model_results_corrected.json',
        'feature_summary': 'feature_engineering_summary.json',
        'cross_validation_results': 'cv_results.json',
        'feature_importance': 'feature_importance_comparison.json'
    },
    'training_data': {
        'X_train': 'X_train.csv',
        'X_test': 'X_test.csv',
        'y_train': 'y_train.csv',
        'y_test': 'y_test.csv'
    },
    'plots': {
        'roc_curves': 'roc_curves_comparison.png',
        'confusion_matrices': 'confusion_matrices.png',
        'feature_importance': 'feature_importance_comparison.png'
    }
}

# Data Validation Rules
VALIDATION_RULES = {
    'min_rows': 100,
    'max_missing_percentage': 20,
    'required_columns': [TARGET_COLUMN],
    'numeric_columns_min': 5,
    'categorical_columns_max': 10,
    'target_classes': [0, 1],  # Binary classification
    'min_samples_per_class': 10
}

# Model Performance Thresholds
PERFORMANCE_THRESHOLDS = {
    'minimum_auc': 0.70,
    'minimum_precision': 0.65,
    'minimum_recall': 0.65,
    'minimum_f1': 0.65,
    'max_overfitting_gap': 0.15  # Max difference between train and test AUC
}

# Feature Selection Parameters
FEATURE_SELECTION = {
    'max_features': 50,
    'correlation_threshold': 0.95,  # For removing highly correlated features
    'variance_threshold': 0.01,     # For removing low-variance features
    'feature_importance_threshold': 0.001,  # Minimum importance to keep feature
    'statistical_test': 'chi2',     # For categorical features
    'statistical_k_best': 20        # Top k features from statistical tests
}

# Hyperparameter Tuning (for future use)
HYPERPARAMETER_TUNING = {
    'random_forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'logistic_regression': {
        'C': [0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    'xgboost': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0]
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': os.path.join(DOCS_DIR, 'processing.log'),
    'console_output': True,
    'file_output': True
}

# Visualization Settings
PLOT_CONFIG = {
    'style': 'seaborn-v0_8',
    'figure_dpi': 300,
    'save_format': 'png',
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'font_size': 12,
    'title_size': 14,
    'label_size': 10
}

# Production Deployment Settings
DEPLOYMENT_CONFIG = {
    'model_selection_criterion': 'test_auc',  # Primary metric for model selection
    'ensemble_models': ['random_forest', 'xgboost'],  # Models to potentially ensemble
    'prediction_threshold': 0.5,
    'confidence_threshold': 0.7,  # Minimum confidence for high-confidence predictions
    'batch_prediction_size': 1000,
    'model_monitoring_metrics': ['auc', 'precision', 'recall', 'f1']
}

# Data Quality Checks
DATA_QUALITY_CHECKS = {
    'check_missing_values': True,
    'check_duplicates': True,
    'check_outliers': True,
    'check_data_types': True,
    'check_target_distribution': True,
    'outlier_method': 'iqr',  # 'iqr' or 'zscore'
    'outlier_threshold': 3.0
}

# Medical Domain Knowledge
MEDICAL_KNOWLEDGE = {
    'high_risk_age_ranges': [(0, 5), (65, 100)],  # Age ranges with higher asthma risk
    'critical_symptoms': ['ShortnessOfBreath', 'ChestTightness', 'Wheezing'],
    'family_history_weight': 2.5,  # Multiplier for family history importance
    'environmental_triggers': ['DustMiteAllergy', 'PollenAllergy', 'PetAllergy'],
    'lifestyle_factors': ['Smoking', 'BMI', 'ExerciseInduced']
}

# Export commonly used configurations for easy importing
__all__ = [
    'PROJECT_NAME', 'PROJECT_VERSION', 'RANDOM_SEED',
    'DATA_DIR', 'PROCESSED_DATA_DIR', 'MODELS_DIR', 'DOCS_DIR',
    'TARGET_COLUMN', 'TEST_SIZE',
    'MODEL_CONFIGS', 'CV_CONFIG', 'EVALUATION_CONFIG',
    'FILE_PATTERNS', 'FEATURE_ENGINEERING',
    'PERFORMANCE_THRESHOLDS', 'MEDICAL_KNOWLEDGE'
]