"""
Feature Engineering Utilities for Asthma Prediction

This module contains specialized feature engineering functions that can be
used independently or as part of the main data processing pipeline.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def create_respiratory_score(df: pd.DataFrame,
                             symptoms: List[str] = None) -> pd.Series:
    """
    Create a composite respiratory symptom score.

    Args:
        df: DataFrame containing symptom columns
        symptoms: List of symptom column names

    Returns:
        Series with respiratory scores
    """
    if symptoms is None:
        symptoms = ['Wheezing', 'ShortnessOfBreath', 'ChestTightness',
                    'Coughing', 'NighttimeSymptoms']

    available_symptoms = [col for col in symptoms if col in df.columns]

    if not available_symptoms:
        raise ValueError("No respiratory symptom columns found in DataFrame")

    score = df[available_symptoms].sum(axis=1)
    logger.info(f"Created respiratory score from: {available_symptoms}")

    return score


def create_allergy_score(df: pd.DataFrame,
                         allergies: List[str] = None) -> pd.Series:
    """
    Create a composite allergy score.

    Args:
        df: DataFrame containing allergy columns
        allergies: List of allergy column names

    Returns:
        Series with allergy scores
    """
    if allergies is None:
        allergies = ['PetAllergy', 'HistoryOfAllergies', 'Eczema', 'HayFever']

    available_allergies = [col for col in allergies if col in df.columns]

    if not available_allergies:
        raise ValueError("No allergy columns found in DataFrame")

    score = df[available_allergies].sum(axis=1)
    logger.info(f"Created allergy score from: {available_allergies}")

    return score


def create_risk_score(df: pd.DataFrame,
                      risk_weights: Dict[str, float] = None) -> pd.Series:
    """
    Create a weighted risk factor score.

    Args:
        df: DataFrame containing risk factor columns
        risk_weights: Dictionary mapping column names to weights

    Returns:
        Series with risk scores
    """
    if risk_weights is None:
        risk_weights = {
            'FamilyHistoryAsthma': 3.0,
            'Smoking': 2.0,
            'Age': 1.0
        }

    score = pd.Series(0, index=df.index)
    used_factors = []

    for factor, weight in risk_weights.items():
        if factor in df.columns:
            if factor == 'Age':
                # Normalize continuous variables
                normalized = (df[factor] - df[factor].min()) / (df[factor].max() - df[factor].min())
                score += normalized * weight
            else:
                score += df[factor] * weight
            used_factors.append(factor)

    if not used_factors:
        raise ValueError("No risk factor columns found in DataFrame")

    logger.info(f"Created risk score from: {used_factors}")
    return score


def create_interaction_features(df: pd.DataFrame,
                                interactions: List[Tuple[str, str]] = None) -> pd.DataFrame:
    """
    Create interaction features between specified columns.

    Args:
        df: DataFrame containing base features
        interactions: List of tuples specifying column pairs

    Returns:
        DataFrame with interaction features added
    """
    if interactions is None:
        interactions = [
            ('ExerciseInduced', 'respiratory_score'),
            ('Smoking', 'FamilyHistoryAsthma'),
            ('Age', 'allergy_score')
        ]

    df_interactions = df.copy()

    for col1, col2 in interactions:
        if col1 in df.columns and col2 in df.columns:
            interaction_name = f"{col1}_{col2}_interaction"
            df_interactions[interaction_name] = df[col1] * df[col2]
            logger.info(f"Created interaction: {interaction_name}")

    return df_interactions


def select_top_features(X: pd.DataFrame, y: pd.Series,
                        method: str = 'correlation',
                        n_features: int = 20) -> List[str]:
    """
    Select top features using specified method.

    Args:
        X: Features DataFrame
        y: Target Series
        method: Selection method ('correlation', 'variance', 'mutual_info')
        n_features: Number of features to select

    Returns:
        List of selected feature names
    """
    if method == 'correlation':
        # Correlation-based selection
        correlations = []
        for col in X.columns:
            if X[col].dtype in [np.number]:
                corr = abs(np.corrcoef(X[col], y)[0, 1])
                correlations.append((col, corr))

        correlations.sort(key=lambda x: x[1], reverse=True)
        selected = [col for col, _ in correlations[:n_features]]

    elif method == 'variance':
        # Variance-based selection
        variances = X.var().sort_values(ascending=False)
        selected = variances.head(n_features).index.tolist()

    else:
        raise ValueError(f"Unknown selection method: {method}")

    logger.info(f"Selected {len(selected)} features using {method} method")
    return selected
