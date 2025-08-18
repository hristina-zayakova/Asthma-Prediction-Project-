import sys

sys.path.append('../src')
import pandas as pd


def test_respiratory_score():
    """Test respiratory score works."""
    from feature_engineering import create_respiratory_score

    df = pd.DataFrame({'Wheezing': [1, 0], 'Coughing': [1, 1]})
    score = create_respiratory_score(df)
    assert len(score) == 2


def test_allergy_score():
    """Test allergy score works."""
    from feature_engineering import create_allergy_score

    df = pd.DataFrame({'PetAllergy': [1, 0], 'Eczema': [0, 1]})
    score = create_allergy_score(df)
    assert len(score) == 2