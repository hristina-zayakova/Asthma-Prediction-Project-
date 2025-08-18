import sys

sys.path.append('../src')
from data_processing import AsthmaDataProcessor


def test_processor_init():
    """Test processor can be created."""
    processor = AsthmaDataProcessor()
    assert processor.target_column == 'Diagnosis'


def test_basic_pipeline():
    """Test pipeline doesn't crash with tiny data."""
    import pandas as pd

    # Tiny fake data
    df = pd.DataFrame({
        'Age': [25, 30], 'Diagnosis': [0, 1], 'Wheezing': [0, 1]
    })
    df.to_csv('tiny_test.csv', index=False)

    processor = AsthmaDataProcessor()
    df_loaded = processor.load_raw_data('tiny_test.csv')
    assert len(df_loaded) == 2

    import os
    os.remove('tiny_test.csv')