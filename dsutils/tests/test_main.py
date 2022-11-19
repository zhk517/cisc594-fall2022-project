import numpy as np
import pandas as pd

import pytest
from unittest.mock import patch

from dsutils.main import _load_data, null_counter, train_test_splitter


def test__load_data_raises_if_file_not_exists():
    with pytest.raises(ValueError) as e:
        _load_data('non_exist_data.csv')
    assert str(e.value).split('\n')[0] == 'Failed to load data from path: non_exist_data.csv'


mocked_data = pd.DataFrame({'Feature 1': [100, np.nan, 50, np.nan],
                            'Feature 2': [np.nan] * 4,
                            'Response': [_ for _ in range(4)]})


@patch('dsutils.main._load_data', return_value=mocked_data)
def test_train_test_splitter_split_data_with_given_ratio(mocked_load_data):
    df_train, df_test = train_test_splitter('', ratio=0.75, save_to=False)
    assert len(df_train) == 3 and len(df_test) == 1


@patch('dsutils.main._load_data', return_value=mocked_data)
@pytest.mark.parametrize('r', [-0.5, 0, 1.0, 1.0001])
def test_train_test_splitter_raises_if_invalid_ratio(mocked_load_data, r):
    with pytest.raises(ValueError) as e:
        train_test_splitter('', ratio=r, save_to=False)
    assert str(e.value) == 'Split ratio must be within range (0, 1)'


@patch('dsutils.main._load_data', return_value=pd.DataFrame())
def test_train_test_splitter_raises_if_empty_df(mocked_load_data):
    with pytest.raises(ValueError) as e:
        train_test_splitter('', save_to=False)
    assert str(e.value) == 'Cannot split an empty dataframe'


@patch('dsutils.main._load_data', return_value=mocked_data)
def test_null_counter_counts_correctly(mocked_load_data):
    df_nulls = null_counter('')
    expected_df = pd.DataFrame({'Feature 1': 2,
                                'Feature 2': 4,
                                'Response': 0}.items(),
                               columns=['Column Name', 'No. Nulls'])
    assert df_nulls.equals(expected_df)


@patch('dsutils.main._load_data', return_value=pd.DataFrame())
def test_null_counters_raises_if_empty_df(mocked_load_data):
    with pytest.raises(ValueError) as e:
        null_counter('')
    assert str(e.value) == 'Cannot count nulls on an empty dataframe'
