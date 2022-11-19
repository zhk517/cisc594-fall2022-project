import numpy as np
import pandas as pd


def _load_data(fp):
    try:
        df = pd.read_csv(fp)
        print(f"Successfully loaded data from {fp}")
    except Exception as e:
        raise ValueError(f'Failed to load data from path: {fp}\n{e}')
    return df


def train_test_splitter(fp, ratio=0.7, seed=42, save_to=True):
    """Load data from the given file path and split into training
    and testing datasets based on ratio"""
    df = _load_data(fp)
    if len(df) == 0:
        raise ValueError('Cannot split an empty dataframe')
    if ratio <= 0 or ratio >= 1:
        raise ValueError('Split ratio must be within range (0, 1)')
    n_rows = len(df)
    np.random.seed(seed)
    mask = np.random.rand(n_rows) < ratio
    df_train, df_test = df[mask], df[~mask]
    print(f"Training set contains {len(df_train)} rows")
    print(f"Testing set contains {len(df_test)} rows")
    if save_to:
        df_train.to_csv('training.csv', index=False)
        df_test.to_csv('testing.csv', index=False)
    else:
        return df_train, df_test


def null_counter(fp):
    """Load data from the given file path, display number of null values per column
    and return a dataframe that contains null values summary"""
    df = _load_data(fp)
    d_nulls = {}
    if len(df) == 0:
        raise ValueError('Cannot count nulls on an empty dataframe')
    for col in df:
        n_nulls = np.sum(df[col].isnull())
        pct = 100 * n_nulls / len(df)
        d_nulls[col] = n_nulls
        if n_nulls == 0:
            continue
        print(f"{col}: {n_nulls} ({pct:0.2f}%) missing values")
    return pd.DataFrame(d_nulls.items(), columns=['Column Name', 'No. Nulls'])
