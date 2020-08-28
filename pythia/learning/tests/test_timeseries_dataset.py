
import numpy as np
import pandas as pd
import pytest
import torch
from torchvision.transforms import Compose
from pythia.learning.datasets import BaseTimeSeriesDataset
from pythia.learning.transforms import *
from sunpy.util import SunpyUserWarning


@pytest.fixture
def tabular_data():
    tabular_data = {
                    'col_1': [i for i in range(30)],
                    'col_2': [i * 2 for i in range(30)],
                    'y': [i ** 2 for i in range(30)]
                    }

    return pd.DataFrame(tabular_data)


@pytest.fixture
def composed_transforms(size=(100, 100)):
    return Compose([RemoveNaN(), ToTensor(), ToFloat()])



@pytest.fixture
def default_dataset(tabular_data):
    X_col = ['col_1', 'col_2']
    y_col = ['y']
    return BaseTimeSeriesDataset(data=tabular_data,
                                    X_col=X_col,
                                    y_col=y_col,
                                    sequence_length=1)


@pytest.mark.parametrize("seq_length", [1, 2, 3, 4, 5,
                                        6, 7, 8, 9, 10,
                                        11, 12, 13, 14, 15])
def test_default_dataset(tabular_data, seq_length):

    X_col = ['col_1', 'col_2']
    y_col = ['y']

    if len(tabular_data) % seq_length != 0:
        with pytest.warns(SunpyUserWarning):
            dataset = BaseTimeSeriesDataset(data=tabular_data,
                                               X_col=X_col,
                                               y_col=y_col,
                                               sequence_length=seq_length)

    else:
        dataset = BaseTimeSeriesDataset(data=tabular_data,
                                            X_col=X_col,
                                            y_col=y_col,
                                            sequence_length=seq_length)

    assert len(dataset.data) % seq_length == 0

    assert len(dataset) ==  (len(tabular_data) - len(tabular_data) % seq_length) // seq_length
    assert len(dataset[0][0]) == seq_length

    X, y = dataset[0]

    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)

    assert X.shape == (seq_length, 2)
    assert y.shape == (seq_length, 1)



def test_seq_len_more_than_half(tabular_data):

    X_col = ['col_1', 'col_2']
    y_col = ['y']

    with pytest.raises(ValueError):
        BaseTimeSeriesDataset(data=tabular_data,
                                 X_col=X_col,
                                 y_col=y_col,
                                 sequence_length=16)


def test_seq_len_more_than_data(tabular_data):

    X_col = ['col_1', 'col_2']
    y_col = ['y']

    with pytest.raises(ValueError):
        BaseTimeSeriesDataset(data=tabular_data,
                                 X_col=X_col,
                                 y_col=y_col,
                                 sequence_length=31)

@pytest.mark.parametrize("seq_length", [1, 2, 3, 4, 5])
def test_apply_transforms(tabular_data, seq_length, composed_transforms):

    X_col = ['col_1', 'col_2']
    y_col = ['y']

    dataset = BaseTimeSeriesDataset(data=tabular_data,
                                        X_col=X_col,
                                        y_col=y_col,
                                        sequence_length=seq_length,
                                        transform=composed_transforms)


    X, y = dataset[0]

    assert X.shape == (seq_length, 2)
    assert y.shape == (seq_length, 1)

    assert isinstance(X, torch.FloatTensor)
    assert isinstance(y, torch.FloatTensor)
