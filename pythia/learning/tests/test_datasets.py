from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from pythia.learning.datasets import BaseDataset
from pythia.learning.transforms import *
from torchvision import transforms

PATH = Path(__file__).parent / "test_data/"


@pytest.fixture
def tabular_data():
    tabular_data = {'id': 0,
                    'X': 42,
                    'y': 1}

    return pd.DataFrame(tabular_data, index=['id'])


@pytest.fixture
def fits_data():
    fits_data = {'id': 0,
                 'filename': '20000101_1247_mdiB_1_8809.fits',
                 'noaa': 8809,
                 'flares': 0,
                 'observation_number': 1}

    return pd.DataFrame(fits_data, index=['id'])


@pytest.fixture
def img_data():
    img_data = {'id': 0,
                'filename': '5397a56aa57caf04c6000001.jpg',
                'noaa': 0,
                'flares': 0,
                'observation_number': 1}

    return pd.DataFrame(img_data, index=['id'])


@pytest.fixture
def composed_transforms(size=(100, 100)):
    return transforms.Compose([RemoveNaN(), Rescale(size), Normalize(),
                               FixChannel(), ToTensor()])


@pytest.fixture
def X_col():
    return ['filename']


@pytest.fixture
def y_col():
    return ['flares']


@pytest.fixture
def fits_file():
    return fits.getdata(PATH / "20000101_1247_mdiB_1_8809.fits")


@pytest.fixture
def img_file():
    return plt.imread(PATH / "5397a56aa57caf04c6000001.jpg")


@pytest.fixture
def default_dataset(fits_data, X_col, y_col):
    return BaseDataset(data=fits_data,
                       root_dir=str(PATH) + "/",
                       X_col=X_col,
                       y_col=y_col)


def test_default_dataset(default_dataset, fits_file):

    assert len(default_dataset) == 1
    assert len(default_dataset[0]) == 2

    X, y = default_dataset[0]

    assert np.array_equal(X, fits_file)
    assert y.dtype == np.int64


def test_img_dataset(img_data, X_col, y_col, img_file):

    dataset = BaseDataset(data=img_data,
                          root_dir=str(PATH) + "/",
                          X_col=X_col,
                          y_col=y_col,
                          is_fits=False)

    assert len(dataset) == 1
    assert len(dataset[0]) == 2

    X, y = dataset[0]

    assert np.array_equal(X, img_file)
    assert y.dtype == np.int64


def test_tabular_dataset(tabular_data):

    dataset = BaseDataset(data=tabular_data,
                          X_col='X',
                          y_col='y',
                          is_fits=False,
                          is_tabular=True)

    assert len(dataset) == 1
    assert len(dataset[0]) == 2

    X, y = dataset[0]

    assert X == tabular_data.iloc[0]['X']
    assert y == tabular_data.iloc[0]['y']


def test_apply_transforms(fits_data, X_col, y_col, composed_transforms):
    dataset = BaseDataset(data=fits_data,
                          root_dir=str(PATH) + "/",
                          X_col=X_col,
                          y_col=y_col,
                          transform=composed_transforms)

    X, y = dataset[0]
    assert len(X.shape) == 3
    assert X.shape == (1, 100, 100)
