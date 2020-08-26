
import numpy as np
import pandas as pd
import pytest
from pythia.learning import AR_DataModule
from pythia.learning.transforms import ToTensor
from sunpy.util import SunpyUserWarning


@pytest.fixture
def dummy_data():
    X = range(0, 100)
    y = np.zeros((100, ))
    y[:80] = 1
    df = pd.DataFrame()
    df['X'] = X
    df['y'] = y
    return df


def distribution(array):
    _, c = np.unique(array, return_counts=True)
    return c / len(array)


# Test Prepare Data
def test_default_loader(dummy_data):
    AR_DataModule(data=dummy_data)


def test_no_y_specified(dummy_data):

    with pytest.raises(ValueError):
        dataloader = AR_DataModule(data=dummy_data)
        dataloader.prepare_data()


def test_incorrect_data():

    dataloader = AR_DataModule(data=42)
    with pytest.raises(TypeError):
        dataloader.prepare_data()


def test_incorrect_test_train_split(dummy_data):

    dataloader = AR_DataModule(data=dummy_data,
                               y_col='y',
                               train_test_split=42)
    with pytest.raises(ValueError):
        dataloader.prepare_data()


def test_incorrect_val_train_split(dummy_data):

    dataloader = AR_DataModule(data=dummy_data,
                               y_col='y',
                               train_val_split=42)
    with pytest.raises(ValueError):
        dataloader.prepare_data()


# Test Setup

def test_disjoint_split(dummy_data):
    dataloader = AR_DataModule(data=dummy_data,
                               y_col='y')
    dataloader.prepare_data()
    dataloader.setup()
    assert set(dataloader.train.index).isdisjoint(dataloader.val.index)
    assert set(dataloader.train.index).isdisjoint(dataloader.test.index)
    assert set(dataloader.test.index).isdisjoint(dataloader.val.index)


def test_split_distribution(dummy_data):
    dataloader = AR_DataModule(data=dummy_data,
                               y_col='y')
    dataloader.prepare_data()
    dataloader.setup()
    data_dist = distribution(dataloader.data['y'])
    assert not np.any(np.abs(distribution(dataloader.train['y']) - data_dist) > 0.1)
    assert not np.any(np.abs(distribution(dataloader.test['y']) - data_dist) > 0.1)
    assert not np.any(np.abs(distribution(dataloader.val['y']) - data_dist) > 0.1)


def test_conf_passed(dummy_data):
    training_conf = {}
    testing_conf = {}
    validation_conf = {}

    dataloader = AR_DataModule(data=dummy_data,
                               y_col='y',
                               train_conf=training_conf,
                               test_conf=testing_conf,
                               val_conf=validation_conf)

    dataloader.prepare_data()

    with pytest.warns(None) as record:
        dataloader.setup()
    assert not record


def test_no_conf(dummy_data):
    training_conf = {}
    testing_conf = {}
    validation_conf = {}

    dataloader_no_train_conf = AR_DataModule(data=dummy_data,
                                             y_col='y',
                                             test_conf=testing_conf,
                                             val_conf=validation_conf)

    dataloader_no_train_conf.prepare_data()
    with pytest.warns(SunpyUserWarning):
        dataloader_no_train_conf.setup()

    dataloader_no_val_conf = AR_DataModule(data=dummy_data,
                                           y_col='y',
                                           test_conf=testing_conf,
                                           train_conf=training_conf)

    dataloader_no_val_conf.prepare_data()
    with pytest.warns(SunpyUserWarning):
        dataloader_no_val_conf.setup()

    dataloader_no_test_conf = AR_DataModule(data=dummy_data,
                                            y_col='y',
                                            train_conf=training_conf,
                                            val_conf=validation_conf)

    dataloader_no_test_conf.prepare_data()
    with pytest.warns(SunpyUserWarning):
        dataloader_no_test_conf.setup()


def test_oversampling(dummy_data):
    train_conf = {'is_tabular': True, 'transform': ToTensor()}
    dataloader = AR_DataModule(data=dummy_data,
                               y_col='y',
                               batch_size=1,
                               train_conf=train_conf)
    dataloader.prepare_data()
    dataloader.setup()

    sampled_labels = []
    for index, sample in enumerate(dataloader.train_dataloader()):
        sampled_labels.append(sample[1].numpy()[0])

    data_distribution = distribution(sampled_labels)

    assert np.abs(data_distribution[0] - data_distribution[1]) < 0.2
