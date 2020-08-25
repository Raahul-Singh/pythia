import numpy as np
import pytest
from pythia.learning.transforms import *
from scipy import ndimage


@pytest.fixture
def nan_array():
    array = np.zeros((5, 5))
    array[:, :] = np.nan
    return array


@pytest.fixture
def random_array():
    return np.random.randint(-100, 100, (5, 5))


def apply_transform(array, transform):
    dummy_label = np.array(0)
    transformed_array, _ = transform((array, dummy_label))
    return transformed_array


def test_RemoveNaN(nan_array):
    transform = RemoveNaN()
    transformed_array = apply_transform(nan_array, transform)

    assert nan_array.shape == transformed_array.shape

    assert np.any(np.isnan(nan_array))
    assert not np.any(np.isnan(transformed_array))


def test_Normalize(random_array):
    transform = Normalize()
    transformed_array = apply_transform(random_array, transform)

    assert random_array.shape == transformed_array.shape

    assert np.sum(random_array > 1) + np.sum(random_array < -1) > 0
    assert np.sum(transformed_array > 1) + np.sum(transformed_array < -1) == 0


def test_Transpose(random_array):
    transform = Transpose()
    transformed_array = apply_transform(random_array, transform)

    assert random_array.shape == transformed_array.shape

    assert np.array_equal(random_array.T, transformed_array)


@pytest.mark.parametrize("angle", [5, 10, 15.0, 90, -45])
def test_Rotate(random_array, angle):
    transform = Rotate(rotation=angle)
    transformed_array = apply_transform(random_array, transform)

    assert random_array.shape == transformed_array.shape
    assert np.allclose(transformed_array, ndimage.rotate(random_array, angle, reshape=False))


def test_Flip(random_array):
    transform = Flip()
    transformed_array = apply_transform(random_array, transform)

    assert random_array.shape == transformed_array.shape

    assert np.array_equal(np.flip(random_array, axis=0), transformed_array)


def test_Rescale(random_array):
    transform = Rescale((3, 3))
    transformed_array = apply_transform(random_array, transform)

    assert random_array.shape == (5, 5)

    assert transformed_array.shape == (3, 3)


def test_FixChannel(random_array):
    transform = FixChannel()
    transformed_array = apply_transform(random_array, transform)

    assert random_array.ndim == 2
    assert transformed_array.ndim == 3

    assert np.array_equal(random_array, transformed_array[:, :, 0])


def test_ToTensor(random_array):
    transform = FixChannel()
    channel_fixed_array = apply_transform(random_array, transform)
    transform = ToTensor()
    transformed_array = apply_transform(channel_fixed_array, transform)

    assert channel_fixed_array.shape == (5, 5, 1)
    assert transformed_array.shape == (1, 5, 5)

    assert np.array_equal(channel_fixed_array[:, :, 0], transformed_array[0, :, :])
