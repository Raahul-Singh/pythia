import numpy as np
import torch
from scipy import ndimage
from skimage import transform
from sklearn.preprocessing import normalize

__all__ = ['RemoveNaN', 'Normalize',  # Essential and should be applied first.
           'Transpose', 'Rotate', 'Flip', # Optional.
           'Rescale', 'FixChannel', 'ToTensor']  # Essential and should be applied Last.


class RemoveNaN(object):
    """
    Removes NaNs from array.
    """
    def __call__(self, sample):
        X, y = sample
        return (np.nan_to_num(X), y)


class Normalize(object):
    """
    Normalize the array.
    """
    def __init__(self, axis=1, norm='l1', **kwargs):
        """
        Parameters
        ----------
        axis : int, optional
            Axis to normalize by, by default 1
        norm : str, optional
            Algorithm to nomalize with, by default 'l1'
        kwargs : dict, optional
            Kwargs to be passed to `sklearn.preprocessing.normalize`
        """
        self.axis = axis
        self.norm = norm
        self.kwargs = kwargs

    def __call__(self, sample):
        X, y = sample
        X = normalize(X, axis=self.axis, norm=self.norm, **self.kwargs)
        return (X, y)


class Transpose(object):
    """
    Transposes the array.
    """
    def __call__(self, sample):
        X, y = sample
        return (X.T, y)


class Rotate(object):
    def __init__(self, rotation, reshape=False):
        """
        Rotate the array.

        Parameters
        ----------
        rotation : int or float, optional
            Angle in deg for rotating the image, by default None
        reshape : bool, optional
            Whether the original image dimensions need to be preserved, by default False

        Raises
        ------
        ValueError
            If rotation angle is not an int or float.
        """
        if not isinstance(rotation, (int, float)):
            raise ValueError("Rotation must be an int or a float.")

        self.rotation = rotation
        self.reshape = reshape

    def __call__(self, sample):

        X, y = sample
        X = ndimage.rotate(X, self.rotation, reshape=self.reshape)
        return (X, y)


class Flip(object):
    """
    Flip the array.
    """
    def __init__(self, axis=0):
        """
        Parameters
        ----------
        axis : int, optional
            Flip Axis, by default 0

        Raises
        ------
        ValueError
            If flip axis is not 0 or 1.
        """
        print(axis)
        if axis != 0 and axis != 1:
            raise ValueError("Flip Axis must be 0 or 1.")

        self.flip_axis = axis

    def __call__(self, sample):
        X, y = sample
        X = np.flip(X, axis=self.flip_axis)
        return (X, y)


class Rescale(object):
    """
    Rescales the input array to the specified size.
    """
    def __init__(self, output_size=(100, 100)):
        """
        Parameters
        ----------
        output_size : tuple or int, optional
            The size to which the input array has to be rescaled, by default (100, 100)
        """
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        X, y = sample

        h, w = X.shape[:2]

        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        X = transform.resize(X, (new_h, new_w))

        return (X, y)


class FixChannel(object):
    """
    Fixes the channels in the input sample.
    """
    def __call__(self, sample):
        X, y = sample
        X = np.stack((X, ), axis=-1)  # ',' is important!
        return (X, y)


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """
    def __call__(self, sample):
        X, y = sample

        # swap depth axis because
        # numpy X: H x W x C
        # torch X: C X H X W
        if X.ndim == 3:
            X = X.transpose((2, 0, 1))

        return (torch.from_numpy(X),
                torch.from_numpy(y))
