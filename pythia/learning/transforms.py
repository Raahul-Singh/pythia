import numpy as np
import torch
from torchvision import transform

__all__ = ['Rescale', 'ToTensor', 'FixChannel']


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

        transform.resize(X, (new_h, new_w))

        return (X, y)

class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """
    def __call__(self, sample):
        X, y = sample

        # swap color axis because
        # numpy X: H x W x C
        # torch X: C X H X W
        X = X.transpose((2, 0, 1))

        return (torch.from_numpy(X),
                torch.from_numpy(y))

class FixChannel(object):
    """
    Fixes the channels in the input sample.
    """
    def __call__(self, sample):
        X, y = sample
        X = np.stack((X,), axis=-1) # ',' is important!
        return (X, y)
