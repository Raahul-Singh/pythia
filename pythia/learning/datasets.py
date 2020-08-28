import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from sunpy.util import SunpyUserWarning
from torch.utils.data import Dataset

__all__ = ['BaseDataset', 'BaseTimeSeriesDataset']


class BaseDataset(Dataset):
    """
    Base dataset.
    """
    def __init__(self, *, data, X_col, y_col, root_dir='data/all_clear/mdi/MDI/fits/',
                 transform=None, is_fits=True, is_tabular=False):
        """
        Parameters
        ----------
        data : pd.DataFrame
            The Dataframe with the FITS data information.
        X_col : list or str
            Data Columns
        y_col : list or str
            Label Column
        root_dir : str, optional
            Path to the FITS files, by default 'data/all_clear/mdi/MDI/fits/'
        transform : torchvision.transforms, optional
            Data transforms, by default None
        is_fits : bool, optional
            Is the input Data in FITS files.
        is_tabular : bool, optional
            Is the input Data in Tabular.
        """
        if not isinstance(y_col, (str, list)):
            raise TypeError("y_col must be a list or string denoting the label column")

        if is_tabular is True and is_fits is True:
            warnings.warn(SunpyUserWarning("`is_tabular` and `is_fits` flags both cannot be simultaneously True "
                                           "Using tabular data for analysis"))

        self.data = data
        self.X_col = X_col
        self.y_col = y_col
        self.root_dir = root_dir
        self.transform = transform
        self.is_fits = is_fits
        self.is_tabular = is_tabular
        self.X = self.data[self.X_col]
        self.y = self.data[self.y_col]

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        length : int
            length of the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns the Datapoint corresponding to the index.

        Parameters
        ----------
        idx : int
            index

        Returns
        -------
        sample : tuple
            The Datapoint

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.is_tabular:
            X = np.array(self.X.iloc[idx])

        else:
            img_name = self.X.iloc[idx]
            if not isinstance(img_name, str):
                img_name = img_name[0]

            if self.is_fits:
                image = fits.getdata(self.root_dir + img_name)
            else:
                image = plt.imread(self.root_dir + img_name)

            X = image

        sample = (X, np.array((self.y.iloc[idx])))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class BaseTimeSeriesDataset(Dataset):
    """
    Base Time Series Dataset.
    """
    def __init__(self, *, data, X_col, y_col, sequence_length,
                 root_dir=None, transform=None, is_tabular=True):
        """
        Parameters
        ----------
        data : pd.DataFrame
            The Dataframe with the data information.
        X_col : list or str
            Feature Columns
        y_col : list or str
            Label Column
        sequence_length : int
            Length of the Sequence in the Time Series.
        root_dir : str, optional
            Path to the data files if any.
        transform : torchvision.transforms, optional
            Data transforms, by default None
        is_tabular : bool, optional
            Is the input Data in Tabular.
        """
        if not isinstance(y_col, (str, list)):
            raise TypeError("y_col must be a string or list denoting the label column(s)")
        if not set(X_col).isdisjoint(set(y_col)):
            raise ValueError("Feature Columns and Label columns must be dijoint")

        self.data = data
        self.X_col = X_col
        self.y_col = y_col
        self.sequence_length = sequence_length
        self.root_dir = root_dir
        self.transform = transform
        self.is_tabular = is_tabular

        if len(data) < self.sequence_length:
            raise ValueError("Length of dataset cannot be smaller than sequence length.")
        if self.sequence_length > len(data) // 2:
            raise ValueError("Length of sequence cannot be greater half of length of data.")

        residual_data_indices = len(data) % self.sequence_length
        if residual_data_indices > 0:
            warning_message = "The following indices cannot be loaded as a sequence : "
            leftover_indices = ", ".join([str(index) for index in range(len(data) - residual_data_indices, len(data))])
            warnings.warn(SunpyUserWarning(warning_message + leftover_indices))

            self.data = self.data[:-residual_data_indices]

        self.X = self.data[self.X_col]
        self.y = self.data[self.y_col]

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns
        -------
        length : int
            length of the dataset.
        """
        return len(self.data) // self.sequence_length

    def __getitem__(self, idx):
        """
        Returns the Datapoint corresponding to the index.

        Parameters
        ----------
        idx : int
            index

        Returns
        -------
        sample : tuple
            The Datapoint
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        idx = idx * self.sequence_length

        if self.is_tabular:
            X = np.array(self.X.iloc[idx:idx + self.sequence_length])

        sample = (X, np.array((self.y.iloc[idx:idx + self.sequence_length])))

        if self.transform:
            sample = self.transform(sample)

        return sample
