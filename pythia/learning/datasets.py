import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from sunpy.util import SunpyUserWarning
from torch.utils.data import Dataset

__all__ = ['AR_Dataset']


class AR_Dataset(Dataset):
    """
    AR dataset for FITS files.
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
        y_col : str
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
        if not isinstance(y_col, str):
            raise TypeError("y_col must be a string denoting the label column")

        if is_tabular is True and is_fits is True:
            warnings.warn(SunpyUserWarning("`is_tabular` and `is_fits` flags both cannot be simultaneously True",
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
            img_name = self.X.iloc[idx][0]

            if self.is_fits:
                image = fits.getdata(self.root_dir + img_name)
            else:
                image = plt.imread(self.root_dir + img_name)

            X = image

        sample = (X, np.array((self.y.iloc[idx])))

        if self.transform:
            sample = self.transform(sample)

        return sample
