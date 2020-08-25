import matplotlib.pyplot as plt
import numpy as np
import torch
from astropy.io import fits
from torch.utils.data import Dataset

__all__ = ['AR_Dataset']

class AR_Dataset(Dataset):
    """
    AR dataset for FITS files.
    """
    def __init__(self, *, data, X_col, y_col, root_dir='data/all_clear/mdi/MDI/fits/',
                 transform=None, is_fits=True):
        """
        Parameters
        ----------
        data : pd.DataFrame
            The Dataframe with the FITS data information.
        X_col : list or str
            Data Columns
        y_col : list or str
            Label Columns
        root_dir : str, optional
            Path to the FITS files, by default 'data/all_clear/mdi/MDI/fits/'
        transform : torchvision.transforms, optional
            Data transforms, by default None
        is_fits : bool, optional
            Is the input Data in FITS files.
        """
        self.data = data
        self.X_col = X_col
        self.y_col = y_col
        self.root_dir = root_dir
        self.transform = transform
        self.is_fits = is_fits

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

        img_name = self.X.iloc[idx][0]

        if self.is_fits:
            image = fits.getdata(self.root_dir + img_name)
        else:
            image = plt.imread(self.root_dir + img_name)

        sample = (image, np.array((self.y.iloc[idx][0])))

        if self.transform:
            sample = self.transform(sample)

        return sample
