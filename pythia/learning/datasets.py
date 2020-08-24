import numpy as np
import torch
from astropy.io import fits
from scipy import ndimage
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset

__all__ = ['AR_FITS_Dataset']

class AR_FITS_Dataset(Dataset):
    """
    AR dataset for FITS files.
    """
    def __init__(self, *, data, X_col, y_col, root_dir='data/all_clear/mdi/MDI/fits/',
                 transform=None, rotation=None, transpose=False, flip=False, flip_axis=0):
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
        rotation : int or float, optional
            Angle in deg for rotating the image, by default None
        transpose : bool, optional
            Flag to denote if image has to be transposed, by default False
        flip : bool, optional
            Flag to denote if image has to be flipped, by default False
        flip_axis : int, optional
            Flip Axis, by default 0
        """
        self.data = data
        self.X_col = X_col
        self.y_col = y_col
        self.root_dir = root_dir
        self.transform = transform
        self.rotation = rotation
        self.transpose = transpose
        self.flip = flip
        self.flip_axis = flip_axis

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
        return len(self.df)

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

        Raises
        ------
        ValueError
            If rotation angle is not an int or float.
        ValueError
            If flip axis is not 0 or 1.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.X.iloc[idx]
        image = fits.getdata(self.root_dir + img_name)

        image = np.nan_to_num(image)
        image = normalize(image, axis=1, norm='l1')

        if self.rotation is not None:
            if not isinstance(self.rotation, (int, float)):
                raise ValueError("Rotation must be an int or a float.")
            image = ndimage.rotate(image, self.rotation, reshape=False)

        if self.transpose is True:
            image = image.T

        if self.flip is True:
            if not self.flip_axis == 0 or self.flip_axis != 1:
                raise ValueError("Flip Axis must be 0 or 1.")
            image = np.flip(image, axis=self.flip_axis)

        sample = (image, np.array((self.y.iloc[idx])))

        if self.transform:
            sample = self.transform(sample)

        return sample
