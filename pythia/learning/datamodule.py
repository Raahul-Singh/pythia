import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from pythia.learning import BaseDataset
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sunpy.util import SunpyUserWarning
from torch.utils.data import DataLoader

__all__ = ['BaseDataModule']


class BaseDataModule(pl.LightningDataModule):

    def __init__(self, data, *, y_col, X_col=None, root_dir=None,
                 train_size=1.0, train_test_split=0.2, train_val_split=0.2,
                 num_splits=1, is_regression=False, stratified_shuffle=True,
                 weighted_sampling=True, batch_size=4, train_conf=None,
                 test_conf=None, val_conf=None):
        """
        Parameters
        ----------
        data : pd.DataFrame or str
            The Dataframe with the data information,
            or the path to the CSV that contains the data information.
        y_col : list or str
            Label Column
        X_col : list or str
            Feature Columns
        root_dir : str, optional
            Path to the data files, if any, by default None
        train_size : float, optional
            Size of training data to be used for training, by default 1.0
        train_test_split : float, optional
            Size of train test split, by default 0.2
        train_val_split : float, optional
            Size of train val split, by default 0.2
        num_splits : int, optional
            Number of splits in K fold splitting, by default 1
        is_regression : bool, optional
            Is the task a regression task?, by default False
        stratified_shuffle : bool, optional
            Maintain class distribution acrosss splits?, by default True
        weighted_sampling : bool, optional
            If training dataset is to be over sampled corresponding to the class distribution.
            by default True
        batch_size : int, optional
            Number of datapoints in a batch, by default 4
        train_conf : dict, optional
            Configuration to be passed to Training dataset, by default None
        test_conf : dict, optional
            Configuration to be passed to Testing dataset, by default None
        val_conf : dict, optional
            Configuration to be passed to Validation dataset, by default None
        """
        super().__init__()

        if X_col is not None and not set(X_col).isdisjoint(set(y_col)):
            raise ValueError("Feature Columns and Label columns must be dijoint")

        self.data = data
        self.data_path = root_dir
        self.X_col = X_col
        self.y_col = y_col
        self.train_size = train_size
        self.train_test_split = train_test_split
        self.num_splits = num_splits
        self.is_regression = is_regression
        self.stratified_shuffle = stratified_shuffle
        self.weighted_sampling = weighted_sampling
        self.train_val_split = train_val_split
        self.batch_size = batch_size
        self.train_conf = train_conf
        self.test_conf = test_conf
        self.val_conf = val_conf

    def prepare_data(self):
        """
        Prepares the data DataFrame.

        Raises
        ------
        TypeError
            If the data argument is invalid.
        ValueError
            If target column is not explicitely specified.
        ValueError
            If train test split is not a fraction between 0 and 1.
        ValueError
            If train val split is not a fraction between 0 and 1.

        # TODO : Add support for K fold cross validataion. only 1 split supported as of now.
        """
        if isinstance(self.data, str):
            self.data = pd.read_csv(self.data)
        elif not isinstance(self.data, pd.DataFrame):
            raise TypeError("Explicitely passed data must be a pandas Dataframe")

        if self.X_col is None:
            warnings.warn(SunpyUserWarning("No Feature Columns specified." +
                                           "Assuming all columns except target columns to be feature columns."))
            self.X_col = set(self.data.columns) - set(self.y_col)

        if not isinstance(self.train_test_split, float) or self.train_test_split >= 1 or self.train_test_split <= 0:
            raise ValueError("train test split must be a fraction between 0 and 1")

        if not isinstance(self.train_val_split, float) or self.train_val_split >= 1 or self.train_val_split <= 0:
            raise ValueError("train val split must be a fraction between 0 and 1")

        if self.is_regression is True and self.stratified_shuffle is True:
            warnings.warn("Cannot use Stratified Shuffling with Regression tasks. Defaulting to Random Shuffling.")
            self.stratified_shuffle = False

        if self.stratified_shuffle is True:
            splitter = StratifiedShuffleSplit
        else:
            splitter = ShuffleSplit

        self.train_test_splitter = splitter(n_splits=self.num_splits, test_size=self.train_test_split)
        self.train_val_splitter = splitter(n_splits=self.num_splits, test_size=self.train_val_split)


    def setup(self, stage=None):
        """
        Dataset Generation function.

        Parameters
        ----------
        stage : str, optional
            Training or Testing stage, by default None
        """
        for train_index, test_index in self.train_test_splitter.split(X=self.data[self.X_col], y=self.data[self.y_col]):
            self.train, self.test = self.data.iloc[train_index], self.data.iloc[test_index]

        for train_index, val_index in self.train_val_splitter.split(X=self.train[self.X_col], y=self.train[self.y_col]):
            self.train, self.val = self.train.iloc[train_index], self.train.iloc[val_index]

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:

            if self.train_size > 1:
                self.train = self.train[:self.train]
            else:
                self.train = self.train[:int(len(self.train) * self.train_size)]

            if isinstance(self.train_conf, dict):
                self.train_dataset = BaseDataset(data=self.train, X_col=self.X_col,
                                                 y_col=self.y_col, **self.train_conf)
            else:
                warnings.warn(SunpyUserWarning("No training configurations specified, using default configuration."))
                self.train_dataset = BaseDataset(data=self.train, X_col=self.X_col,
                                                 y_col=self.y_col)

            if isinstance(self.val_conf, dict):
                self.val_dataset = BaseDataset(data=self.val, X_col=self.X_col,
                                              y_col=self.y_col, **self.val_conf)
            else:
                warnings.warn(SunpyUserWarning("No validation configurations specified, using default configuration."))
                self.val_dataset = BaseDataset(data=self.val, X_col=self.X_col,
                                              y_col=self.y_col)

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:

            if isinstance(self.test_conf, dict):
                self.test_dataset = BaseDataset(data=self.test, X_col=self.X_col,
                                                y_col=self.y_col, **self.test_conf)
            else:
                warnings.warn(SunpyUserWarning("No testing configurations specified, using default configuration."))
                self.test_dataset = BaseDataset(data=self.test, X_col=self.X_col,
                                                y_col=self.y_col)

    def train_dataloader(self):
        """
        Returns the Training Dataloader.

        Returns
        -------
        Dataloader : torch.DataLoader
            The Training Dataloader.
        """
        if self.is_regression is True and self.weighted_sampling is True:
            warnings.warn("Cannot use Weighted Sampling with Regression tasks. Defaulting to Random Shuffling.")
            self.weighted_sampling = False

        if self.weighted_sampling is True:
            classes, class_counts = np.unique(self.train[self.y_col], return_counts=True)
            class_weights = {}
            weights = 1 / torch.DoubleTensor(class_counts)

            for index, weight in enumerate(weights):
                class_weights[index] = weight
            weight_list = [class_weights[i] for i in self.train[self.y_col]]
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weight_list, len(weight_list))

            return DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=sampler)

        else:
            return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        """
        Returns the Validation Dataloader.

        Returns
        -------
        Dataloader : torch.DataLoader
            The Validation Dataloader.
        """
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """
        Returns the Testing Dataloader.

        Returns
        -------
        Dataloader : torch.DataLoader
            The Testing Dataloader.
        """
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
