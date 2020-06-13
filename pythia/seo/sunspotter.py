import pandas as pd
import numpy as np
from sunpy.util import SunpyUserWarning
from pathlib import Path

__all__ = ['Sunspotter']

path = Path(__file__).parent.parent.parent / "data/all_clear"

class Sunspotter:

    def __init__(self, *, timesfits: str = path / "lookup_timesfits.csv", get_all_timesfits_columns: bool = True,
                 properties: str = path / "lookup_properties.csv", get_all_properties_columns: bool = True, 
                 timesfits_columns: list = ['#id'], properties_columns: list = ['#id'],
                 classifications=None, classifications_columns=None,
                 delimiter: str = ';', datetime_fmt: str = '%Y-%m-%d %H:%M:%S'):
        """
        Parameters
        ----------
        timesfits : str
            filepath to `lookup_timesfits.csv`
            by default points to the Timesfits file from All Clear Dataset
            stored in `~pythia/data/all_clear`
        get_all_timesfits_columns : bool, optional
            Load all columns from the Timesfits CSV file, by default True
        properties : str
            filepath to `lookup_properties.csv`
            by default points to the Properties file from All Clear Dataset
            stored in `~pythia/data/all_clear`
        get_all_properties_columns : bool, optional
            Load all columns from the Properties CSV file, by default True
        timesfits_columns : list, optional
            Columns required from lookup_timesfits.csv, by default ['#id']
            Will be overridden if `get_all_timesfits_columns` is True.
        properties_columns : list, optional
            Columns required from lookup_properties.csv, by default ['#id']
            Will be overridden if `get_all_properties_columns` is True.
        classifications : str, optional
            filepath to `classifications.csv`
            Default behaviour is not to load the file, hence by default None
        classifications_columns : list, optional
            Columns required from `classifications.csv`
            Default behaviour is not to load the file, hence by default None
        delimiter : str, optional
            Delimiter for the CSV files, by default ';'
        datetime_fmt : str, optional
            Format for interpreting the observation datetimes in the CSV files,
            by default '%Y-%m-%d %H:%M:%S'
        """
        self.timesfits = timesfits
        self.get_all_timesfits_columns = get_all_timesfits_columns

        self.properties = properties
        self.get_all_properties_columns = get_all_properties_columns

        self.timesfits_columns = set(timesfits_columns)
        self.properties_columns = set(properties_columns)

        self.classifications = classifications
        self.classifications_columns = classifications_columns

        self.datetime_fmt = datetime_fmt

        self._get_data(delimiter)

    def _get_data(self, delimiter: str):
        # Reading the Timesfits file
        try:
            if self.get_all_timesfits_columns:
                self.timesfits = pd.read_csv(self.timesfits,
                                            delimiter=delimiter)
            else:
                self.timesfits = pd.read_csv(self.timesfits,
                                            delimiter=delimiter,
                                            usecols=self.timesfits_columns)
        except ValueError:
            raise SunpyUserWarning("Sunspotter Object cannot be created."
                                   " Either the Timesfits columns do not match, or the file is corrupted")

        if not self.timesfits_columns.issubset(self.timesfits.columns):
            missing_columns = self.timesfits_columns - self.timesfits_columns.intersection(self.timesfits.columns)
            missing_columns = ", ".join(missing_columns)

            raise SunpyUserWarning("Sunspotter Object cannot be created."
                                   " The Timesfits CSV is missing the following columns: " +
                                   missing_columns)

        if 'obs_date' in self.timesfits.columns:
            self.timesfits.obs_date = pd.to_datetime(self.timesfits.obs_date,
                                                     format=self.datetime_fmt)
            self.timesfits.set_index("obs_date", inplace=True)

        # Reading the Properties file
        try:
            if self.get_all_properties_columns:
                self.properties = pd.read_csv(self.properties,
                                            delimiter=delimiter)
            else:
                self.properties = pd.read_csv(self.properties,
                                            delimiter=delimiter,
                                            usecols=self.properties_columns)
        except ValueError:
            raise SunpyUserWarning("Sunspotter Object cannot be created."
                                   " Either the Properties columns do not match, or the file is corrupted")

        if not self.properties_columns.issubset(self.properties.columns):
            missing_columns = self.properties_columns - self.properties_columns.intersection(self.properties.columns)
            missing_columns = ", ".join(missing_columns)

            raise SunpyUserWarning("Sunspotter Object cannot be created."
                                   " The Properties CSV is missing the following columns: " +
                                   missing_columns)

        # Reading the Classification file
        if self.classifications is not None:

            if self.classifications_columns is None:
                raise SunpyUserWarning("Classifications columns cannot be None"
                                       "  when classifications.csv is to be loaded.")
            try:
                self.classifications = pd.read_csv(self.classifications,
                                                   delimiter=delimiter,
                                                   usecols=self.classifications_columns)
            except ValueError:
                raise SunpyUserWarning("Sunspotter Object cannot be created."
                                       " Either the Classifications columns do not match, or the file is corrupted")

            self.classifications_columns = set(self.classifications_columns)

            if not self.classifications_columns.issubset(self.classifications.columns):
                missing_columns = self.classifications_columns - self.classifications_columns.intersection(self.classifications.columns)
                missing_columns = ", ".join(missing_columns)

                raise SunpyUserWarning("Sunspotter Object cannot be created."
                                       " The Classifications CSV is missing the following columns: " +
                                       missing_columns)

    def get_timesfits_id(self, obsdate: str):
        """
        Returns the Sunspotter observation id for the
        first observation a given observation date and time.

        Parameters
        ----------
        obsdate : str
            The observation time and date.

        Returns
        -------
        id : int
            The Sunspotter observation id for the first observation
            for the given observation date and time.

        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter()
        >>> obsdate = '2000-01-01 12:47:02'
        >>> sunspotter.get_timesfits_id(obsdate)
        1
        """
        return self.timesfits.loc[obsdate].get(key='#id').iloc[0]

    def get_all_ids_for_observation(self, obsdate: str):
        """
        Returns all the Sunspotter observation ids for the
        given observation date and time.

        Parameters
        ----------
        obsdate : str
            The observation time and date.

        Returns
        -------
        ids : pandas.Series
            All the Sunspotter observation ids for the 
            given observation date and time.

        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter()
        >>> obsdate = '2000-01-01 12:47:02'
        >>> sunspotter.get_all_ids_for_observation(obsdate)
        array([1, 2, 3, 4, 5])
        """
        return self.timesfits.loc[obsdate].get(key='#id').values

    def get_properties(self, idx: int):
        """
        Returns the observed properties for a given Sunspotter id.

        Parameters
        ----------
        idx : int
            The Sunspotter observation id for a particualar observation.

        Returns
        -------
        properties : pandas.DataFrame
            The observed properties for the given Sunspotter id.

        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter()
        >>> sunspotter.get_properties(1)
            #id                      filename zooniverse_id  ...  pxpos_y  sszn  zurich
        0    1  530be1183ae74079c300000d.jpg    ASZ000090y  ...  166.877     1     bxo
        [1 rows x 23 columns]
        """
        return self.properties[self.properties.id_filename == idx]

    def get_properties_from_obsdate(self, obsdate: str):
        """
        Returns the observed properties for a given observation time and date.

        Parameters
        ----------
        obsdate : str
            The observation time and date.

        Returns
        -------
        properties : pandas.DataFrame
            The observed properties for the given observation time and date.

        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter()
        >>> obsdate = '2000-01-01 12:47:02'
        >>> sunspotter.get_properties_from_obsdate(obsdate)
            #id                      filename zooniverse_id  ...  pxpos_y  sszn  zurich
        0    1  530be1183ae74079c300000d.jpg    ASZ000090y  ...  166.877     1     bxo
        [1 rows x 23 columns]
        """
        return self.get_properties(self.get_timesfits_id(obsdate))
