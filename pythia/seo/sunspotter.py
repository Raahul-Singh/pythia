import pandas as pd
from sunpy.util import SunpyUserWarning

__all__ = ['Sunspotter']


class Sunspotter:

    def __init__(self, *, timesfits: str, properties: str, delimiter: str=';',
                 properties_columns: list=['#id'], timesfits_columns: list=['#id'],
                 classifications=None, classifications_columns=None):
        """
        Parameters
        ----------
        timesfits : str
            filepath to `lookup_timesfits.csv`
        properties : str
            filepath to `lookup_properties.csv`
        delimiter : str, optional
            Delimiter for the CSV files, by default ';'
        properties_columns : list, optional
            Columns required from lookup_properties.csv, by default ['#id']
        timesfits_columns : list, optional
            Columns required from lookup_timesfits.csv, by default ['#id']
        classifications : str, optional
            filepath to `classifications.csv`
            Default behaviour is not to load the file, hence by default None
        classifications_columns : list, optional
            Columns required from `classifications.csv`
            Default behaviour is not to load the file, hence by default None
        """
        self.timesfits = timesfits
        self.properties = properties
        self.timesfits_columns = set(timesfits_columns)
        self.properties_columns = set(properties_columns)
        self.classifications = classifications
        self.classifications_columns = classifications_columns

        self._get_data(delimiter)

    def _get_data(self, delimiter: str):

        try:
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
        try:
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
        Returns the Sunspotter observation id for a given observation date and time.

        Parameters
        ----------
        obsdate : str
            The observation time and date.

        Returns
        -------
        id : int
            The Sunspotter observation id for the given observation date and time.

        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter(timesfits="lookup_timesfits.csv", properties="lookup_properties.csv")
        >>> obsdate = '2000-01-01 12:47:02'
        >>> ssp.get_timesfits_id(obsdate)
        1
        """
        return self.timesfits[self.timesfits.obs_date == obsdate].get(key='#id').iloc[0]

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
        >>> sunspotter = Sunspotter(timesfits="lookup_timesfits.csv", properties="lookup_properties.csv")
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
        >>> sunspotter = Sunspotter(timesfits="lookup_timesfits.csv", properties="lookup_properties.csv")
        >>> obsdate = '2000-01-01 12:47:02'
        >>> sunspotter.get_properties_from_obsdate(obsdate)
            #id                      filename zooniverse_id  ...  pxpos_y  sszn  zurich
        0    1  530be1183ae74079c300000d.jpg    ASZ000090y  ...  166.877     1     bxo
        [1 rows x 23 columns]
        """
        return self.get_properties(self.get_timesfits_id(obsdate))