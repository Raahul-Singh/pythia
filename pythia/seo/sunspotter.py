import pandas as pd
from sunpy.util import SunpyUserWarning

__all__ = ['Sunspotter']


class Sunspotter:

    def __init__(self, *, timefits: str, properties: str, delimiter=';',
                 properties_columns=['#id'], timefits_columns=['#id']):
        """
        Parameters
        ----------
        timefits : str
            filepath to `lookup_timefits.csv`
        properties : str
            filepath to `lookup_properties.csv`
        delimiter : str, optional
            Delimiter for the CSV files, by default ';'
        properties_columns : list, optional
            Columns required from lookup_properties.csv, by default ['#id']
        timefits_columns : list, optional
            Columns required from lookup_timefits.csv, by default ['#id']
        """
        self.timefits = timefits
        self.properties = properties
        self.timefits_columns = set(timefits_columns)
        self.properties_columns = set(properties_columns)

        self._get_data(delimiter)

    def _get_data(self, delimiter):

        self.timefits = pd.read_csv(self.timefits, delimiter=delimiter)

        if not self.timefits_columns.issubset(self.timefits.columns):
            missing_columns = self.timefits_columns - self.timefits_columns.intersection(self.timefits.columns)
            missing_columns = ", ".join(missing_columns)

            raise SunpyUserWarning("Sunspotter Object cannot be created."
                                   " The Timefits CSV is missing the following columns: " +
                                   missing_columns)

        self.properties = pd.read_csv(self.properties, delimiter=delimiter)

        if not self.properties_columns.issubset(self.properties.columns):
            missing_columns = self.properties_columns - self.properties_columns.intersection(self.properties.columns)
            missing_columns = ", ".join(missing_columns)

            raise SunpyUserWarning("Sunspotter Object cannot be created."
                                   " The Properties CSV is missing the following columns: " +
                                   missing_columns)

    def get_timefits_id(self, obsdate: str):
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
        >>> from pythia.cleaning.midnight_rotation import SunspotterMidnightRotation
        >>> ssp = SunspotterMidnightRotation(timefits="lookup_timefits.csv", properties="lookup_properties.csv")
        >>> obsdate = '2000-01-01 12:47:02'
        >>> ssp.get_timefits_id(obsdate)
        1
        """
        return self.timefits[self.timefits.obs_date == obsdate].get(key='#id').iloc[0]

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
        >>> from pythia.cleaning.midnight_rotation import SunspotterMidnightRotation
        >>> ssp = SunspotterMidnightRotation(timefits="lookup_timefits.csv", properties="lookup_properties.csv")
        >>> ssp.get_properties(1)
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
        >>> from pythia.cleaning.midnight_rotation import SunspotterMidnightRotation
        >>> ssp = SunspotterMidnightRotation(timefits="lookup_timefits.csv", properties="lookup_properties.csv")
        >>> obsdate = '2000-01-01 12:47:02'
        >>> ssp.get_properties_from_obsdate(obsdate)
            #id                      filename zooniverse_id  ...  pxpos_y  sszn  zurich
        0    1  530be1183ae74079c300000d.jpg    ASZ000090y  ...  166.877     1     bxo
        [1 rows x 23 columns]
        """
        return self.get_properties(self.get_timefits_id(obsdate))
