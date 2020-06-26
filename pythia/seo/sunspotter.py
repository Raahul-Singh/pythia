import warnings
from pathlib import Path

import astropy.units as u
import matplotlib.pyplot as plt
import pandas as pd
from astropy.coordinates import SkyCoord
from pythia.cleaning import MidnightRotation
from sunpy.map import Map, MapSequence
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.net import hek
from sunpy.util import SunpyUserWarning

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

        if '#id' in self.properties.columns:
            self.properties.set_index("#id", inplace=True)

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
        obsdate = self.get_nearest_observation(obsdate)
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
        obsdate = self.get_nearest_observation(obsdate)
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
        properties : pandas.Series
            The observed properties for the given Sunspotter id.

        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter()
        >>> idx = 0
        >>> sunspotter.get_properties(idx)
        filename         530be1183ae74079c300000d.jpg
        zooniverse_id                      ASZ000090y
        angle                                 37.8021
        area                                    34400
        areafrac                                 0.12
        areathesh                                2890
        bipolesep                                3.72
        c1flr24hr                                   0
        id_filename                                 1
        flux                                 2.18e+22
        fluxfrac                                 0.01
        hale                                     beta
        hcpos_x                                452.27
        hcpos_y                                443.93
        m1flr12hr                                   0
        m5flr12hr                                   0
        n_nar                                       1
        noaa                                     8809
        pxpos_x                               229.193
        pxpos_y                               166.877
        sszn                                        1
        zurich                                    bxo
        Name: 1, dtype: object
        """
        return self.properties.loc[idx]

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
        filename         530be1183ae74079c300000d.jpg
        zooniverse_id                      ASZ000090y
        angle                                 37.8021
        area                                    34400
        areafrac                                 0.12
        areathesh                                2890
        bipolesep                                3.72
        c1flr24hr                                   0
        id_filename                                 1
        flux                                 2.18e+22
        fluxfrac                                 0.01
        hale                                     beta
        hcpos_x                                452.27
        hcpos_y                                443.93
        m1flr12hr                                   0
        m5flr12hr                                   0
        n_nar                                       1
        noaa                                     8809
        pxpos_x                               229.193
        pxpos_y                               166.877
        sszn                                        1
        zurich                                    bxo
        Name: 1, dtype: object
        [1 rows x 23 columns]
        """
        return self.get_properties(self.get_timesfits_id(obsdate))

    def number_of_observations(self, obsdate: str):
        """
        Returns number of Sunspotter observations for the
        given observation date and time.

        Parameters
        ----------
        obsdate : str
            The observation time and date.

        Returns
        -------
        number_of_observations : int
            Number of Sunspotter observations
            for the given observation date and time.

        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter()
        >>> obsdate = '2000-01-01 12:47:02'
        >>> sunspotter.number_of_observations(obsdate)
        5
        """
        return self.timesfits.loc[obsdate].shape[0]

    def get_nearest_observation(self, obsdate: str):
        """
        Returns the observation time and date in the Timesfits that is
        closest to the given observation time and date.

        Parameters
        ----------
        obsdate : str
            The observation time and date.

        Returns
        -------
        closest_observation : str
            Observation time and date in the Timesfits that is
            closest to the given observation time and date.

        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter()
        >>> obsdate = '2000-01-01 22:47:02'
        >>> sunspotter.get_nearest_observation(obsdate)
        '2000-01-01 12:47:02'
        """
        unique_dates = self.timesfits.index.unique()
        index = unique_dates.get_loc(obsdate, method='nearest')
        nearest_date = str(unique_dates[index])
        if nearest_date != str(obsdate):  # casting to str because obsdate can be a pandas.Timestamp
            warnings.warn(SunpyUserWarning("The given observation date isn't in the Timesfits file.\n"
                                           "Using the observation nearest to the given obsdate instead."))
        return nearest_date

    def get_all_observations_ids_in_range(self, start: str, end: str):
        """
        Returns all the observations ids in the given timerange.
        The nearest start and end time in the Timesfits are used
        to form the time range.

        Parameters
        ----------
        start : str
            The starting observation time and date.
        end : str
            The ending observation time and date.

        Returns
        -------
        ids : numpy.array
            All the Sunspotter observation ids for the
            given observation time range.

        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter()
        >>> start = '2000-01-02 12:51:02'
        >>> end = '2000-01-03 12:51:02'
        >>> sunspotter.get_all_observations_ids_in_range(start, end)
        array([ 6,  7,  8,  9, 10, 11, 12, 13])
        """
        start = self.get_nearest_observation(start)
        end = self.get_nearest_observation(end)
        return self.timesfits[start:end]['#id'].values

    def get_fits_filenames_from_range(self, start: str, end: str):
        """
        Returns all the FITS filenames for observations in the given timerange.
        The nearest start and end time in the Timesfits are used to form the
        time range.

        Parameters
        ----------
        start : str
            The starting observation time and date.
        end : str
            The ending observation time and date.

        Returns
        -------
        filenames : pandas.Series
            all the FITS filenames for observations in the given timerange.

        Notes
        -----
        If start time is equal to end time, all the filenames corresponding to
        that particular observation will be returned.

        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter()
        >>> start = '2000-01-02 12:51:02'
        >>> end = '2000-01-03 12:51:02'
        >>> sunspotter.get_fits_filenames_from_range(start, end)
        obs_date
        2000-01-02 12:51:02    20000102_1251_mdiB_1_8810.fits
        2000-01-02 12:51:02    20000102_1251_mdiB_1_8813.fits
        2000-01-02 12:51:02    20000102_1251_mdiB_1_8814.fits
        2000-01-02 12:51:02    20000102_1251_mdiB_1_8815.fits
        2000-01-03 12:51:02    20000103_1251_mdiB_1_8810.fits
        2000-01-03 12:51:02    20000103_1251_mdiB_1_8813.fits
        2000-01-03 12:51:02    20000103_1251_mdiB_1_8814.fits
        2000-01-03 12:51:02    20000103_1251_mdiB_1_8815.fits
        Name: filename, dtype: object
        """
        ids_in_range = self.get_all_observations_ids_in_range(start, end)
        return self.timesfits[self.timesfits['#id'].isin(ids_in_range)]['filename']

    def get_mdi_fulldisk_fits_file(self, obsdate: str, filepath: str = str(path) + "/fulldisk/"):
        """
        Downloads the MDI Fulldisk FITS file corresponding to a particular observation.

        Parameters
        ----------
        obsdate : str
            The observation time and date.
        filepath : mdi_mapsequence : sunpy.map.MapSequence,
            By default downloaded files are stored in `~pythia/data/fulldisk`

        Returns
        -------
        filepath : str
            Filepath to the downloaded FITS file.

        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter()
        >>> obsdate = '2000-01-01 12:47:02'
        >>> sunspotter.get_mdi_fulldisk_fits_file(obsdate)
        '~pythia/data/all_clear/fulldisk/fd_m_96m_01d_2556_0008.fits'
        """
        # TODO: Figure out a way to test the downloaded file.
        obsdate = self.get_nearest_observation(obsdate)
        search_results = Fido.search(a.Time(obsdate, obsdate), a.Instrument.mdi)
        downloaded_file = Fido.fetch(search_results, path=filepath)
        return downloaded_file[0]

    def get_mdi_fulldisk_map(self, obsdate: str, filepath: str = str(path) + "/fulldisk/"):
        """
        Downloads the MDI Fulldisk FITS file corresponding to a particular observation.
        And returns a SunPy Map corresponding to the downloaded file.

        Parameters
        ----------
        obsdate : str
            The observation time and date.
        filepath : mdi_mapsequence : sunpy.map.MapSequence,
            By default downloaded files are stored in `~pythia/data/fulldisk`

        Returns
        -------
        filepath : str
            Filepath to the downloaded FITS file.

        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter()
        >>> obsdate = '2000-01-01 12:47:02'
        >>> sunspotter.get_mdi_fulldisk_map(obsdate)
        <sunpy.map.sources.soho.MDIMap object at 0x7f6ca7aedc88>
        SunPy Map
        ---------
        Observatory:		 SOHO
        Instrument:		 MDI
        Detector:		 MDI
        Measurement:		 magnetogram
        Wavelength:		 0.0 Angstrom
        Observation Date:	 2000-01-01 12:47:02
        Exposure Time:		 0.000000 s
        Dimension:		 [1024. 1024.] pix
        Coordinate System:	 helioprojective
        Scale:			 [1.98083342 1.98083342] arcsec / pix
        Reference Pixel:	 [511.36929067 511.76453018] pix
        Reference Coord:	 [0. 0.] arcsec
        array([[nan, nan, nan, ..., nan, nan, nan],
            [nan, nan, nan, ..., nan, nan, nan],
            [nan, nan, nan, ..., nan, nan, nan],
            ...,
            [nan, nan, nan, ..., nan, nan, nan],
            [nan, nan, nan, ..., nan, nan, nan],
            [nan, nan, nan, ..., nan, nan, nan]], dtype=float32)
        """
        # TODO: Figure out the file naming convention to check if the file has been downloaded already.
        # TODO: Test this!
        obsdate = self.get_nearest_observation(obsdate)
        search_results = Fido.search(a.Time(obsdate, obsdate), a.Instrument.mdi)
        downloaded_file = Fido.fetch(search_results, path=filepath)
        return Map(downloaded_file[0])

    def get_available_obsdatetime_range(self, start: str, end: str):
        """
        Returns all the observations datetimes in the given timerange.
        The nearest start and end time in the Timesfits are used
        to form the time range.

        Parameters
        ----------
        start : str
            The starting observation time and date.
        end : str
            The ending observation time and date.

        Returns
        -------
        obs_list : pandas.DatetimeIndex
            All the Sunspotter observation datetimes for the
            given observation time range.

        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter()
        >>> start = '2000-01-01 12:47:02'
        >>> end = '2000-01-15 12:47:02'
        >>> sunspotter.get_available_obsdatetime_range(start, end)
        DatetimeIndex(['2000-01-01 12:47:02', '2000-01-02 12:51:02',
                    '2000-01-03 12:51:02', '2000-01-04 12:51:02',
                    '2000-01-05 12:51:02', '2000-01-06 12:51:02',
                    '2000-01-11 12:51:02', '2000-01-12 12:51:02',
                    '2000-01-13 12:51:02', '2000-01-14 12:47:02',
                    '2000-01-15 12:47:02'],
                    dtype='datetime64[ns]', name='obs_date', freq=None)
        """
        start = self.get_nearest_observation(start)
        end = self.get_nearest_observation(end)

        return self.timesfits[start: end].index.unique()

    def get_mdi_map_sequence(self, start: str, end: str, filepath: str = str(path) + "/fulldisk/"):
        """
        Get MDI Map Sequence for observations from given range.

        Parameters
        ----------
        start : str
            The starting observation time and date.
        end : str
            The ending observation time and date.
        filepath : str, optional
            [description], by default str(path)+"/fulldisk/"

        Returns
        -------
        mdi_mapsequence : sunpy.map.MapSequence
            Map Sequece of the MDI maps in the given range.

        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter()
        >>> start = '2000-01-01 12:47:02'
        >>> end = '2000-01-05 12:51:02'
        >>> sunspotter.get_mdi_map_sequence(start, end)
        <sunpy.map.mapsequence.MapSequence object at 0x7f2c7b85cda0>
        MapSequence of 5 elements, with maps from MDIMap
        """
        # TODO: Test this!
        obsrange = self.get_available_obsdatetime_range(start, end)
        maplist = []

        for obsdate in obsrange:
            maplist.append(self.get_mdi_fulldisk_map(obsdate, filepath))

        return MapSequence(maplist)

    def get_observations_from_hek(self, obsdate: str, event_type: str = 'AR',
                                  observatory: str = 'SOHO'):
        """
        Gets the observation metadata from HEK for the given obsdate.
        By default gets Active Region data recieved from SOHO.

        Parameters
        ----------
        obsdate : str
            The observation time and date.
        event_type : str, optional
            The type of Event, by default 'AR'
        observatory : str, optional
            Observatory that observed the Event, by default 'SOHO'

        Returns
        -------
        results = sunpy.hek.HEKTable
            The table of results recieved from HEK.

        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter()
        >>> obsdate = '2000-01-01 12:47:02'
        >>> sunspotter.get_observations_from_hek(obsdate)
        <HEKTable length=5>
                 SOL_standard          absnetcurrenthelicity ... unsignedvertcurrent
                    str30                      object        ...        object
        ------------------------------ --------------------- ... -------------------
        SOL2000-01-01T09:35:02L054C117                  None ...                None
        SOL2000-01-01T09:35:02L058C100                  None ...                None
        SOL2000-01-01T09:35:02L333C106                  None ...                None
        SOL2000-01-01T09:35:02L033C066                  None ...                None
        SOL2000-01-01T09:35:02L012C054                  None ...                None
        """
        obsdate = self.get_nearest_observation(obsdate)

        client = hek.HEKClient()
        result = client.search(hek.attrs.Time(obsdate, obsdate), hek.attrs.EventType(event_type))

        obsdate = "T".join(str(obsdate).split())

        result = result[result['obs_observatory'] == 'SOHO']
        result = result[result['event_starttime'] <= obsdate]
        result = result[result['event_endtime'] > obsdate]

        return result

    def plot_observations(self, obsdate: str, mdi_map: Map = None):
        """
        Plots the Active Regions for a given observation on the
        MDI map corresponding to that observation.

        Parameters
        ----------
        obsdate : str
            The observation time and date.
        mdi_map : Map, optional
            The MDI map corresponding to the given observation,
            If None, the Map will be downloaded first.
            By default None.

        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter()
        >>> obsdate = '2000-01-01 12:47:02'
        >>> sunspotter.plot_observations(obsdate)
        """
        obsdate = self.get_nearest_observation(obsdate)

        if mdi_map is None:
            mdi_map = self.get_mdi_fulldisk_map(obsdate)

        hek_result = self.get_observations_from_hek(obsdate)

        bottom_left_x = hek_result['boundbox_c1ll']
        bottom_left_y = hek_result['boundbox_c2ll']
        top_right_x = hek_result['boundbox_c1ur']
        top_right_y = hek_result['boundbox_c2ur']

        number_of_observations = len(hek_result)

        bottom_left_coords = SkyCoord([(bottom_left_x[i], bottom_left_y[i]) * u.arcsec
                                      for i in range(number_of_observations)],
                                      frame=mdi_map.coordinate_frame)

        top_right_coords = SkyCoord([(top_right_x[i], top_right_y[i]) * u.arcsec
                                    for i in range(number_of_observations)],
                                    frame=mdi_map.coordinate_frame)

        fig = plt.figure(figsize=(12, 10), dpi=100)
        mdi_map.plot()

        for i in range(number_of_observations):
            mdi_map.draw_rectangle(bottom_left_coords[i],
                                   top_right=top_right_coords[i],
                                   color='b', label="Active Regions")

        hek_legend, = plt.plot([], color='b', label="Active Regions")

        plt.legend(handles=[hek_legend])
        plt.show()

    def rotate_to_midnight(self, obsdate: str, fmt='%Y-%m-%d %H:%M:%S'):
        """
        Returns the Longitude at midnight, for a given observation time and date.
        Parameters
        ----------
        obsdate : str
            The observation time and date.
        fmt : str, optional
            The format in which obsdate is represented, by default '%Y-%m-%d %H:%M:%S'
        Returns
        -------
        longitude : u.deg
            longitude of the observation at midnight.
        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter(timesfits="lookup_timefits.csv", properties="lookup_properties.csv")
        >>> obsdate = '2000-01-01 12:47:02'
        >>> sunspotter.rotate_to_midnight(obsdate)
        <Longitude [4.87918286] deg>
        """
        rotator = MidnightRotation()
        properties = self.get_properties_from_obsdate(obsdate)
        latitude = properties['hcpos_y'].values * u.deg
        time_to_nearest_midnight = rotator.get_seconds_to_nearest_midnight(obsdate) * u.s
        return rotator.get_longitude_at_nearest_midnight(time_to_nearest_midnight, latitude)

    def rotate_list_to_midnight(self, obslist: list, fmt='%Y-%m-%d %H:%M:%S'):
        """
        Returns list of Longitudes at midnight,
        for a given list of observation times and dates.
        Parameters
        ----------
        obslist : list
            List of observation times and dates.
        fmt : str, optional
            The format in which each obsdate is represented, by default '%Y-%m-%d %H:%M:%S'
        Returns
        -------
        longitudes : list of u.deg
            list of Longitudes at midnight, for the given list of observation times and dates.
        Examples
        --------
        >>> from pythia.seo import Sunspotter
        >>> sunspotter = Sunspotter(timesfits="lookup_timefits.csv", properties="lookup_properties.csv")
        >>> obslist = ['2000-01-02 12:51:02', '2000-01-14 12:47:02', '2000-01-18 12:51:02', '2000-01-23 12:47:02', '2000-01-24 12:51:02']
        >>> sunspotter.rotate_list_to_midnight(obslist)
        [<Longitude [4.8817556] deg>,
        <Longitude [5.85374373] deg>,
        <Longitude [5.15349168] deg>,
        <Longitude [5.28433062] deg>,
        <Longitude [4.82622275] deg>]
        """
        return [self.rotate_to_midnight(obsdate) for obsdate in obslist]
