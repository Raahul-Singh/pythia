import pytest
import pandas as pd
from pathlib import Path
from pythia.seo.sunspotter import Sunspotter
from sunpy.util import SunpyUserWarning


path = Path(__file__).resolve().parent.parent.parent.parent / "data/all_clear"


@pytest.fixture
def properties_columns():
    return ['#id', 'filename', 'zooniverse_id', 'angle', 'area', 'areafrac',
            'areathesh', 'bipolesep', 'c1flr24hr', 'id_filename', 'flux',
            'fluxfrac', 'hale', 'hcpos_x', 'hcpos_y', 'm1flr12hr', 'm5flr12hr',
            'n_nar', 'noaa', 'pxpos_x', 'pxpos_y', 'sszn', 'zurich']


@pytest.fixture
def properties():
    # Properties corresponding to `obs_date` '2000-01-01 12:47:02' and `#id` 1
    data = [[1, '530be1183ae74079c300000d.jpg', 'ASZ000090y', 37.8021,
             34400.0, 0.12, 2890.0, 3.72, 0, 1, 2.18e+22, 0.01, 'beta', 452.26991,
             443.92976, 0, 0, 1, 8809, 229.19343999999998, 166.877, 1, 'bxo']]

    columns = ['#id', 'filename', 'zooniverse_id', 'angle', 'area', 'areafrac',
               'areathesh', 'bipolesep', 'c1flr24hr', 'id_filename', 'flux',
               'fluxfrac', 'hale', 'hcpos_x', 'hcpos_y', 'm1flr12hr', 'm5flr12hr',
               'n_nar', 'noaa', 'pxpos_x', 'pxpos_y', 'sszn', 'zurich']

    return pd.DataFrame(data=data, columns=columns)


@pytest.fixture
def timefits_columns():
    return ['#id', 'filename', 'obs_date']


@pytest.fixture
def sunspotter(properties_columns, timefits_columns):
    return Sunspotter(timefits=path / "lookup_timesfits.csv",
                      properties=path / "lookup_properties.csv",
                      delimiter=';',
                      timefits_columns=timefits_columns,
                      properties_columns=properties_columns)


@pytest.fixture
def obsdate():
    # `obs_date` corresponding to `#id` 1
    return '2000-01-01 12:47:02'


def test_sunspotter_base_object(properties_columns, timefits_columns):

    sunspotter = Sunspotter(timefits=path / "lookup_timesfits.csv",
                            properties=path / "lookup_properties.csv",
                            timefits_columns=timefits_columns,
                            properties_columns=properties_columns)

    assert set(sunspotter.properties_columns) == set(properties_columns)
    assert set(sunspotter.timefits_columns) == set(timefits_columns)


def test_sunspotter_incorrect_delimiter():

    with pytest.raises(SunpyUserWarning):
        Sunspotter(timefits=path / "lookup_timesfits.csv",
                   properties=path / "lookup_properties.csv",
                   delimiter=',')


def test_sunspotter_properties_columns():

    with pytest.raises(SunpyUserWarning):
        Sunspotter(timefits=path / "lookup_timesfits.csv",
                   properties=path / "lookup_properties.csv",
                   properties_columns=["This shouldn't be present"],
                   delimiter=';')


def test_sunspotter_timesfits_columns():

    with pytest.raises(SunpyUserWarning):
        Sunspotter(timefits=path / "lookup_timesfits.csv",
                   properties=path / "lookup_properties.csv",
                   timefits_columns=["This shouldn't be present"],
                   delimiter=';')


def test_get_timefits_id(sunspotter, obsdate):
    assert sunspotter.get_timefits_id(obsdate) == 1


def test_get_properties(sunspotter, properties):
    assert sunspotter.get_properties(1).equals(properties)


def test_get_properties_from_obsdate(sunspotter, obsdate, properties):
    assert sunspotter.get_properties_from_obsdate(obsdate).equals(properties)
