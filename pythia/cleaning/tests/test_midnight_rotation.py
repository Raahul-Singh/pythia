import datetime
from pathlib import Path

import pytest
from pythia.cleaning import MidnightRotation
from pythia.seo import Sunspotter

path = Path(__file__).resolve().parent.parent.parent.parent / "data/all_clear"


@pytest.fixture
def midnight_rotation():
    return MidnightRotation()


@pytest.fixture
def sunspotter(properties_columns, timesfits_columns):
    return Sunspotter(timesfits=path / "lookup_timesfits.csv",
                      properties=path / "lookup_properties.csv",
                      delimiter=';',
                      timesfits_columns=timesfits_columns,
                      properties_columns=properties_columns)




@pytest.mark.parametrize('obsdate,nearest_midnight',
                         [('2000-02-29 12:51:02', datetime.datetime(2000, 3, 1, 0, 0)),
                          ('2000-01-01 11:47:02', datetime.datetime(2000, 1, 1, 0, 0)),
                          ('2000-01-24 12:00:00', datetime.datetime(2000, 1, 25, 0, 0)),
                          ('2010-01-18 10:51:02', datetime.datetime(2010, 1, 18, 0, 0)),
                          ('2000-03-31 12:47:02', datetime.datetime(2000, 4, 1, 0, 0)),
                          ('2000-11-25 12:51:02', datetime.datetime(2000, 11, 26, 0, 0))])
def test_nearest_midnight(midnight_rotation, obsdate, nearest_midnight):
    assert midnight_rotation.get_nearest_midnight(obsdate) == nearest_midnight


@pytest.mark.parametrize('obsdate,seconds_to_nearest_midnight',
                         [('2000-02-28 12:51:02', 40138),
                          ('2000-01-01 11:47:02', -42422),
                          ('2000-01-24 12:00:00', 43200),
                          ('2010-01-18 10:51:02', -39062),
                          ('2000-03-31 12:47:02', 40378),
                          ('2000-11-25 12:51:02', 40138)])
def test_seconds_to_nearest_midnight(midnight_rotation, obsdate, seconds_to_nearest_midnight):
    assert midnight_rotation.get_seconds_to_nearest_midnight(obsdate) == seconds_to_nearest_midnight
