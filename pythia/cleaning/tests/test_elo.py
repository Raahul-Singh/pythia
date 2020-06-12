import pytest
import pandas as pd
from pythia.cleaning import ELO
from pathlib import Path
from pythia.seo import Sunspotter
from sunpy.util import SunpyUserWarning

path = Path(__file__).resolve().parent.parent.parent.parent / "data/elo_test"


@pytest.fixture
def elo():
    sunspotter = Sunspotter(timesfits=path / "../all_clear/lookup_timesfits.csv",
                            properties=path / "../all_clear/lookup_properties.csv",
                            classifications=path / "test_classifications.csv",
                            classifications_columns=['image_id_0', 'image_id_1',
                                                     'image0_more_complex_image1'])
    column_map = {"player 0": "image_id_0",
                  "player 1": "image_id_1",
                  "score for player 0": "image0_more_complex_image1"}

    return ELO(score_board=sunspotter.classifications, column_map=column_map)


@pytest.mark.parametrize('rating_0,rating_1,expected_score',
                         [(1400, 1400.0, 0.5),
                          (1450, 1450.5, 0.49928044265518673),
                          (1500, 1602.0, 0.3572869311673796),
                          (1550, 1854.5, 0.14768898365874825),
                          (1600, 2208.0, 0.029314241270450396)])
def test_expected_score(elo, rating_0, rating_1, expected_score):
    assert elo.expected_score(rating_0, rating_1) == expected_score


@pytest.mark.parametrize('rating_for_image,k_value,score_for_image,image_expected_score,new_rating',
                         [(1400.0, 32, 1, 0.5, 1416.0),
                          (1450.0, 32, 0, 0.49928044265518673, 1434.023025835034),
                          (1500.0, 32, 0, 0.3572869311673796, 1488.5668182026438),
                          (1550.0, 32, 1, 0.14768898365874825, 1577.2739525229201),
                          (1600.0, 32, 1, 0.029314241270450396, 1631.0619442793457),
                          (1400.0, 32, 0, 0.5, 1384.0),
                          (1450.5, 32, 1, 0.5007195573448133, 1466.476974164966),
                          (1602.0, 32, 1, 0.6427130688326204, 1613.4331817973562),
                          (1854.5, 32, 0, 0.8523110163412517, 1827.2260474770799),
                          (2208.0, 32, 0, 0.9706857587295497, 2176.9380557206546)])
def test_new_rating(elo, rating_for_image, k_value, score_for_image, image_expected_score, new_rating):
    assert elo.new_rating(rating_for_image, k_value, score_for_image, image_expected_score) == new_rating


def test_column_map(elo):
    assert set(elo.column_map.values()).issubset(elo.score_board.columns)


def test_incorrect_column_map():

    sunspotter = Sunspotter(timesfits=path / "../all_clear/lookup_timesfits.csv",
                            properties=path / "../all_clear/lookup_properties.csv",
                            classifications=path / "test_classifications.csv",
                            classifications_columns=['image_id_0', 'image_id_1',
                                                     'image0_more_complex_image1'])
    column_map = {"player 0": "This is not player 0",
                  "player 1": "This is not player 1",
                  "score for player 0": "Player 0 is in it for the fun"}

    with pytest.raises(SunpyUserWarning):
        ELO(score_board=sunspotter.classifications, column_map=column_map)
