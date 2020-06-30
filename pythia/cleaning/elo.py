from collections import deque

import numpy as np
import pandas as pd
from sunpy.util import SunpyUserWarning

__all__ = ['ELO']


class ELO:
    """
    Recreating the ELO rating algirithm for Sunspotter.
    """

    def __init__(self, score_board: pd.DataFrame, *, k_value=32, default_score=1400,
                 max_comparisons=50, max_score_change=32, min_score_change=16, score_memory=10,
                 delimiter=';', column_map={"player 0": "image_id_0",
                                            "player 1": "image_id_1",
                                            "score for player 0": "image0_more_complex_image1"}):
        """
        Parameters
        ----------
        score_board : pandas.DataFrame
            DataFrame holding the scores of individual matches.
        k_value : int, optional
            Initial K Value to be used for calculating new ratings, by default 32
        default_score : int, optional
            Initial rating, by default 1400
        max_comparisons : int, optional
            Max comparisions for any player, by default 50
        max_score_change : int, optional
            Upper limit on K Value updation, by default 32
        min_score_change : int, optional
            Lower limit on K Value updation, by default 16
        score_memory : int, optional
            Number of previous scores to consider while calculating
            standard deviation and new K value, by default 10
        column_map : dict, optional
            Dictionary, for mapping the column names of the score_board dataframe
            to variable names used in the ELO ranking system.
            by default {"player 0": "image_id_0",
                                    "player 1": "image_id_1",
                                    "score for player 0": "image0_more_complex_image1"}
        """
        self.score_board = score_board
        self.k_value = k_value
        self.default_score = default_score
        self.score_change = {'min': min_score_change, 'max': max_score_change}
        self.max_comparisions = max_comparisons
        self.score_memory = score_memory
        self.column_map = column_map

        if not set(self.column_map.values()).issubset(self.score_board.columns):
            missing_columns = set(self.column_map.values()) - set(self.column_map.values()).intersection(self.score_board.columns)
            missing_columns = ", ".join(missing_columns)

            raise SunpyUserWarning("The following columns mentioned in the column map"
                                   f" are not present in the score board: {missing_columns}")

        self._create_ranking()

    def _create_ranking(self):
        """
        Prepares the Ranking DataFrame.
        """
        image_ids = set(self.score_board[self.column_map['player 0']]).union(self.score_board[self.column_map['player 1']])
        self.rankings = pd.DataFrame(image_ids, columns=['player id'])
        self.rankings.set_axis(self.rankings['player id'], inplace=True)
        self.rankings['score'] = self.default_score
        self.rankings['k value'] = self.k_value
        self.rankings['count'] = 0
        self.rankings['std dev'] = self.score_change['max']
        self.rankings['last scores'] = str(self.default_score)

    def expected_score(self, score_image_0, score_image_1):
        """
        Given two AR scores, calculates expected probability of `image_0` being more complex.

        Parameters
        ----------
        score_image_0 : int
            Score for first image
        score_image_1 : int
            Score for second image

        Returns
        -------
        expected_0_score : float
            Expected probability of `image_0` being more complex.
        """
        expected_0_score = 1.0 / (1.0 + 10 ** ((score_image_1 - score_image_0) / 400.00))
        return expected_0_score

    def new_rating(self, rating_for_image, k_value, score_for_image, image_expected_score):
        """
        Calculates new rating based on the ELO algorithm.

        Parameters
        ----------
        rating_for_image : float
            Current Rating for the image
        k_value : float
            Current k_value for the image
        score_for_image : int
            Actual result of classification of the image in a pairwise match.
            `0` denotes less complex, `1` denotes more complex
        image_expected_score : float
            Expected result of classification of image in a pairwise match
            based on current rating of the image.

        Returns
        -------
        new_image_rating : float
            New rating of image after the classification match.
        """
        new_image_rating = rating_for_image + k_value * (score_for_image - image_expected_score)
        return new_image_rating

    def score_update(self, image_0, image_1, score_for_image_0):
        """
        Updates the ratings of the two images based on the complexity classification.

        Parameters
        ----------
        image_0 : int
            Image id for first image
        image_1 : int
            Image id for second image
        score_for_image_0 : int
            Actual result of classification of the image 0 in a pairwise match.
            `0` denotes less complex, `1` denotes more complex

        Notes
        -----
        To make updates in the original rankings DataFrame, for each classification,
        two state dictionaries need to be maintained, corresponfing to the two AR images.
        The changes are made to these state dictionaries and then the ranking DataFrame is updated.
        """
        # state dicts
        state_dict_0 = self.rankings.loc[image_0].to_dict()
        state_dict_0['last scores'] = deque(map(float, state_dict_0['last scores'].split(',')), maxlen=self.score_memory)
        state_dict_1 = self.rankings.loc[image_1].to_dict()
        state_dict_1['last scores'] = deque(map(float, state_dict_1['last scores'].split(',')), maxlen=self.score_memory)

        expected_score_0 = self.expected_score(self.rankings.loc[image_0]['score'],
                                               self.rankings.loc[image_1]['score'])
        expected_score_1 = 1 - expected_score_0

        _update_state_dict(state_dict_0, image_0, expected_score_0, score_for_image_0)
        _update_state_dict(state_dict_1, image_1, expected_score_1, 1 - score_for_image_0)

        # Making the Update DataFrames
        update_df = pd.DataFrame([state_dict_0, state_dict_1])
        update_df.set_index("player id", inplace=True)

        # Updating the original DataFrame
        self.rankings.update(update_df)

    def _update_state_dict(state_dict, image, expected_score, score):
        new_rating = self.new_rating(self.rankings.loc[image]['score'], self.rankings.loc[image]['k value'],
                                       score, expected_score)
        state_dict['last scores'].append(new_rating)
        new_std_dev = min(np.std(state_dict['last scores']), 1_000_000)  # prevents Infinity
        new_k = min(max(new_std_dev, self.score_change['min']), self.score_change['max'])
        # Updating Data
        state_dict['score'] = new_rating
        state_dict['std dev'] = new_std_dev
        state_dict['k value'] = new_k
        state_dict['count'] += 1
        state_dict['last scores'] = ",".join(map(str, state_dict['last scores']))  # Storing the list of states as a String

    def run(self, save_to_disk=True, filename='run_results.csv'):
        """
        Runs the ELO ranking Algorithm for all score_board.

        Parameters
        ----------
        save_to_disk : bool, optional
            If true, saves the rankins in a CSV file on the disk, by default True
        filename : str, optional
            filename to store the results, by default 'run_results.csv'
        """
        for index, row in self.score_board.iterrows():

            if row[self.column_map['player 0']] == row[self.column_map['player 1']]:
                continue

            self.score_update(image_0=row[self.column_map['player 0']], image_1=row[self.column_map['player 1']],
                              score_for_image_0=row[self.column_map['score for player 0']])
            print(f"Index {index} done!")

        if save_to_disk:
            self.save_as_csv(filename)

    def save_as_csv(self, filename):
        """
        Saves the Ranking DataFrame to the disk as a CSV file.

        Parameters
        ----------
        filename : str
            filename to store the results.
        """
        self.rankings.drop(columns=["last_scores"], inplace=True)
        self.rankings.to_csv(filename)
