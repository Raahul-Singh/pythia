import warnings

import numpy as np
from sunpy.util import SunpyUserWarning

__all__ = ['TableMatcher']


class TableMatcher:
    """
    Table Matcher Object for finding corresponding rows across two distinct dataframes.
    """
    def __init__(self, match_type='cosine'):
        """
        Parameters
        ----------
        match_type : str, optional
            The row matching algorithm, by default 'cosine'

        Raises
        ------
        SunpyUserWarning
            If unrecognized match type is passed.
        """
        self.match_type = match_type

        if self.match_type is not 'cosine' and self.match_type != 'euclidean':
            raise SunpyUserWarning('Incorrect matching algorithm specified.')

    def _prepare_tables(self, df_1, df_2, feature_1=None, feature_2=None):
        """
        Prepares tables for matching.
        Left match on df_1 and df_2

        Parameters
        ----------
        df_1 : pd.DataFrame
            Dataframe that will be used for matching.
        df_2 : pd.DataFrame
            Dataframe that will be used for matching.
        feature_1 : str, by default None
            List of features of df_1 that will be used for matching.
            None indicates all features will be used.
        feature_2 : str, by default None
            List of features of df_2 that will be used for matching.
            None indicates all features will be used.

        Returns
        -------
        df_1 : pd.DataFrame
            df_1 with only those columns that will be used for matching.
        df_2 : pd.DataFrame
            df_2 with only those columns that will be used for matching.

        Raises
        ------
        SunpyUserWarning
            If number of features for df_1 is not equal to
            number of features for df_2.
        SunpyUserWarning
            if key from feature_1 is not present in df_1
        SunpyUserWarning
            if key from feature_2 is not present in df_2
        """
        if feature_1 is None:
            feature_1 = df_1.columns.values

        if feature_2 is None:
            feature_2 = df_2.columns.values

        if len(feature_1) != len(feature_2):
            raise SunpyUserWarning("The number of columns to match the rows on must be the same.")

        try:
            df_1 = df_1[feature_1]
        except KeyError:
            raise SunpyUserWarning("The features specified for table 1 do not "
                                   "correspond to any columns in table 1.")
        try:
            df_2 = df_2[feature_2]
        except KeyError:
            raise SunpyUserWarning("The features specified for table 2 do not "
                                   "correspond to any columns in table 2.")

        return df_1, df_2

    def match_cosine(self, df_1, df_2):
        """
        Finds Cosine similarity between the rows of the two dataframes.
        Parameters
        ----------
        df_1: `pd.DataFrame`
            First DataFrame to match the rows from.
        df_2: `pd.DataFrame`
            Second DataFrame to match the rows from.
            Array of size `(n,)` where n is the number of rows in df_1.
        match_score: `numpy.ndarray`
            Array of size `(n,)` where n is the number of rows in df_1.
            Contains match score for  corresponding best matches.
        """

        try:
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            raise SunpyUserWarning("Table Matcher requires Scikit Learn to be installed")

        cosine = cosine_similarity(X=df_1, Y=df_2)
        result = np.argmax(cosine, axis=1)
        match_score = np.max(cosine, axis=1)

        return result, match_score

    def match_euclidean(self, df_1, df_2):
        """
        Finds euclidean distance between the rows of the two dataframes.
        Parameters
        ----------
        df_1: `pd.DataFrame`
            First DataFrame to match the rows from.
        df_2: `pd.DataFrame`
            Second DataFrame to match the rows from.
        Returns
        -------
        result: `numpy.ndarray`
            Array of size `(n,)` where n is the number of rows in df_1.
            Contains indices of rows from df_2 that best correspond to rows from df_1.
        match_score: `numpy.ndarray`
            Array of size `(n,)` where n is the number of rows in df_1.
            Contains match score for  corresponding best matches.
        """
        try:
            from sklearn.metrics.pairwise import euclidean_distances
        except ImportError:
            raise SunpyUserWarning("Table Matcher requires Scikit Learn to be installed")

        euclidean = euclidean_distances(X=df_1, Y=df_2)
        result = np.argmin(euclidean, axis=1)
        match_score = np.min(euclidean, axis=1)

        return result, match_score

    def verify(self, match_score, threshold):
        """
        Verify matching quality. If any match score is less than the threshold,
        raises Sunpy User Warnings.
        Parameters
        ----------
        match_score: `numpy.ndarray`
            Array of size `(n,)` where n is the number of rows in df_1.
            Contains match score for  corresponding best matches.
        threshold: `float`
            Minimum score for considering a proper match.
        """
        if self.match_type == 'euclidean':
            for index, score_value in enumerate(match_score):
                if score_value > threshold:
                    warnings.warn(SunpyUserWarning(f"\nMatch at Index {index} is likely to be incorrect\n"))

        if self.match_type == 'cosine':
            for index, score_value in enumerate(match_score):
                if score_value < threshold:
                    warnings.warn(SunpyUserWarning(f"\nMatch at Index {index} is likely to be incorrect\n"))

    def match(self, df_1, df_2, feature_1=None, feature_2=None, threshold=5):
        """
        Finds best match between the rows of the two dataframes.
        Raises warning id matching is dubious.
        Parameters
        ----------
        feature_1: `list`
            List of columns from df_1 to match the rows with.
        feature_2: `list`
            List of columns from df_2 to match the rows with.
        threshold: `float`
            Minimum score for considering a proper match.
        Returns
        -------
        result: `numpy.ndarray`
            Array of size `(n,)` where n is the number of rows in df_1.
            Contains indices of rows from df_2 that best correspond to rows from df_1.
        """
        df_1, df_2 = self._prepare_tables(df_1, df_2, feature_1, feature_2)

        if self.match_type == 'cosine':
            result, match_score = self.match_cosine(df_1, df_2)
        elif self.match_type == 'euclidean':
            result, match_score = self.match_euclidean(df_1, df_2)

        self.verify(match_score, threshold)

        return result
