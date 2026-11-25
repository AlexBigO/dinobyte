"""
Module containing the class used to import data, convert them in the right format
and split data sets for training and testing of ML model
"""

from sklearn.model_selection import train_test_split
import numpy as np


class DataHandler:
    """
    Class ...
    """

    # TODO: add name_file argument and convert it to pd df
    # according to its type
    def __init__(self, df, label, training_columns=None):
        """Helper init"""
        # remove label column from data if training_columns is None
        df_x = df.drop(label, axis=1) if training_columns is None else df[training_columns]
        df_y = df[label]

        # convert pd.DataFrame to numpy arrays
        self.x = df_x.to_numpy()  # values
        self.y = df_y.to_numpy()  # values

        self.train_test_data = None

    def get_train_test_data(self, test_size: float = 0.3, random_state: int = 42):
        """
        _summary_

        Args:
            test_size (float, optional): _description_. Defaults to 0.3.
            random_state (int, optional): _description_. Defaults to 42.

        Returns:
            tuple[4]: _description_

        Example:
        >>> print('Hello')
        """
        self.train_test_data = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state
        )
        return self.train_test_data

    def get_x(self):
        """
        _summary_

        Returns:
            np.array: _description_
        """
        return self.x

    def get_x_train(self):
        """
        _summary_

        Returns:
            np.array: _description_
        """
        return self.train_test_data[0]

    def get_x_test(self):
        """
        _summary_

        Returns:
            np.array: _description_
        """
        return self.train_test_data[1]

    def get_y(self):
        """
        _summary_

        Returns:
            np.array: _description_
        """
        return self.y

    def get_y_train(self):
        """
        _summary_

        Returns:
            np.array: _description_
        """
        return self.train_test_data[2]

    def get_y_test(self):
        """
        _summary_

        Returns:
            np.array: _description_
        """
        return self.train_test_data[3]
