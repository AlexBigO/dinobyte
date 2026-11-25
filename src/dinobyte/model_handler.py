"""
Module containing the class used for model manipulations (creation, training, testing, ...)
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
import keras_tuner as kt
from keras.models import Sequential, load_model  # , regularizers
from keras.layers import Input, Dense, Dropout
from keras.callbacks import EarlyStopping

from dinobyte.log_utils import Logger as LOG

ID_X_TRAIN = 0
ID_X_TEST = 1
ID_Y_TRAIN = 2
ID_Y_TEST = 3


# pylint:disable=too-many-instance-attributes
class ModelHandler:
    """
    Class used for wrapping models from Tensorflow-Keras
    """

    def __init__(self, hyperpars: dict, name_training_variables: list) -> None:
        """
        Init method of the class

        Args:
            hyperpars (dict): Hyperparameters configuration
                (the structure depends on whether regularisation will be used or not)
            name_training_variables (list): List of training variables names
        """

        self.name_training_variables = name_training_variables
        self.shape = (len(name_training_variables),)
        # create a Sequential model
        self.model: tf.keras.Model = Sequential()

        self.hyperpars = hyperpars
        self.do_hyperpar_optimisation: bool = False
        for _, hyperpar in self.hyperpars["layers"].items():
            for _, cfg in hyperpar.items():
                if isinstance(cfg, dict):
                    self.do_hyperpar_optimisation = True

        self.hyperpar_tuner = None
        self.early_stopping = None
        self.optim_hyperpars = None
        self.fit_res: tf.keras.callbacks.History | None = None

    @classmethod
    def apply_exported_model(cls, file_model: str, data: np.array) -> np.array:
        """
        Constructor of ModelHandler when one simply needs
        to import a model exported in a file a priori

        Args:
            file_model str: Path to the file containing the exported model
            data (np.array): Input data

        Returns:
            np.array: Output of the model
        """
        imported_model = tf.saved_model.load(file_model)
        return imported_model.serve(data)

    def build_model(self) -> None:
        """
        Method to build the model (without hyperparameter optimisation)
        """
        if self.do_hyperpar_optimisation:
            LOG(
                "The hyperparameters configuration set in the constructor"
                " is formatted for hyperparameter optimisation."
                " Please use 'set_hyperpar_tuner' instead.",
                "FATAL",
            )
        self.model = Sequential()
        self.model.add(Input(shape=self.shape))

        for key, hyperpars in self.hyperpars["layers"].items():
            if "Dense" in key:
                ConstructorLayer = Dense
            elif "Dropout" in key:
                ConstructorLayer = Dropout
            else:
                raise ValueError("Layer type must be 'Dense' or 'Dropout'")

            hyperpar_layer_cfg = {}

            for hyperpar, value in hyperpars.items():
                hyperpar_layer_cfg[hyperpar] = value

            self.model.add(ConstructorLayer(name=key, **hyperpar_layer_cfg))

        hyperpars_training_cfg = {}
        for hyperpar, value in self.hyperpars["training"].items():
            hyperpars_training_cfg[hyperpar] = value

        hyperpars_training_cfg["metrics"] = self.hyperpars["metrics"]

        self.model.compile(**hyperpars_training_cfg)

    # pylint:disable=too-many-branches
    def __build_model(self, hp: kt.HyperParameters) -> tf.keras.Model:
        """
        Method to build model for hyperparameter optimisation purposes

        Args:
            hp (kt.HyperParameters): Hyperparameters

        Returns:
            tf.keras.Model: Model configured with hyperparameters configuration for optimisation
        """
        model: tf.keras.Model = Sequential()
        model.add(Input(shape=self.shape))

        for key, hyperpars in self.hyperpars["layers"].items():
            if "Dense" in key:
                ConstructorLayer = Dense
            elif "Dropout" in key:
                ConstructorLayer = Dropout
            else:
                raise ValueError("Layer type must be 'Input', 'Dense' or 'Dropout'")

            hyperpar_layer_cfg = {}

            for hyperpar, cfg in hyperpars.items():
                if cfg["type"] == "Int":
                    hp_method = hp.Int
                elif cfg["type"] == "Float":
                    hp_method = hp.Float
                elif cfg["type"] == "Choice":
                    hp_method = hp.Choice
                elif cfg["type"] == "Fixed":
                    hp_method = hp.Fixed
                else:
                    raise ValueError(
                        "The hyperparameter 'type' must be: Int, Float, Choice, Fixed or Boolean"
                    )

                cfg_copy = cfg.copy()
                cfg_copy.pop("type")
                cfg_copy["name"] = hyperpar + "_" + key
                hyperpar_layer_cfg[hyperpar] = hp_method(**cfg_copy)

                LOG(f"cfg: {cfg_copy}", "DEBUG")

            model.add(ConstructorLayer(name=key, **hyperpar_layer_cfg))

        hyperpars_training_cfg = {}
        for hyperpar, cfg in self.hyperpars["training"].items():
            if cfg["type"] == "Int":
                hp_method = hp.Int
            elif cfg["type"] == "Float":
                hp_method = hp.Float
            elif cfg["type"] == "Choice":
                hp_method = hp.Choice
            elif cfg["type"] == "Fixed":
                hp_method = hp.Fixed
            else:
                raise ValueError(
                    "The hyperparameter 'type' must be: Int, Float, Choice, Fixed or Boolean"
                )
            cfg_copy = cfg.copy()
            cfg_copy.pop("type")
            cfg_copy["name"] = hyperpar
            hyperpars_training_cfg[hyperpar] = hp_method(**cfg_copy)

        hyperpars_training_cfg["metrics"] = self.hyperpars["metrics"]

        model.compile(**hyperpars_training_cfg)

        return model

    def set_early_stopping(self, monitor="loss", patience=5, **kwargs) -> None:
        """
        Method to set early stopping

        Args:
            monitor (str, optional): _description_. Defaults to "loss".
            patience (int, optional): _description_. Defaults to 5.
        """
        self.early_stopping = EarlyStopping(monitor=monitor, patience=patience, **kwargs)

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def set_hyperpar_tuner(
        self,
        objective: str = "val_loss",
        max_trials: int = 3,
        directory: str = "dino_hp_tuning",
        name_project: str = "my_project",
        overwrite: bool = False,
    ) -> None:
        """
        Method to set the keras-tuner instance

        Args:
            objective (str, optional): Reference metric for the tuning. Defaults to "val_loss".
            max_trials (int, optional): Maximum number of trials. Defaults to 3.
            directory (str, optional): Name of the directory to store the project.
                Defaults to "dino_hp_tuning".
            name_project (str, optional): Name of the folder containing the tuner
                output within directory. Defaults to "my_project".
            overwrite (bool, optional): Resume the last tuning (saved in project) if True
                else perform new tuning. Defaults to False.
        """
        if not self.do_hyperpar_optimisation:
            LOG(
                "The hyperparameters configuration set in the constructor"
                "is not formatted for hyperparameter optimisation."
                " Please use the regular workflow instead.",
                "FATAL",
            )
        self.hyperpar_tuner = kt.RandomSearch(
            self.__build_model,
            objective=objective,
            max_trials=max_trials,
            executions_per_trial=1,
            directory=directory,
            project_name=name_project,
            overwrite=overwrite,
        )

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def optimise_hyperpars(
        self,
        x_train: np.array,
        y_train: np.array,
        epochs: int = 5,
        validation_split: float = 0.2,
        verbose: int = 0,
    ) -> list[kt.HyperParameters]:
        """
        Method to run the hyperparameter optimisation on training data set
        and obtain the optimal set of hyperparameters

        Args:
            x_train (np.array): Training data
            y_train (np.array): Labels of training data
            epochs (int, optional): Number of epochs. Defaults to 5.
            validation_split (float, optional): Fraction of data set dedicated to cross-validation.
                Defaults to 0.2.
            verbose (int, optional): Verbosity level. Defaults to 0.

        Returns:
            list[kt.HyperParameters]: Optimal set of hyperparameters


        """
        if self.hyperpar_tuner is None:
            LOG("'set_hyperpar_tuner' must be called before calling this method.", "FATAL")
        args = (x_train, y_train)
        kwargs = {"epochs": epochs, "validation_split": validation_split, "verbose": verbose}
        if self.early_stopping is not None:
            kwargs["callbacks"] = [self.early_stopping]
        self.hyperpar_tuner.search(*args, **kwargs)

        self.optim_hyperpars = self.hyperpar_tuner.get_best_hyperparameters(num_trials=1)[0]
        return self.optim_hyperpars

    def build_optimised_model(self) -> None:
        """
        Method to build the model with optimal set of hyperparameters obtain a priori
        """
        if not self.do_hyperpar_optimisation:
            LOG(
                "Hyperparameter optimisation needs to be performed"
                "before calling 'build_optimised_model' method",
                "FATAL",
            )
        self.model = self.hyperpar_tuner.hypermodel.build(self.optim_hyperpars)

    def get_optimal_epochs(
        self, metric: str = "val_accuracy", direction: str = "max"
    ) -> int | None:
        """
        Method to get the optimal number of epochs for training

        Args:
            metric (str, optional): Metric of reference. Defaults to "val_accuracy".
            direction (str, optional): Direction ('min' or 'max'). Defaults to "max".

        Returns:
            int: Number of epochs to optimise the metric
        """
        if self.fit_res is None:
            LOG("train_test_model needs to be run before calling get_optimal_epochs", "FATAL")
        val_per_epoch = self.fit_res.history[metric]

        if "min" in direction:
            return val_per_epoch.index(min(val_per_epoch)) + 1
        if "max" in direction:
            return val_per_epoch.index(max(val_per_epoch)) + 1

        LOG("'direction' must be 'min' or 'max' in get_optimal_epochs", "FATAL")
        return None

    def summary(self) -> None:
        """
        Method to print a summary of the model
        """
        self.model.summary()  # TODO test w/o return

    def __fit(self, data: tuple, **kwargs) -> tf.keras.callbacks.History:
        """
        Helper method to fit the model

        Args:
            data (tuple): Data set for training and testing

        Returns:
            tf.keras.callbacks.History: History of the fitted model
        """
        kwds = {"validation_data": (data[ID_X_TEST], data[ID_Y_TEST])}
        if self.early_stopping is not None:
            kwds["callbacks"] = [self.early_stopping]
        return self.model.fit(
            data[ID_X_TRAIN],
            data[ID_Y_TRAIN],
            **kwds,
            **kwargs,
        )

    def train_test_model(
        self, data: tuple, return_prediction: bool = False, **kwargs
    ) -> tuple[tf.keras.callbacks.History, np.array]:
        """
        Method to train and test the model.

        Args:
            data (tuple): Data set for training and testing
            return_prediction (bool, optional): Returns the predictions
                for the test data if enabled. Defaults to False.

        Returns:
            None | np.array: Predictions for the test data if option enabled, else None
        """
        self.fit_res = self.__fit(data, **kwargs)
        ypred = None
        if return_prediction:
            ypred = self.model.predict(data[ID_X_TEST])
        return self.fit_res, ypred

    # TODO add method to dump history into another file

    def save(self, filepath: str) -> None:
        """
        Method to save the model

        Args:
            filepath (str): Path to the output file
        """
        self.model.save(filepath)

    def export(self, filepath: str) -> None:
        """
        Method to export the model (for inference purpose only)

        Args:
            filepath (str): Path to the output file
        """
        self.model.export(filepath)

    def load_saved_model(self, file_model: str) -> tf.keras.Model:
        """
        Method to load a model that was saved

        Args:
            file_model (str): Path to input file

        Returns:
            tf.keras.Model: Loaded model
        """
        self.model = load_model(file_model)
        return self.model

    def evaluate(self, x: np.array, y: np.array, **kwargs) -> dict:
        """
        Method to evaluate the trained model on a given dataset

        Args:
            x (np.array): Input data
            y (np.array): Target data

        Returns:
            dict: Loss and other metrics set in compile parameters of the model
        """
        return self.model.evaluate(x, y, return_dict=True, **kwargs)

    def get_fit_res(self) -> tf.keras.callbacks.History:
        """
        Method to get the output of the model fit (i.e. training)

        Returns:
            tf.keras.callbacks.History: Output of model fit
        """
        return self.fit_res

    def predict(self, x: np.array) -> np.array:
        """
        Method to pass data through the (trained) model

        Args:
            x (np.array): Input data

        Returns:
            np.array: Model predictions
        """
        return self.model.predict(x)  # , **kwargs)

    def get_model(self) -> tf.keras.Model:
        """
        Method to get the model

        Returns:
            tf.keras.Model: the model
        """
        return self.model
