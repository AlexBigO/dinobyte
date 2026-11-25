"""
file: test_iris.py
brief: Simple script to test the library on the iris database
usage: python3 test_iris.py
note:
author: Alexandre BIGOT, alexandre.bigot@iphc.cnrs.fr
"""

import pandas as pd
import tensorflow as tf
import numpy as np

import keras_tuner as kt

from dinobyte.data_handler import DataHandler
from dinobyte.model_handler import ModelHandler
from dinobyte.plot_utils import plot_history

from sklearn.metrics import confusion_matrix, classification_report

ID_X_TRAIN = 0
ID_X_TEST = 1
ID_Y_TRAIN = 2
ID_Y_TEST = 3


def main(debug: bool) -> None:
    """
    Main function

    Parameters
    ------------------------------------------------
    - debug: bool
        Switch for debugging
    """
    if debug:
        print("Debug mode activated!")

    url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv"
    df = pd.read_csv(url)
    print(df.head())

    label = "species"

    ID_SETOSA = 0.0
    ID_VERSICOLOR = 1.0
    ID_VIRGINICA = 2.0

    df[label] = df[label].replace("setosa", ID_SETOSA)
    df[label] = df[label].replace("versicolor", ID_VERSICOLOR)
    df[label] = df[label].replace("virginica", ID_VIRGINICA)

    data_handler = DataHandler(df, label)

    train_test_data = data_handler.get_train_test_data()

    train_test_data[2] = tf.keras.utils.to_categorical(train_test_data[2], 3)
    train_test_data[3] = tf.keras.utils.to_categorical(train_test_data[3], 3)

    # print(f"data: {train_test_data}")

    # params_model = {
    #     "shape": (4,),
    #     "n_neurons": [
    #         None,
    #         16,
    #         None,
    #         32,
    #         1,
    #     ],
    #     "name_layers": [
    #         "InputLayer",
    #         "DenseLayer",
    #         "DropoutLayer",
    #         "DenseLayer",
    #         "OutputLayer",
    #     ],
    #     "name_activs": [
    #         None,
    #         "relu",
    #         None,
    #         "relu",
    #         "softmax",
    #     ],
    #     "types": [
    #         "Input",
    #         "Dense",
    #         "Dropout",
    #         "Dense",
    #         "Dense",
    #     ],
    # }

    # params_algo = None

    hyperpars = {
        "layers": {
            "Dense_1": {
                "units": {
                    "type": "Int",
                    "min_value": 50,
                    "max_value": 600,
                    "step": 20,
                    "default": 100,
                },
                "activation": {"type": "Fixed", "value": "relu"},
            },
            "Dropout_2": {"rate": {"type": "Fixed", "value": 0.5}},
            # "Dense_3": {
            #     "units": {
            #         "type": "Int",
            #         "min_value": 50,
            #         "max_value": 600,
            #         "step": 20,
            #     },
            #     "activation": {"type": "Fixed", "value": "relu"},
            # },
            "Dense_4": {  # output
                "units": {"type": "Fixed", "value": 3},
                "activation": {"type": "Fixed", "value": "softmax"},
            },
        },
        "training": {
            "optimizer": {"type": "Fixed", "value": "adam"},
            "loss": {"type": "Fixed", "value": "categorical_crossentropy"},
        },
        "metrics": ["accuracy", "mae", "mse"],
    }

    # hyperpars = {
    #     "layers": {
    #         "Dense_1": {"units": 200, "activation": "relu"},
    #         "Dropout_2": {"rate": 0.5},
    #         "Dense_3": {"units": 100, "activation": "relu"},
    #         "Dense_4": {"units": 3, "activation": "softmax"},  # output
    #     },
    #     "training": {"optimizer": "adam", "loss": "categorical_crossentropy"},
    #     "metrics": ["accuracy", "mae", "mse"],
    # }

    # hyperpars_opt = {
    #     "activate": True,
    #     "layers": {
    #         "Dense_2": {
    #             "units": {"type": "Int", "name": "units_layer2", "min_value": 50, "max_value": 600, "step": 20}
    #         },
    #         "Dense_4": {
    #             "units": {"type": "Int", "name": "units_layer4", "min_value": 50, "max_value": 600, "step": 20}
    #         },
    #     },
    #     "training": {
    #         "loss": {"type": "Choice", "name": "loss", ""}
    #     }
    # }

    # handle use of GPU
    use_gpu: bool = True
    if use_gpu:
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        n_gpu = len(physical_devices)
        print(f"Number of GPUs available: {n_gpu}")
        if n_gpu > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

    training_columns = df.columns.to_list()
    training_columns.remove("species")
    # print(training_columns)

    model = ModelHandler(hyperpars, training_columns)  # params_model, params_algo)
    # model.summary()

    # model.build_model()
    # model.summary()
    # model.set_early_stopping()

    # fit_res, ypreds = model.train_test_model(
    #     train_test_data, return_prediction=True, epochs=5, batch_size=5, verbose=2
    # )

    # res = model.evaluate(train_test_data[1], train_test_data[3])
    # print(res)

    # return

    # hyperparemter tuning
    model.set_hyperpar_tuner(max_trials=1, overwrite=False)
    best_hp = model.optimise_hyperpars(train_test_data[0], train_test_data[2], epochs=20)

    # stop_early = model.early_stopping(monitor="loss", patience=5)
    # tuner = model.hyperpar_tuner()
    # tuner.search(
    #     train_test_data[0],
    #     train_test_data[2],
    #     epochs=5,
    #     validation_split=0.2,
    #     callbacks=[stop_early],
    #     verbose=0,
    # )

    print(f"Best units_1: {best_hp.get('units_Dense_1')}")
    print(f"Best units_2: {best_hp.get('units_Dense_3')}")
    # print(f"Best loss: {best_hp.get('loss')}")
    # print(f"Best learning rate: {best_hp.get('learning_rate')}")

    model.build_optimised_model()
    model.summary()

    import matplotlib.pyplot as plt

    # Get the best trials (e.g., top 3)
    best_trials = model.hyperpar_tuner.oracle.get_best_trials(num_trials=3)

    for trial in best_trials:
        # Retrieve the metric history object for a specific metric like 'val_accuracy'
        val_acc_history = trial.metrics.get_history("val_accuracy")

        print(val_acc_history)

        # Extract the numerical values from the history data
        values = [record.value[0] for record in val_acc_history]
        print(values)

        # plt.plot(values, label=f"Trial {trial.trial_id}")

        # plt.title("Model Accuracy Evolution across Epochs")
        # plt.xlabel("Epoch")
        # plt.ylabel("Val Accuracy")
        # plt.legend()
        # plt.show()

    return

    fit_res, ypreds = model.train_test_model(
        train_test_data, return_prediction=True, epochs=20, batch_size=5, verbose=2
    )

    optimal_epochs = model.get_optimal_epochs("mse", "minimise")
    print(optimal_epochs)

    # model2 = tuner.hypermodel.build(best_hp)
    # model2.summary()

    y_test_class = np.argmax(train_test_data[3], axis=1)
    y_pred_class = np.argmax(ypreds, axis=1)

    print(classification_report(y_test_class, y_pred_class))

    return

    fit_res, ypreds = model.train_test_model(
        train_test_data, return_prediction=True, epochs=10, batch_size=5, verbose=2
    )

    print(fit_res.history)

    # ypreds2 = model.predict(train_test_data[1])

    for y_true, y_pred in zip(train_test_data[3], ypreds):
        print(f"{y_true} | {y_pred[0]}")

    # plot_history(
    #     fit_res,
    #     metrics={
    #         "MSE": ["mse", "val_mse"],
    #         "MAE": ["mae", "val_mae"],
    #         "LOSS": ["loss", "val_loss"],
    #     },
    # )

    # export
    # model.export("./model.keras")

    # saved_model = model.load("./model.keras")
    # saved_model.summary()

    y_preds_imported = ModelHandler.apply_exported_model("./model.keras", train_test_data[1])

    print(y_preds_imported)


if __name__ == "__main__":
    DEBUG: bool = True
    main(DEBUG)
