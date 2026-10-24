#!/usr/bin/env python3
# Working in a team with: 3e5d0eb9-3c05-40f7-a7b8-405d0329b98c
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt

import sklearn.compose
import sklearn.linear_model
import sklearn.pipeline
import sklearn.metrics
import sklearn.preprocessing
import sklearn.model_selection
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")

class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2425/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)

def prepare_features(data: npt.NDArray) -> npt.NDArray:
    # delete indicator about working day because it can be deducted from day of week and holiday
    # also deleting holiday because it makes the model better
    # -> I don't agree with this one, for linear model it's better to keep explicit features, it's harder for the model to explore these dependencies
    # train.data = np.delete(train.data, [4, 6], 1)
    # making the month column less granular
    # train.data[:, 2] = train.data[:, 2]//2
    # making features out of hour to help the model find better connections
    is_afternoon = ((11 <= data[:, 3]).astype(int) * (data[:, 3] <= 16).astype(int)).reshape(data.shape[0], 1)
    is_evening = ((17 <= data[:, 3]).astype(int) * (data[:, 3] <= 19).astype(int)).reshape(data.shape[0], 1)
    is_morning = ((7 <= data[:, 3]).astype(int) * (data[:, 3] <= 8).astype(int)).reshape(data.shape[0], 1)
    hour_working_day = (data[:, 3] + 24 * data[:, 6]).reshape(-1, 1) # hour x working day interaction
    # plt.plot(train.data[:, 3], train.target, 'o')
    # plt.show()

    # interaction between temperatures and windspeed
    inter_temp_wind = np.hstack(((data[:, -4] * data[:, -1]).reshape(data.shape[0], 1),
               (data[:, -3] * data[:, -1]).reshape(data.shape[0], 1)))
    # interaction between temperatures and weather
    inter_temp_weather = np.hstack(((data[:, -4] * data[:, -5]).reshape(data.shape[0], 1),
                (data[:, -3] * data[:, -5]).reshape(data.shape[0], 1)))
    is_raining = (data[:, -5] > 2).astype(int).reshape(data.shape[0], 1)
    is_heavy_raining = (data[:, -5] == 4).astype(int).reshape(data.shape[0], 1)
    data = np.hstack((data, is_raining, is_heavy_raining, inter_temp_wind, is_evening, is_afternoon, hour_working_day))
    return data


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        train.data = prepare_features(train.data)

        cat_cols = [0, 1, 2, 3, 4, 5, 6, 7, 18] # this might need to be changed if you apply changes to the data above this line
        # I also added to categorial: year, holiday, working day, weather + one new feature (hour x working day)
        transformer = sklearn.compose.ColumnTransformer(transformers =
            [("OHEncoding", sklearn.preprocessing.OneHotEncoder(sparse_output = False, handle_unknown = "ignore"), cat_cols)], remainder = 'passthrough')

        pipeline = sklearn.pipeline.Pipeline([
            ("transformer", transformer),
            ("model", sklearn.linear_model.PoissonRegressor(alpha=0.2, max_iter=1000)),
        ])

        # Try an ensemble of models

        for l in [0.2, 0.3]:
            pipeline.set_params(model__alpha=l)
            kf = sklearn.model_selection.KFold(n_splits = 5, shuffle = True, random_state = args.seed)
            scores = sklearn.model_selection.cross_val_score(pipeline, train.data, train.target, scoring = "neg_root_mean_squared_error", cv = kf)
            print(-scores)
            print(-np.mean(scores))

        # TODO: Train a model on the given dataset and store it in `model`.
        model = pipeline.set_params(model__alpha=0.3).fit(train.data, train.target)

        # Serialize the model.
        # with lzma.open(args.model_path, "wb") as model_file:
        #     pickle.dump(model, model_file)

        # This is a workaround to use Poisson Regression from AI. I think we can use Poisson, because it's a GLM model.
        model_data = model

        with lzma.open(args.model_path, "wb") as f:
            pickle.dump(model_data, f)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        # interaction between temperatures and windspeed
        test.data = prepare_features(test.data)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        # predictions = model.predict(test.data)

        # Specific for Poisson
        predictions = model.predict(test.data)

        return predictions

if __name__ == "__main__":
    main_args = parser.parse_args([] if "__file__" not in globals() else None)
    main(main_args)