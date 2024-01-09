from typing import Type, Dict

import pandas as pd
from sklearn.base import BaseEstimator
import numpy as np
from numpy import ndarray
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from typing import List, Tuple


class Tuner:

    def __init__(self, pipeline: Pipeline, param_grid: Dict[str, str])-> None:

        self.pipeline = pipeline
        self.param_grid = param_grid


    def grid_search_tune(self, X: pd.DataFrame, y: pd.Series)-> (Dict[str, float], pd.DataFrame):

        grid_search = GridSearchCV(self.pipeline,
                                   self.param_grid,
                                   cv = 5,
                                   scoring = 'neg_root_mean_squared_error')
        grid_search.fit(X, y)

        # Best hyperparameters
        best_params = grid_search.best_params_
        # overall results
        results = pd.DataFrame(grid_search.cv_results_)

        return best_params, results.sort_values(by="mean_test_score", ascending=False)

    def randomized_search_tune(self, X: pd.DataFrame,
                               y: pd.Series,
                               model_name:str,
                               transformer='transformer',
                               pipeline = 'preprocessing')-> BaseEstimator:

        randomized_search = RandomizedSearchCV(self.pipeline,
                                               self.param_grid,
                                               cv = 3,
                                               n_iter=10,
                                               scoring = 'neg_root_mean_squared_error',
                                               random_state=42)
        randomized_search.fit(X, y)

        final_model = randomized_search.best_estimator_
        #feature_importances = final_model[model_name].feature_importances_
        #feature_names = final_model[pipeline].named_steps[transformer].get_feature_names_out()
        feature_names = final_model[pipeline].get_feature_names_out()

        #return final_model, sorted(zip(feature_importances, feature_names))
        return final_model



