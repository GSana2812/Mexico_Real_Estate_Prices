from typing import Type, Dict
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
import pandas as pd
import pickle
from sklearn.model_selection import cross_val_score
from numpy import ndarray

class MLWorkflow:

    def build_modelling_pipeline(self, preprocessing_pipeline: Pipeline,
                                 model_name:str,
                                 model: BaseEstimator) -> Pipeline:

            pipeline = Pipeline([('preprocessing',preprocessing_pipeline),
                                 (model_name, model)])

            return pipeline

    @staticmethod
    def return_loss(full_pipeline: Pipeline,
                    components: Dict[str, pd.DataFrame],
                    process: str)->float:

        if process == 'train':
            full_pipeline.fit(components['X_train'], components['y_train'])
            y_pred_train = full_pipeline.predict(components['X_train'])
            return mean_squared_error(y_pred_train, components['y_train'])

        if process == 'test':
            y_pred_test = full_pipeline.predict(components['X_test'])
            return mean_squared_error(y_pred_test, components['y_test'])

        else:
            return 'Error'

    @staticmethod
    def cross_validation(model: Pipeline, X: pd.DataFrame, y: pd.Series, cv=10) -> ndarray:
        return -cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=cv)

    def save_model(self, pipeline: Pipeline, file_path: str) -> None:

        with open(file_path, 'wb') as file:
            pickle.dump(pipeline, file)

    def load_model(self, file_path: str):

        with open(file_path, 'rb') as file:
            loaded_model = pickle.load(file)

        return loaded_model









