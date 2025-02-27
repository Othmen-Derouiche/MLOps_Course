import logging
import pandas as pd
from sklearn.base import RegressorMixin

from zenml import step 
from zenml.client import Client

from src.model_dev import LinearRegressionModel
from steps.config import ModelNameConfig

import mlflow 

experiment_tracker = Client().active_stack.experiment_tracker

@step( experiment_tracker= experiment_tracker.name)
def train_model(
    X_train : pd.DataFrame,
    X_test : pd.DataFrame,
    y_train : pd.Series,
    y_test : pd.Series,
    config : ModelNameConfig,
) -> RegressorMixin:
    """
    Trains the model on the ingested data

    Args:
        X_train : pd.DataFrame,
        X_test : pd.DataFrame,
        y_train : pd.Series,
        y_test : pd.Series,
    """
    try:
        model = None
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog() # automatically log your models , scores 
            model = LinearRegressionModel()
            trained_model = model.train(X_train,y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supported".format(config.model_name))
    except Exception as e :
        logging.error("Error in training model :{}".format(e))
        raise e 