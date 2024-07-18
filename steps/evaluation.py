import logging
import pandas as pd 

from zenml import step
from src.evaluation import MSE ,R2Score,RMSE
from sklearn.base import RegressorMixin
from typing import Annotated
from typing_extensions import Tuple
@step 
def evaluate_model(
    model : RegressorMixin,
    X_test : pd.DataFrame,
    y_test : pd.Series,
)-> Tuple[
    Annotated[float,"r2"],
    Annotated[float,"rmse"],
]:
    """
    Evaluates the model on the ingested data
    Args:
        df: the ingested data
    """
    try:
        prediction = model.predict(X_test)
        mse_class = MSE()
        mse = mse_class.calculate_score(y_test,prediction)

        r2_class = R2Score()
        r2 = r2_class.calculate_score(y_test,prediction)

        rmse_class = RMSE()
        rmse = rmse_class.calculate_score(y_test,prediction)

        return r2 , rmse
    except Exception as e :
        logging.error("Error in evaluating model : {}".format(e))
        raise e