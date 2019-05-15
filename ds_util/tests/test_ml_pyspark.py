import pytest
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.datasets import load_iris, load_boston
from sklearn.linear_model import LogisticRegression, Ridge
from ds_util.ml.pyspark import deploy_python_model


@pytest.fixture
def gen_class_model_and_data():
    """
    Generate classification data, train a model and get predictions.
    """
    spark = SparkSession.builder.master("local").getOrCreate()

    # load data
    X, y = load_iris(True)

    # fit model
    model = LogisticRegression(solver="liblinear", multi_class="auto")
    model.fit(X, y)

    # get predictions
    truth_pred = model.predict_proba(X)

    # build pandas DF to plug into model
    df = pd.DataFrame(
        X,
        columns=[x.split(' (')[0].replace(' ', '_') for x in load_iris().feature_names]
    )
    df['id'] = ['id_' + str(i + 1) for i in range(X.shape[0])]
    df = spark.createDataFrame(df).repartition(10)

    return df, model, X, y, truth_pred


@pytest.fixture
def gen_reg_model_and_data():
    """
    Generate regression data, train a model and get predictions.
    """
    spark = SparkSession.builder.master("local").getOrCreate()

    # load data
    X, y = load_boston(True)

    # fit model
    model = Ridge()
    model.fit(X, y)

    # get predictions
    truth_pred = model.predict(X)

    # build pandas DF to plug into model
    df = pd.DataFrame(
        X,
        columns=[x.split(' (')[0].replace(' ', '_') for x in load_boston().feature_names]
    )
    df['id'] = ['id_' + str(i + 1) for i in range(X.shape[0])]
    df = spark.createDataFrame(df).repartition(10)

    return df, model, X, y, truth_pred


def test_deploy_python_model_class(gen_class_model_and_data):
    """
    Ensure `ds_util.ml.pyspark.deploy_python_model` generates the desired prediction values
    """
    df, model, X, y, truth_pred = gen_class_model_and_data
    pred_names = list(load_iris().target_names)

    pred = deploy_python_model(df, model, id_cols=['id'], pred_names=pred_names, predict_method="predict_proba")
    assert np.abs(pred.groupBy().sum().collect()[0]['sum(setosa)'] - float(truth_pred[:, 0].sum())) <= 0.00001


def test_deploy_python_model_reg(gen_reg_model_and_data):
    """
    Ensure `ds_util.ml.pyspark.deploy_python_model` generates the desired prediction values
    """
    df, model, X, y, truth_pred = gen_reg_model_and_data

    pred = deploy_python_model(df, model, id_cols=['id'], predict_method="predict")
    assert np.abs(pred.groupBy().sum().collect()[0]['sum(pred_1)'] - float(truth_pred.sum())) <= 0.00001


def test_deploy_python_model_missing_id_exception(gen_class_model_and_data):
    df, model, X, y, truth_pred = gen_class_model_and_data

    with pytest.raises(Exception):
        deploy_python_model(df, model, id_cols=['not_an_id'])


def test_deploy_python_model_input_col_exception(gen_class_model_and_data):
    df, model, X, y, truth_pred = gen_class_model_and_data

    with pytest.raises(Exception):
        deploy_python_model(df, model, input_cols=['not_a_column'])


def test_deploy_python_model_wrong_predict_method_exception(gen_class_model_and_data):
    df, model, X, y, truth_pred = gen_class_model_and_data

    with pytest.raises(Exception):
        deploy_python_model(df, model, predict_method='not_a_predict_method')
