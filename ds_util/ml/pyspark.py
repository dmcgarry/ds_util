import warnings
import numpy as np
import pandas as pd
from pyspark.sql import Row


def _convert_numpy_type(d):
    """
    Convert all numpy data types contained in a dictionary to native python types.

    Parameters
    ----------
    d : dict<str, object>
        The dictionary of column names as keys and data types as values.

    Returns
    -------
    dict<str, object>
        The input dictionary with numpy types as python types.

    """
    for col in d:
        if isinstance(d[col], np.generic):
            d[col] = d[col].item()
    return d


def _deploy_python_model_partition(df_part, model, id_cols, input_cols, pred_names, predict_method, predict_params):
    """
    Helper for deploy_python_model that applies the `predict_method` to an individual partition of data.

    Parameters
    ----------
    df_part : pyspark.sql.DataFrame
        A partition of data to be fed into the `predict_method` of `model`.
    model : object
        Python model object that contains a "predict" method (ex: a sklearn model object).
    id_cols : list<str>
        The name(s) of the column(s) in `df` that you would like included in the resulting prediction dataframe. If
        None then no columns are included.
    input_cols : list<str>
        The name(s) of the column(s) that should be used as a subset from `df` when calling the "predict_method". If
        None then all columns are used.
    pred_names : list<str>
        This output name(s) for the resulting prediction columns. If None then "pred_1", ..., "pred_n" is used.
    predict_method : str
        The name of the model object's method to call on a partition of data.
    predict_params : dict
        Any extra parameters to be fed into each call of the `model` object's `predict_method`.

    Returns
    -------
    list<pyspark.sql.Row>
        A list where each element is a row of data that contains the resulting predictions & `id_cols`.

    """
    # init lists
    ids = []
    feat = []

    for row in df_part:
        # grab id_cols if they exist
        if id_cols is not None:
            try:
                ids.append([row[c] for c in id_cols])
            except ValueError:
                raise ValueError("id_cols {} are not present in data, columns include: {}".format(
                    str([c for c in id_cols if c not in row.__fields__]),
                    str(row.__fields__))
                )
        else:
            id_cols = []

        # grab input_cols
        if input_cols is None:
            input_cols = [c for c in row.__fields__ if c not in id_cols]
            feat.append([row[c] for c in input_cols])
        else:
            try:
                feat.append([row[c] for c in input_cols])
            except ValueError:
                raise ValueError("input_cols {} are not present in data, columns include: {}".format(
                    str([c for c in input_cols if c not in row.__fields__]),
                    str(row.__fields__))
                )

    # format input features as pandas DF
    feat = pd.DataFrame(feat, columns=input_cols)

    # ensure partition isn't empty
    if feat.shape[0] == 0:
        warnings.warn("Partition has no prediction data")
        return []

    # get predictions
    pred_func = getattr(model, predict_method)
    pred = pred_func(feat, **predict_params)

    # reshape preds if they a single column
    if len(pred.shape) == 1:
        pred = pred.reshape(-1, 1)

    # format preds into pandas DF
    if pred_names is not None:
        if len(pred_names) != pred.shape[1]:
            raise ValueError("pred_names has length {} but resulting predictions has length {}".format(
                len(pred_names), pred.shape[1]
            ))
    else:
        pred_names = ["pred_" + str(i+1) for i in range(pred.shape[1])]

    pred = pd.DataFrame(pred, columns=pred_names)

    # add ids
    if len(id_cols) > 0:
        ids = pd.DataFrame(ids, columns=id_cols)
        pred = pd.concat([ids, pred], axis=1, join="inner")

    return [Row(**row) for row in [_convert_numpy_type(row) for row in pred.to_dict("records")]]


def deploy_python_model(
        df, model, id_cols=None, input_cols=None, pred_names=None, predict_method="predict", predict_params=None
):
    """
    Distribute the predictions of an in-memory python model on a pyspark dataframe.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Spark DataFrame containing your input data.
    model : object
        Python model object that contains a "predict" method (ex: a sklearn model object).
    id_cols : list<str>
        The name(s) of the column(s) in `df` that you would like included in the resulting prediction dataframe. If
        None then no columns are included.
    input_cols : list<str>
        The name(s) of the column(s) that should be used as a subset from `df` when calling the "predict_method". If
        None then all columns are used.
    pred_names : list<str>
        This output name(s) for the resulting prediction columns. If None then "pred_1", ..., "pred_n" is used.
    predict_method : str
        The name of the model object's method to call on a partition of data.
    predict_params : dict
        Any extra parameters to be fed into each call of the `model` object's `predict_method`.

    Returns
    -------
    pyspark.sql.DataFrame
        A spark dataframe containing the `id_cols` & `pred_names` columns of predictions.
    """
    if predict_params is None:
        predict_params = {}

    preds = df.rdd.mapPartitions(
        lambda part: _deploy_python_model_partition(
            part, model, id_cols, input_cols, pred_names, predict_method, predict_params
        )
    )

    return preds.toDF()
