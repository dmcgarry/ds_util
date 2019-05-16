# Example Doc for `ds_util.ml.pyspark` Utilities
### Overview
This function is useful when two condititions are met:
1. You have a model that was trained in memory
2. You want to apply that model (ie: generate predictions) on more data than can fit in memory

For example, perhaps you built a training set using approximately 500k rows of data and trained a model using `sklearn`. Now you would like to apply this model to ~1 billion rows of data, which cannot easily fit into memory all at once. 

### sklearn `Pipeline` objects
This function is most commonly used on sklearn `Pipeline` objects, or other python models trained that were trained in-memory. 
```python
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from ds_util.ml.pyspark import deploy_python_model

# fit in-memory model
X, y = load_boston(True)
model = Pipeline([
    ('model', Ridge())
])
model.fit(X, y)


# build spark dataframe
spark = SparkSession.builder.master("local").getOrCreate()
df_pandas = pd.DataFrame(
    X,
    columns=[x.split(' (')[0].replace(' ', '_') for x in load_boston().feature_names]
)
df_pandas['id'] = ['id_' + str(i + 1) for i in range(X.shape[0])]
df = spark.createDataFrame(df_pandas)
df.show(5)
# +-------+----+-----+----+-----+-----+-----+------+----+-----+-------+------+-----+------+
# |   CRIM|  ZN|INDUS|CHAS|  NOX|   RM|  AGE|   DIS| RAD|  TAX|PTRATIO|     B|LSTAT|    id|
# +-------+----+-----+----+-----+-----+-----+------+----+-----+-------+------+-----+------+
# |0.06127|40.0| 6.41| 1.0|0.447|6.826| 27.6|4.8628| 4.0|254.0|   17.6|393.45| 4.16|id_278|
# |0.08447| 0.0| 4.05| 0.0| 0.51|5.859| 68.7|2.7019| 5.0|296.0|   16.6|393.23| 9.64|id_175|
# |13.9134| 0.0| 18.1| 0.0|0.713|6.208| 95.0|2.2222|24.0|666.0|   20.2|100.63|15.17|id_435|
# |1.83377| 0.0|19.58| 1.0|0.605|7.802| 98.2|2.0407| 5.0|403.0|   14.7|389.61| 1.92|id_163|
# |13.5222| 0.0| 18.1| 0.0|0.631|3.863|100.0|1.5106|24.0|666.0|   20.2|131.42|13.33|id_368|
# +-------+----+-----+----+-----+-----+-----+------+----+-----+-------+------+-----+------+

# distribute the predictions of the model on a dataframe (same as training set for simplicity)
pred = deploy_python_model(df, model, id_cols=['id'], predict_method="predict")
pred.show(10)
# +------+------------------+
# |    id|            pred_1|
# +------+------------------+
# |id_278| 34.69998730443524|
# |id_175|26.305154757423697|
# |id_435|  16.2101035091303|
# |id_163| 40.11560604478411|
# |id_368|10.326005285061168|
# |id_416| 9.403439563201246|
# |id_252| 25.30322500112694|
# |id_228| 32.14860605950641|
# |id_425|14.008365163106767|
# |id_336|21.290764257583554|
# +------+------------------+

# demonstrate that these values are the same as in the in-memory prediction
df_pandas['in_mem_pred'] = model.predict(X)
spark.createDataFrame(df_pandas).select(
    "id",
    "in_mem_pred"
).join(
    pred,
    "id"
).show(5)
# +------+------------------+------------------+                                  
# |    id|       in_mem_pred|            pred_1|
# +------+------------------+------------------+
# |id_268|40.948907974263946|40.948907974263946|
# |id_337|20.660530128720517| 20.66053012872052|
# | id_83|25.988633937447602|25.988633937447602|
# |id_290|26.744510411780894|26.744510411780894|
# |id_296|28.046257008020305|28.046257008020305|
# +------+------------------+------------------+
```

### Methods other than `predict`
The `predict_method` param controls the method from the `model` that is applied to each partition of data and does not need to be called "predict".
```python
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from ds_util.ml.pyspark import deploy_python_model

# fit in-memory model
X, y = load_iris(True)
model = LogisticRegression(solver="liblinear", multi_class="auto")
model.fit(X, y)


# build spark dataframe
spark = SparkSession.builder.master("local").getOrCreate()
df_pandas = pd.DataFrame(
    X,
    columns=[x.split(' (')[0].replace(' ', '_') for x in load_iris().feature_names]
)
df_pandas['id'] = ['id_' + str(i + 1) for i in range(X.shape[0])]
df = spark.createDataFrame(df_pandas)
df.show(5)
# +------------+-----------+------------+-----------+----+                        
# |sepal_length|sepal_width|petal_length|petal_width|  id|
# +------------+-----------+------------+-----------+----+
# |         5.1|        3.5|         1.4|        0.2|id_1|
# |         4.9|        3.0|         1.4|        0.2|id_2|
# |         4.7|        3.2|         1.3|        0.2|id_3|
# |         4.6|        3.1|         1.5|        0.2|id_4|
# |         5.0|        3.6|         1.4|        0.2|id_5|
# +------------+-----------+------------+-----------+----+

# distribute the predictions of the model on a dataframe (same as training set for simplicity)
pred = deploy_python_model(df, model, id_cols=['id'], predict_method="predict")
pred.show(10)
# +-----+------+
# |   id|pred_1|
# +-----+------+
# | id_1|     0|
# | id_2|     0|
# | id_3|     0|
# | id_4|     0|
# | id_5|     0|
# | id_6|     0|
# | id_7|     0|
# | id_8|     0|
# | id_9|     0|
# |id_10|     0|
# +-----+------+

# distribute predictions using a different method than "predict"
deploy_python_model(df, model, id_cols=['id'], predict_method="predict_proba").show(10)
# +-----+------------------+-------------------+--------------------+             
# |   id|            pred_1|             pred_2|              pred_3|
# +-----+------------------+-------------------+--------------------+
# | id_1|0.8780303050242675|0.12195890005077532|1.079492495714774...|
# | id_2|0.7970582918790622|0.20291141319675696|3.029492418097142...|
# | id_3|0.8519976652685965|0.14797647964563018|2.585508577322806...|
# | id_4|0.8234060190878008|0.17653615914181406|5.782177038521727...|
# | id_5|0.8960349729151966|0.10395383635093679|1.119073386671748...|
# | id_6|0.9262342542843656|0.07375278445623415|1.296125940011852...|
# | id_7|0.8940968483208616|0.10586393505967033|3.921661946818649...|
# | id_8|0.8600344102752581|0.13994671461232863|1.887511241322740...|
# | id_9|0.8010286426370733|0.19888675480346427| 8.46025594623212E-5|
# |id_10|0.7926623921874856|0.20731200265621905|2.560515629541092E-5|
# +-----+------------------+-------------------+--------------------+
```

### Custom wrapper classes
Some cases use call for using model objects that may not have the a method that gives the user the exact output they desire and/or building a custom model object that may combine many steps/objects together (such as stacking models together). An easy way to address this issue is wrapping the model object within a new class that has the desired "predict" method. A simple example is only wanting the first column of probability predictions of a binary classification model.
```python
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from ds_util.ml.pyspark import deploy_python_model

# wrapper class for binary model
class BinaryClassModel():
    def __init__(self, m):
        self.model = m
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1].reshape(-1, 1)

# fit in-memory model
X, y = load_breast_cancer(True)
model = LogisticRegression(solver="liblinear", multi_class="auto")
model.fit(X, y)
wrapped_model = BinaryClassModel(model)


# build spark dataframe
spark = SparkSession.builder.master("local").getOrCreate()
df_pandas = pd.DataFrame(
    X,
    columns=[x.split(' (')[0].replace(' ', '_') for x in load_breast_cancer().feature_names]
)
df_pandas['id'] = ['id_' + str(i + 1) for i in range(X.shape[0])]
df = spark.createDataFrame(df_pandas)

# distribute the predictions of the model on a dataframe (same as training set for simplicity)
deploy_python_model(df, model, id_cols=['id'], predict_method="predict_proba").show(5)
# +----+-------------------+--------------------+
# |  id|             pred_1|              pred_2|
# +----+-------------------+--------------------+
# |id_1| 0.9999999999999993|6.14585344540706E-16|
# |id_2| 0.9999999838508578|1.614914227295068...|
# |id_3| 0.9999999645651536|  3.5434846388136E-8|
# |id_4|0.34702600990605603|   0.652973990093944|
# |id_5|  0.999991415954924|8.584045076011482E-6|
# +----+-------------------+--------------------+

# distribute predictions using a different method than "predict"
deploy_python_model(df, wrapped_model, id_cols=['id'], pred_names=["pred"]).show(5)
# +----+--------------------+
# |  id|                pred|
# +----+--------------------+
# |id_1|6.14585344540706E-16|
# |id_2|1.614914227295068...|
# |id_3|  3.5434846388136E-8|
# |id_4|   0.652973990093944|
# |id_5|8.584045076011482E-6|
# +----+--------------------+
```

### Tips, tricks & gotchas
* *Memory Overhead*: A common gotcha is not being aware that the heavy lifting of this process (ie: applying the `predict_method` to each partition of data) is not a native pyspark function. This means that the majority of memory required for processes using `deploy_python_model` typically come from spark's executor `memoryOverhead`. By default `memoryOverhead` is typically set to be a small percantage of an executor's total memory so that should be adjusted in order to not waste resources. 
* *Number of Partitions*: If the process is not working as expected, for example taking a long time, not parallelizing as much as desired, or running into memory errors, a common remedy is adjusting the number of partitions in the input dataframe. The underlying code in `deploy_python_model` calls `df.rdd.mapPartitions` in order to ensure each partition is passed into the model's `predict_method` in one big batch and if each of those partitions are too big or small then the process will not run optimally.
* *Why not just use `map`?*: A mistake that some make is simply passing each row of data into the model individually via Spark's `map` method. While this will give you the same ultimate result, it will be horribly inefficient because the computational optimizations that occur when generating preditions for batches of data rather than individual rows (more details [here](https://scikit-learn.org/stable/modules/computing.html)).