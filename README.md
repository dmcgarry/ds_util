# Data Science Util (ds_util)
## Overview
At some point this may be a collection of utility functions and classes for data science work, but right now it's just a skeleton.

## To Install
This package hasn't been uploaded to any PyPI repositories yet, so pull down the repo and install locally via:
`pip install ./`

## To Contribute
TODO - streamline and document how to: run tests, bump the version, etc..

## Utility Glossary
#### ml.pyspark: [source](./ds_util/ml/pyspark.py) & [example doc](examples/ml_pyspark.md)
Utilties for doing machine learning that leverage spark.
* `deploy_python_model`: Distribute the process of generating predictions using a model object that was training in memory.
