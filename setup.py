from setuptools import setup, find_packages

setup(
    name='ds_util',
    version="0.0.1",
    packages=find_packages(),
    url='https://github.com/dmcgarry/ds_util',
    author='David McGarry',
    author_email='dave.mcgarry@gmail.com',
    description='A collection of utility functions and classes for data science work.',
    install_requires=[
        'pandas>=0.22.0',
        'scikit-learn>=0.19.0',
        'pyspark>=2.3.0',
        'numpy>=1.14.0',
    ],
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
