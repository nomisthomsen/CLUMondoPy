from setuptools import setup, find_packages

setup(
    name="CLUMondoPy",
    version="0.1.0",
    description="Land suitability and land use change modeling",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "rasterio",
        "geopandas",
        "xgboost",
        "joblib"
    ],
)