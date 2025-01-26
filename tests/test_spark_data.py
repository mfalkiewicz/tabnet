import pytest
import torch
from pyspark.sql import SparkSession
import pandas as pd
from pytorch_tabnet.data import TabularDataBatch
import pytorch_tabnet.spark_data as spark_data

@pytest.fixture(scope="module")
def spark():
    spark = SparkSession.builder \
                .master("local[2]") \
                .appName("tabnet-spark-test") \
                .getOrCreate()
    yield spark
    spark.stop()
    
@pytest.fixture    
def data(spark):
    pdf = pd.DataFrame({
        'A': [1, 2, 3, 4, 5], 
        'B': [10, 20, 30, 40, 50],
        'y': [0, 1, 0, 1, 0]
    })
    return spark.createDataFrame(pdf)

def test_spark_provider_iter(data):
    provider = spark_data.SparkDataProvider(data, ['A', 'B'], 'y', batch_size=2)
    batches = list(iter(provider))
    
    # With 5 rows and batch_size=2, we should get 3 batches:
    # 2 full batches and 1 partial batch
    assert len(batches) == 3
    
    # First two batches should be full
    assert batches[0].X.shape == (2, 2)  # 2 rows, 2 features
    assert batches[1].X.shape == (2, 2)
    # Last batch should have remaining row
    assert batches[2].X.shape == (1, 2)
    
    # Check target shapes
    assert batches[0].y.shape == (2, 1)  # 2 rows, 1 target
    assert batches[1].y.shape == (2, 1)
    assert batches[2].y.shape == (1, 1)
    
def test_spark_provider_len(data):
    provider = spark_data.SparkDataProvider(data, ['A', 'B'], 'y', batch_size=3)
    assert len(provider) == 2