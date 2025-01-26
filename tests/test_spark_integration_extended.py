import pytest
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.spark_utils import SparkDataset
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from typing import Dict, Union, Tuple, List, Generator
import tempfile
from numpy.typing import NDArray


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    """Create a SparkSession fixture"""
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("tabnet_tests")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.default.parallelism", "2")
        .config("spark.sql.execution.arrow.enabled", "true")
        .config(
            "spark.driver.extraJavaOptions",
            f"-Djava.io.tmpdir={tempfile.gettempdir()}",
        )
        .getOrCreate()
    )
    yield spark
    spark.stop()


@pytest.fixture
def spark_classification_data(spark: SparkSession):
    """Create classification dataset in Spark"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Create numpy data
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)

    # Convert to pandas
    feature_names = [f"feature_{i}" for i in range(n_features)]
    pdf = pd.DataFrame(X, columns=feature_names)
    pdf["target"] = y

    # Convert to Spark DataFrame
    sdf = spark.createDataFrame(pdf)

    # Split into train and test
    train_sdf, test_sdf = sdf.randomSplit([0.8, 0.2], seed=42)

    return {
        "train": train_sdf,
        "test": test_sdf,
        "feature_cols": feature_names,
        "target_col": "target"
    }


@pytest.fixture
def spark_regression_data(spark: SparkSession):
    """Create regression dataset in Spark"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Create numpy data
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = 0.8 * X[:, 0] + 0.2 * X[:, 1]

    # Convert to pandas
    feature_names = [f"feature_{i}" for i in range(n_features)]
    pdf = pd.DataFrame(X, columns=feature_names)
    pdf["target"] = y

    # Convert to Spark DataFrame
    sdf = spark.createDataFrame(pdf)

    # Split into train and test
    train_sdf, test_sdf = sdf.randomSplit([0.8, 0.2], seed=42)

    return {
        "train": train_sdf,
        "test": test_sdf,
        "feature_cols": feature_names,
        "target_col": "target"
    }


def test_spark_dataset_creation(spark_classification_data):
    """Test SparkDataset creation and basic functionality"""
    train_sdf = spark_classification_data["train"]
    feature_cols = spark_classification_data["feature_cols"]
    target_col = spark_classification_data["target_col"]

    # Create dataset
    dataset = SparkDataset(
        train_sdf,
        feature_cols=feature_cols,
        target_col=target_col
    )

    # Test basic attributes
    assert dataset.feature_cols == feature_cols
    assert dataset.target_col == target_col
    assert isinstance(dataset.spark_df, DataFrame)

    # Test data loader creation
    loader = dataset.make_loader(
        batch_size=32,
        shuffle=True,
        num_epochs=1
    )
    assert loader is not None


def test_spark_classifier_training(spark_classification_data):
    """Test TabNetClassifier with Spark DataFrame input"""
    train_sdf = spark_classification_data["train"]
    test_sdf = spark_classification_data["test"]
    feature_cols = spark_classification_data["feature_cols"]
    target_col = spark_classification_data["target_col"]

    clf = TabNetClassifier(
        n_d=8,
        n_a=8,
        n_steps=3,
    )

    # Test fit
    clf.fit(
        X_train=train_sdf,
        y_train=target_col,
        eval_set=[(test_sdf, target_col)],
        max_epochs=3,
        patience=2,
        batch_size=128,
        virtual_batch_size=64,
    )

    # Test predict
    predictions = clf.predict(test_sdf)
    assert len(predictions) == test_sdf.count()
    assert all(pred in [0, 1] for pred in predictions)

    # Test predict_proba
    probas = clf.predict_proba(test_sdf)
    assert probas.shape[0] == test_sdf.count()
    assert probas.shape[1] == 2
    assert np.allclose(probas.sum(axis=1), 1.0)


def test_spark_regressor_training(spark_regression_data):
    """Test TabNetRegressor with Spark DataFrame input"""
    train_sdf = spark_regression_data["train"]
    test_sdf = spark_regression_data["test"]
    feature_cols = spark_regression_data["feature_cols"]
    target_col = spark_regression_data["target_col"]

    reg = TabNetRegressor(
        n_d=8,
        n_a=8,
        n_steps=3,
    )

    # Test fit
    reg.fit(
        X_train=train_sdf,
        y_train=target_col,
        eval_set=[(test_sdf, target_col)],
        max_epochs=3,
        patience=2,
        batch_size=128,
        virtual_batch_size=64,
    )

    # Test predict
    predictions = reg.predict(test_sdf)
    assert len(predictions) == test_sdf.count()
    assert predictions.ndim == 1


def test_spark_categorical_features(spark: SparkSession):
    """Test handling of categorical features in Spark"""
    # Create data with categorical features
    n_samples = 1000
    pdf = pd.DataFrame({
        "num_feat1": np.random.randn(n_samples),
        "num_feat2": np.random.randn(n_samples),
        "cat_feat1": np.random.choice(["A", "B", "C"], n_samples),
        "cat_feat2": np.random.choice(["X", "Y"], n_samples),
        "target": np.random.randint(0, 2, n_samples)
    })

    sdf = spark.createDataFrame(pdf)
    train_sdf, test_sdf = sdf.randomSplit([0.8, 0.2], seed=42)

    # Create classifier with categorical features
    clf = TabNetClassifier(
        cat_idxs=[2, 3],  # Indices of categorical features
        cat_dims=[3, 2],  # Number of categories for each feature
        cat_emb_dim=2     # Embedding dimension
    )

    # Test fit
    clf.fit(
        X_train=train_sdf,
        y_train="target",
        eval_set=[(test_sdf, "target")],
        max_epochs=3
    )

    # Test predict
    predictions = clf.predict(test_sdf)
    assert len(predictions) == test_sdf.count()


def test_spark_missing_values(spark: SparkSession):
    """Test handling of missing values in Spark DataFrame"""
    # Create data with missing values
    n_samples = 1000
    pdf = pd.DataFrame({
        "feat1": np.random.randn(n_samples),
        "feat2": np.random.randn(n_samples),
        "target": np.random.randint(0, 2, n_samples)
    })

    # Introduce missing values
    pdf.loc[np.random.choice(n_samples, 100), "feat1"] = None
    pdf.loc[np.random.choice(n_samples, 100), "feat2"] = None

    sdf = spark.createDataFrame(pdf)
    train_sdf, test_sdf = sdf.randomSplit([0.8, 0.2], seed=42)

    # Test with missing values
    with pytest.raises(ValueError):
        clf = TabNetClassifier()
        clf.fit(
            X_train=train_sdf,
            y_train="target",
            eval_set=[(test_sdf, "target")],
            max_epochs=3
        )


def test_spark_data_types(spark: SparkSession):
    """Test handling of different data types in Spark DataFrame"""
    # Create data with different types
    n_samples = 1000
    pdf = pd.DataFrame({
        "int_feat": np.random.randint(0, 100, n_samples),
        "float_feat": np.random.randn(n_samples),
        "bool_feat": np.random.choice([True, False], n_samples),
        "target": np.random.randint(0, 2, n_samples)
    })

    sdf = spark.createDataFrame(pdf)
    train_sdf, test_sdf = sdf.randomSplit([0.8, 0.2], seed=42)

    clf = TabNetClassifier()
    clf.fit(
        X_train=train_sdf,
        y_train="target",
        eval_set=[(test_sdf, "target")],
        max_epochs=3
    )

    predictions = clf.predict(test_sdf)
    assert len(predictions) == test_sdf.count()


def test_spark_dataset_partitioning(spark_classification_data):
    """Test SparkDataset with different partitioning strategies"""
    train_sdf = spark_classification_data["train"]
    feature_cols = spark_classification_data["feature_cols"]
    target_col = spark_classification_data["target_col"]

    # Test with different numbers of partitions
    for num_parts in [2, 4, 8]:
        sdf_repartitioned = train_sdf.repartition(num_parts)
        dataset = SparkDataset(
            sdf_repartitioned,
            feature_cols=feature_cols,
            target_col=target_col
        )
        loader = dataset.make_loader(
            batch_size=32,
            shuffle=True,
            num_epochs=1
        )
        assert loader is not None


def test_spark_dataset_memory_usage(spark_classification_data):
    """Test memory usage patterns of SparkDataset"""
    train_sdf = spark_classification_data["train"]
    feature_cols = spark_classification_data["feature_cols"]
    target_col = spark_classification_data["target_col"]

    # Create dataset with different cache settings
    dataset = SparkDataset(
        train_sdf,
        feature_cols=feature_cols,
        target_col=target_col
    )

    # Test loader with different batch sizes
    for batch_size in [32, 64, 128, 256]:
        loader = dataset.make_loader(
            batch_size=batch_size,
            shuffle=True,
            num_epochs=1
        )
        # Load one batch to check memory
        batch = next(iter(loader))
        assert isinstance(batch, dict)
        assert all(key in batch for key in feature_cols + [target_col])


def test_spark_dataset_shuffle(spark_classification_data):
    """Test shuffling behavior of SparkDataset"""
    train_sdf = spark_classification_data["train"]
    feature_cols = spark_classification_data["feature_cols"]
    target_col = spark_classification_data["target_col"]

    dataset = SparkDataset(
        train_sdf,
        feature_cols=feature_cols,
        target_col=target_col
    )

    # Create two loaders with same shuffle=True
    loader1 = dataset.make_loader(batch_size=32, shuffle=True, num_epochs=1)
    loader2 = dataset.make_loader(batch_size=32, shuffle=True, num_epochs=1)

    # Get first batch from each loader
    batch1 = next(iter(loader1))
    batch2 = next(iter(loader2))

    # Verify that batches are different (shuffled)
    assert not np.array_equal(
        batch1[feature_cols[0]],
        batch2[feature_cols[0]]
    )


def test_spark_dataset_epochs(spark_classification_data):
    """Test epoch behavior of SparkDataset"""
    train_sdf = spark_classification_data["train"]
    feature_cols = spark_classification_data["feature_cols"]
    target_col = spark_classification_data["target_col"]

    dataset = SparkDataset(
        train_sdf,
        feature_cols=feature_cols,
        target_col=target_col
    )

    # Test with finite epochs
    loader = dataset.make_loader(batch_size=32, shuffle=True, num_epochs=2)
    total_samples = 0
    for batch in loader:
        total_samples += len(batch[feature_cols[0]])
    
    expected_samples = train_sdf.count() * 2
    assert total_samples == expected_samples

    # Test with infinite epochs (num_epochs=None)
    loader = dataset.make_loader(batch_size=32, shuffle=True, num_epochs=None)
    sample_count = 0
    for batch in loader:
        sample_count += len(batch[feature_cols[0]])
        if sample_count >= train_sdf.count():
            break
    assert sample_count >= train_sdf.count()
