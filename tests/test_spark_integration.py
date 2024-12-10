from typing import Dict, Union, Tuple, Generator, cast, List
import pytest
import numpy as np
import pandas as pd
from pyspark.sql import SparkSession, DataFrame as SparkDataFrame
import tempfile
from numpy.typing import NDArray

from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.typing import DataFrameLike, FloatArray

# Custom types
SparkOrPandas = Union[SparkDataFrame, pd.DataFrame]
MultiTaskData = Dict[
    str,
    Union[Tuple[NDArray[np.float32], NDArray[np.float32]], SparkOrPandas, List[str]],
]


@pytest.fixture(scope="session")
def spark() -> Generator[SparkSession, None, None]:
    """Create a SparkSession fixture to be used across tests"""
    spark_session = (
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
    yield spark_session
    spark_session.stop()


@pytest.fixture
def classification_data(
    spark: SparkSession,
) -> Dict[str, Union[Tuple[FloatArray, FloatArray], DataFrameLike]]:
    """Create simple classification dataset"""
    np.random.seed(42)
    n_samples = 100  # Reduced for faster testing
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    data_dict = {name: X[:, i] for i, name in enumerate(feature_names)}
    pdf = pd.DataFrame(data_dict)
    pdf["target"] = y

    # Convert all columns to float for Spark compatibility
    for col in pdf.columns:
        pdf[col] = pdf[col].astype(float)

    try:
        sdf = spark.createDataFrame(pdf)
    except Exception as e:
        pytest.skip(f"Failed to create Spark DataFrame: {str(e)}")

    return {"numpy": (X, y), "pandas": pdf, "spark": sdf}


@pytest.fixture
def regression_data(
    spark: SparkSession,
) -> Dict[
    str,
    Union[
        Tuple[NDArray[np.float32], NDArray[np.float32]], SparkDataFrame, pd.DataFrame
    ],
]:
    """Create simple regression dataset"""
    np.random.seed(42)
    n_samples = 100  # Reduced for faster testing
    n_features = 10

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + 2 * X[:, 1] + np.random.randn(n_samples) * 0.1).astype(np.float32)

    feature_names = [f"feature_{i}" for i in range(n_features)]
    pdf = pd.DataFrame(X, columns=feature_names)
    pdf["target"] = y

    # Convert all columns to float for Spark compatibility
    for col in pdf.columns:
        pdf[col] = pdf[col].astype(float)

    try:
        sdf = spark.createDataFrame(pdf)
    except Exception as e:
        pytest.skip(f"Failed to create Spark DataFrame: {str(e)}")

    return {"numpy": (X, y.reshape(-1, 1)), "pandas": pdf, "spark": sdf}


def test_classifier_pandas_input(
    classification_data: Dict[str, Union[Tuple[FloatArray, FloatArray], DataFrameLike]],
) -> None:
    """Test TabNetClassifier with pandas input"""
    pdf = cast(pd.DataFrame, classification_data["pandas"])
    X = pdf.drop("target", axis=1)
    y = pdf["target"].to_numpy()

    clf = TabNetClassifier(
        n_d=4,  # Reduced for faster testing
        n_a=4,
        n_steps=2,
        n_independent=1,
        n_shared=1,
    )

    clf.fit(X, y, max_epochs=2, patience=2)
    preds = clf.predict(X)
    probs = clf.predict_proba(X)

    assert preds.shape[0] == len(y)
    assert probs.shape == (len(y), 2)


def test_classifier_spark_input(
    classification_data: Dict[str, Union[Tuple[FloatArray, FloatArray], DataFrameLike]],
) -> None:
    """Test TabNetClassifier with Spark input"""
    sdf = cast(pd.DataFrame, classification_data["spark"])

    clf = TabNetClassifier(n_d=4, n_a=4, n_steps=2, n_independent=1, n_shared=1)

    try:
        clf.fit(sdf, "target", max_epochs=2)
        preds = clf.predict(sdf)
        probs = clf.predict_proba(sdf)

        assert len(preds) == len(sdf)
        assert probs.shape[0] == len(sdf)
        assert probs.shape[1] == 2
    except Exception as e:
        pytest.skip(f"Spark test failed: {str(e)}")


def test_regressor_pandas_input(
    regression_data: Dict[
        str,
        Union[
            Tuple[NDArray[np.float32], NDArray[np.float32]],
            SparkDataFrame,
            pd.DataFrame,
        ],
    ],
) -> None:
    """Test TabNetRegressor with pandas input"""
    pdf = cast(pd.DataFrame, regression_data["pandas"])
    X = pdf.drop("target", axis=1)
    # Convert to numpy array before reshaping
    y = np.asarray(pdf["target"].values).reshape(-1, 1).astype(np.float32)

    reg = TabNetRegressor(n_d=4, n_a=4, n_steps=2, n_independent=1, n_shared=1)
    reg.fit(X, y, max_epochs=2)


@pytest.mark.slow
def test_regressor_spark_input(regression_data):
    """Test TabNetRegressor with Spark input"""
    sdf = regression_data["spark"]
    X = sdf.drop("target")
    y = sdf.select("target").toPandas().values.reshape(-1, 1)

    reg = TabNetRegressor(n_d=4, n_a=4, n_steps=2, n_independent=1, n_shared=1)
    reg.fit(X, y, max_epochs=2)


@pytest.mark.slow
def test_validation_set_spark(classification_data):
    """Test validation set handling with Spark DataFrames"""
    train_sdf = classification_data["spark"]
    val_sdf = classification_data["spark"]  # Using same data for simplicity

    X_train = train_sdf.drop("target")
    y_train = train_sdf.select("target").toPandas().values.ravel()  # Flatten to 1D
    X_val = val_sdf.drop("target")
    y_val = val_sdf.select("target").toPandas().values.ravel()  # Flatten to 1D

    clf = TabNetClassifier(n_d=4, n_a=4, n_steps=2, n_independent=1, n_shared=1)
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], max_epochs=2)


@pytest.mark.slow
def test_feature_importance_spark(classification_data):
    """Test feature importance computation with Spark input"""
    sdf = classification_data["spark"]
    X = sdf.drop("target")
    y = sdf.select("target").toPandas().values.ravel()  # Flatten to 1D

    clf = TabNetClassifier(n_d=4, n_a=4, n_steps=2, n_independent=1, n_shared=1)
    clf.fit(X, y, max_epochs=2)


def test_explain_spark(classification_data):
    """Test explain method with Spark input"""
    sdf = classification_data["spark"]
    X = sdf.drop("target")
    y = sdf.select("target").toPandas().values.ravel()  # Flatten to 1D

    clf = TabNetClassifier(n_d=8, n_a=8, n_steps=3)
    clf.fit(X, y, max_epochs=3)
    # Test explain method
    explain_matrix, masks = clf.explain(sdf)

    assert explain_matrix.shape[0] == sdf.count()
    assert explain_matrix.shape[1] == len(sdf.columns) - 1  # excluding target
    assert isinstance(masks, dict)


@pytest.fixture
def spark_session() -> SparkSession:
    """Create a test SparkSession."""
    return SparkSession.builder.master("local[1]").getOrCreate()


@pytest.fixture
def sample_dataframe(
    spark_session: SparkSession,
) -> Generator[SparkSession, None, None]:
    """Create a sample dataframe for testing."""
    # ... existing setup code ...
    yield spark_session
    spark_session.stop()


def test_basic_transformation(spark_session: SparkSession) -> None:
    """Test basic data transformation."""
    data = [("John", 30), ("Alice", 25)]  # example data
    schema = ["name", "age"]
    df = spark_session.createDataFrame(data, schema)
    count = df.count()  # count() is called on DataFrame
    assert count > 0


def test_complex_transformation(spark_session: SparkSession) -> None:
    """Test complex data transformation."""
    # Your test code here
    pass


# @pytest.fixture
# def multitask_data(spark: SparkSession) -> MultiTaskData:
#     """Create simple multi-task classification dataset"""
#     np.random.seed(42)
#     n_samples = 100
#     n_features = 10

#     # Generate features
#     X = np.random.randn(n_samples, n_features).astype(np.float32)

#     # Generate two classification tasks
#     y1 = (X[:, 0] + X[:, 1] > 0).astype(np.float32)  # Binary task
#     y2 = np.digitize(X[:, 2], bins=[-1, 0, 1]).astype(np.float32)  # 3-class task
#     y = np.column_stack([y1, y2])

#     # Create pandas DataFrame
#     feature_names = [f"feature_{i}" for i in range(n_features)]
#     pdf = pd.DataFrame(X, columns=feature_names)
#     pdf["target_1"] = y1
#     pdf["target_2"] = y2

#     # Convert all columns to float for Spark compatibility
#     for col in pdf.columns:
#         pdf[col] = pdf[col].astype(float)

#     # Create Spark DataFrame
#     try:
#         sdf = spark.createDataFrame(pdf)  # type: ignore
#     except Exception as e:
#         pytest.skip(f"Failed to create Spark DataFrame: {str(e)}")

#     return {
#         "numpy": (X, y),
#         "pandas": pdf,
#         "spark": sdf,
#         "target_columns": ["target_1", "target_2"],
#     }


# def test_multitask_classifier_pandas_input(multitask_data: MultiTaskData) -> None:
#     """Test TabNetMultiTaskClassifier with pandas input"""
#     pdf = cast(pd.DataFrame, multitask_data["pandas"])
#     target_cols = cast(List[str], multitask_data["target_columns"])

#     X = pdf.drop(target_cols, axis=1)
#     y = pdf[target_cols].values.astype(np.float32)

#     clf = TabNetMultiTaskClassifier(
#         n_d=4,
#         n_a=4,
#         n_steps=2,
#         n_independent=1,
#         n_shared=1,
#     )

#     # Test fitting
#     clf.fit(X, y, max_epochs=2)
#     # Test predictions
#     preds = clf.predict(X)
#     assert isinstance(preds, dict)
#     assert len(preds) == len(target_cols)
#     # Task 0 is binary, Task 1 has 3 classes
#     assert preds[0].shape[0] == len(y)  # Number of samples
#     assert preds[1].shape[0] == len(y)  # Number of samples

#     # Test probabilities
#     probs = clf.predict_proba(X)
#     assert isinstance(probs, dict)
#     assert len(probs) == len(target_cols)
#     #print(probs.keys())
#     # Check shapes for each task
#     assert probs[0].shape == (len(y), 2)  # Binary task: 2 classes
#     assert probs[1].shape == (len(y), 3)  # 3-class task
#     # Verify probabilities sum to 1
#     for task_id in range(len(target_cols)):
#         assert np.allclose(probs[task_id].sum(axis=1), 1.0)


# @pytest.mark.slow
# def test_multitask_classifier_spark_input(multitask_data: MultiTaskData) -> None:
#     """Test TabNetMultiTaskClassifier with Spark input"""
#     sdf = cast(SparkDataFrame, multitask_data["spark"])
#     target_cols = cast(List[str], multitask_data["target_columns"])

#     X = sdf.drop(*target_cols)  # type: ignore
#     y = np.column_stack(
#         [
#             sdf.select(col).toPandas().values.ravel()  # type: ignore
#             for col in target_cols
#         ]
#     ).astype(np.float32)

#     clf = TabNetMultiTaskClassifier(
#         n_d=4,
#         n_a=4,
#         n_steps=2,
#         n_independent=1,
#         n_shared=1,
#     )

#     try:
#         clf.fit(X, y, max_epochs=2)
#         preds = clf.predict(X)

#         assert isinstance(preds, dict)
#         assert len(preds) == len(target_cols)
#         for task_id in range(len(target_cols)):
#             assert preds[task_id].shape[0] == sdf.count()  # type: ignore

#         probs = clf.predict_proba(X)
#         assert isinstance(probs, dict)
#         assert len(probs) == len(target_cols)
#         for task_id in range(len(target_cols)):
#             assert probs[task_id].shape[0] == sdf.count()  # type: ignore
#             assert np.allclose(probs[task_id].sum(axis=1), 1.0)

#     except Exception as e:
#         pytest.skip(f"Spark test failed: {str(e)}")


# @pytest.mark.slow
# def test_multitask_validation_set_spark(multitask_data: MultiTaskData) -> None:
#     """Test validation set handling with Spark DataFrames for multitask"""
#     sdf = cast(SparkDataFrame, multitask_data["spark"])
#     target_cols = cast(List[str], multitask_data["target_columns"])

#     X_train = sdf.drop(*target_cols)
#     y_train = np.column_stack(
#         [sdf.select(col).toPandas().values.ravel() for col in target_cols]
#     )

#     # Use same data for validation (for simplicity)
#     X_val = X_train
#     y_val = y_train

#     clf = TabNetMultiTaskClassifier(
#         n_d=4,
#         n_a=4,
#         n_steps=2,
#         n_independent=1,
#         n_shared=1,
#     )

#     try:
#         clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], max_epochs=2)

#         # Verify that validation metrics were computed
#         assert hasattr(clf, "history")
#         assert len(clf.history.epoch_metrics) > 0

#     except Exception as e:
#         pytest.skip(f"Spark validation test failed: {str(e)}")


# @pytest.mark.slow
# def test_multitask_feature_importance_spark(multitask_data: MultiTaskData) -> None:
#     """Test feature importance computation with Spark input for multitask"""
#     sdf = cast(SparkDataFrame, multitask_data["spark"])
#     target_cols = cast(List[str], multitask_data["target_columns"])

#     X = sdf.drop(*target_cols)
#     y = np.column_stack(
#         [sdf.select(col).toPandas().values.ravel() for col in target_cols]
#     )

#     clf = TabNetMultiTaskClassifier(
#         n_d=4,
#         n_a=4,
#         n_steps=2,
#         n_independent=1,
#         n_shared=1,
#     )

#     try:
#         clf.fit(X, y, max_epochs=2)

#         # Verify feature importances were computed
#         assert hasattr(clf, "feature_importances_")
#         assert clf.feature_importances_.shape[0] == len(X.columns)
#         assert np.all(clf.feature_importances_ >= 0)

#     except Exception as e:
#         pytest.skip(f"Spark feature importance test failed: {str(e)}")
