"""Integration tests for training workflow with different DataFrame backends."""

import pytest
import numpy as np
import pandas as pd
import polars as pl
from pyspark.sql import SparkSession
import torch
from typing import Dict, Any, Type

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.dataframe import (
    TabNetDataFrame,
    PandasDataFrame,
    PolarsDataFrame,
    SparkDataFrame
)

# Test Data Fixtures

@pytest.fixture
def training_data() -> Dict[str, Any]:
    """Create synthetic training data."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    # Generate binary classification target
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    feature_cols = [f'feature_{i}' for i in range(n_features)]
    
    return {
        'X': X,
        'y': y,
        'feature_cols': feature_cols,
        'n_samples': n_samples,
        'n_features': n_features
    }

@pytest.fixture(scope="module")
def spark():
    """Create a SparkSession for testing."""
    spark = (SparkSession.builder
            .master("local[2]")
            .appName("tabnet-training-test")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            .config("spark.sql.execution.arrow.maxRecordsPerBatch", "10000")
            .getOrCreate())
    yield spark
    spark.stop()

@pytest.fixture
def pandas_df(training_data) -> PandasDataFrame:
    """Create PandasDataFrame for training."""
    df = pd.DataFrame(
        training_data['X'],
        columns=training_data['feature_cols']
    )
    df['target'] = training_data['y']
    return PandasDataFrame(df)

@pytest.fixture
def polars_df(training_data) -> PolarsDataFrame:
    """Create PolarsDataFrame for training."""
    df = pl.DataFrame(
        {
            **{col: training_data['X'][:, i] for i, col in enumerate(training_data['feature_cols'])},
            'target': training_data['y']
        }
    )
    return PolarsDataFrame(df)

@pytest.fixture
def spark_df(spark, training_data) -> SparkDataFrame:
    """Create SparkDataFrame for training."""
    pandas_df = pd.DataFrame(
        training_data['X'],
        columns=training_data['feature_cols']
    )
    pandas_df['target'] = training_data['y']
    df = spark.createDataFrame(pandas_df)
    return SparkDataFrame(df)

class TestTrainingWorkflow:
    """Test training workflow with different DataFrame backends."""
    
    def _get_features_target(self, df, df_fixture):
        """Extract features and target from DataFrame."""
        if isinstance(df, PandasDataFrame):
            X = df._df.drop('target', axis=1).values
            y = df._df['target'].values
        elif isinstance(df, PolarsDataFrame):
            X = df._df.drop('target').to_numpy()
            y = df._df['target'].to_numpy()
        elif isinstance(df, SparkDataFrame):
            X = df._df.drop('target').toPandas().values
            y = df._df.select('target').toPandas()['target'].values
        return X, y
    
    @pytest.mark.parametrize("df_fixture", [
        "pandas_df",
        "polars_df",
        "spark_df"
    ])
    def test_basic_training(self, df_fixture, training_data, request):
        """Test basic model training with different backends."""
        df = request.getfixturevalue(df_fixture)
        X_train, y_train = self._get_features_target(df, df_fixture)
        
        # Initialize model
        model = TabNetClassifier(
            n_d=8,  # Smaller architecture for faster testing
            n_a=8,
            n_steps=3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params=dict(step_size=10, gamma=0.9),
        )
        
        # Train for a few epochs
        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_train, y_train)],
            max_epochs=3,
            batch_size=32,
            virtual_batch_size=16
        )
        
        # Make predictions
        preds = model.predict(X_train)
        assert isinstance(preds, np.ndarray)
        assert preds.shape[0] == training_data['n_samples']
        
        # Get feature importance
        importance = model.feature_importances_
        assert isinstance(importance, np.ndarray)
        assert importance.shape[0] == training_data['n_features']
    
    @pytest.mark.parametrize("df_fixture", [
        "pandas_df",
        "polars_df",
        "spark_df"
    ])
    def test_early_stopping(self, df_fixture, training_data, request):
        """Test early stopping with different backends."""
        df = request.getfixturevalue(df_fixture)
        X_train, y_train = self._get_features_target(df, df_fixture)
        
        model = TabNetClassifier(
            n_d=8,
            n_a=8,
            n_steps=3,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2)
        )
        
        # Train with early stopping
        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_train, y_train)],
            max_epochs=10,
            patience=2,
            batch_size=32
        )
        
        # Verify training history is recorded
        assert hasattr(model, "history")
        assert len(model.history["loss"]) > 0
    
    @pytest.mark.parametrize("df_fixture", [
        "pandas_df",
        "polars_df",
        "spark_df"
    ])
    def test_save_load_model(self, df_fixture, training_data, request, tmp_path):
        """Test model saving and loading with different backends."""
        df = request.getfixturevalue(df_fixture)
        X_train, y_train = self._get_features_target(df, df_fixture)
        
        # Train model
        model = TabNetClassifier(
            n_d=8,
            n_a=8,
            n_steps=3
        )
        model.fit(
            X_train=X_train,
            y_train=y_train,
            eval_set=[(X_train, y_train)],
            max_epochs=2,
            batch_size=32
        )
        
        # Save model
        save_path = tmp_path / "model.zip"  # Changed from .pt to .zip
        model.save_model(save_path)
        
        # Load model
        loaded_model = TabNetClassifier()
        loaded_model.load_model(save_path)
        
        # Verify predictions match
        orig_preds = model.predict(X_train)
        loaded_preds = loaded_model.predict(X_train)
        np.testing.assert_array_almost_equal(orig_preds, loaded_preds)

# Note: Additional integration tests for specific features like:
# - Multi-class classification
# - Regression
# - Multi-task learning
# - Pretraining
# Will be added in subsequent commits