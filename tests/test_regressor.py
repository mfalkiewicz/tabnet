import pytest
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from pytorch_tabnet.utils import check_input
from pyspark.sql import DataFrame as SparkDataFrame
from typing import Dict, Union, Tuple, List
from numpy.typing import NDArray


@pytest.fixture
def regression_data():
    """Create regression dataset"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_targets = 2  # Multiple regression targets

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.zeros((n_samples, n_targets))
    # First target is linear combination
    y[:, 0] = 0.8 * X[:, 0] + 0.2 * X[:, 1]
    # Second target is non-linear combination
    y[:, 1] = (X[:, 0] * X[:, 1] + np.sin(X[:, 2])) / 2

    feature_names = [f"feature_{i}" for i in range(n_features)]
    X_train = X[:800]
    X_test = X[800:]
    y_train = y[:800]
    y_test = y[800:]

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names
    }


def test_regressor_initialization():
    """Test regressor initialization with various parameters"""
    # Test default initialization
    reg = TabNetRegressor()
    assert reg._task == "regression"
    assert reg._default_loss == torch.nn.functional.mse_loss
    assert reg._default_metric == "mse"

    # Test custom initialization
    reg = TabNetRegressor(
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-3,
        momentum=0.3,
        clip_value=2.
    )
    assert reg.n_d == 8
    assert reg.n_a == 8
    assert reg.n_steps == 3
    assert reg.gamma == 1.5
    assert reg.n_independent == 2
    assert reg.n_shared == 2
    assert reg.lambda_sparse == 1e-3
    assert reg.momentum == 0.3
    assert reg.clip_value == 2.


def test_single_target_regression(regression_data):
    """Test regression with single target"""
    X_train = regression_data["X_train"]
    X_test = regression_data["X_test"]
    y_train = regression_data["y_train"][:, 0:1]  # Single target
    y_test = regression_data["y_test"][:, 0:1]

    reg = TabNetRegressor(
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax",
        n_shared_decoder=1,
        n_indep_decoder=1,
    )

    # Test fit
    reg.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=3,
        patience=2,
        batch_size=128,
        virtual_batch_size=64,
    )

    # Test predict
    y_pred = reg.predict(X_test)
    assert y_pred.shape == y_test.shape

    # Test feature importance
    feat_imp = reg.feature_importances_
    assert len(feat_imp) == X_train.shape[1]
    assert np.all(feat_imp >= 0)

    # Test explain
    explain_matrix = reg.explain(X_test[:10])
    assert explain_matrix.shape == (10, X_test.shape[1])


def test_multi_target_regression(regression_data):
    """Test regression with multiple targets"""
    X_train = regression_data["X_train"]
    X_test = regression_data["X_test"]
    y_train = regression_data["y_train"]
    y_test = regression_data["y_test"]

    reg = TabNetRegressor(
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
    )

    # Test fit
    reg.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=3,
        patience=2,
        batch_size=128,
        virtual_batch_size=64,
    )

    # Test predict
    y_pred = reg.predict(X_test)
    assert y_pred.shape == y_test.shape


def test_regressor_input_validation():
    """Test input validation"""
    reg = TabNetRegressor()

    # Test invalid X shape
    with pytest.raises(ValueError):
        X = np.random.randn(100)  # 1D array
        y = np.random.randn(100, 1)
        reg.fit(X, y)

    # Test mismatched X, y shapes
    with pytest.raises(ValueError):
        X = np.random.randn(100, 10)
        y = np.random.randn(50, 1)  # Wrong length
        reg.fit(X, y)

    # Test non-finite values
    with pytest.raises(ValueError):
        X = np.random.randn(100, 10)
        X[0, 0] = np.inf
        y = np.random.randn(100, 1)
        reg.fit(X, y)

    # Test 1D y array
    with pytest.raises(ValueError):
        X = np.random.randn(100, 10)
        y = np.random.randn(100)  # Should be 2D
        reg.fit(X, y)


def test_regressor_save_load(tmp_path, regression_data):
    """Test model saving and loading"""
    X_train = regression_data["X_train"]
    X_test = regression_data["X_test"]
    y_train = regression_data["y_train"]
    y_test = regression_data["y_test"]

    reg = TabNetRegressor()
    reg.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=3
    )

    # Save model
    save_path = tmp_path / "model.zip"
    reg.save_model(save_path)

    # Load model
    loaded_reg = TabNetRegressor()
    loaded_reg.load_model(save_path)

    # Compare predictions
    y_pred = reg.predict(X_test)
    y_pred_loaded = loaded_reg.predict(X_test)
    assert np.allclose(y_pred, y_pred_loaded)


@pytest.mark.parametrize("batch_size", [16, 32, 64])
@pytest.mark.parametrize("virtual_batch_size", [8, 16, 32])
def test_regressor_batch_sizes(batch_size, virtual_batch_size, regression_data):
    """Test different batch size configurations"""
    X_train = regression_data["X_train"]
    X_test = regression_data["X_test"]
    y_train = regression_data["y_train"]
    y_test = regression_data["y_test"]

    reg = TabNetRegressor()
    reg.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=3,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size
    )

    y_pred = reg.predict(X_test)
    assert y_pred.shape == y_test.shape


def test_regressor_early_stopping(regression_data):
    """Test early stopping functionality"""
    X_train = regression_data["X_train"]
    X_test = regression_data["X_test"]
    y_train = regression_data["y_train"]
    y_test = regression_data["y_test"]

    reg = TabNetRegressor()
    reg.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=100,
        patience=2
    )

    # Check if training stopped early
    assert reg.best_epoch < 100


def test_regressor_feature_selection(regression_data):
    """Test feature selection and importance"""
    X_train = regression_data["X_train"]
    X_test = regression_data["X_test"]
    y_train = regression_data["y_train"]
    y_test = regression_data["y_test"]
    feature_names = regression_data["feature_names"]

    reg = TabNetRegressor()
    reg.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=3
    )

    # Test feature importance shape and values
    feat_imp = reg.feature_importances_
    assert len(feat_imp) == len(feature_names)
    assert np.all(feat_imp >= 0)
    assert np.sum(feat_imp) > 0

    # Test explain matrix
    explain_matrix = reg.explain(X_test)
    assert explain_matrix.shape == (len(X_test), len(feature_names))
    assert np.all(explain_matrix >= 0)
    assert np.all(explain_matrix <= 1)


def test_custom_loss_function(regression_data):
    """Test regressor with custom loss function"""
    X_train = regression_data["X_train"]
    X_test = regression_data["X_test"]
    y_train = regression_data["y_train"][:, 0:1]  # Single target
    y_test = regression_data["y_test"][:, 0:1]

    def custom_loss(y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true))

    reg = TabNetRegressor(
        n_d=8,
        n_a=8,
        n_steps=3
    )

    reg.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=3,
        loss_fn=custom_loss
    )

    y_pred = reg.predict(X_test)
    assert y_pred.shape == y_test.shape


def test_regressor_categorical_features(regression_data):
    """Test regressor with categorical features"""
    X_train = regression_data["X_train"]
    X_test = regression_data["X_test"]
    y_train = regression_data["y_train"]
    y_test = regression_data["y_test"]

    # Add categorical features
    cat_feat1 = np.random.randint(0, 5, size=len(X_train))
    cat_feat2 = np.random.randint(0, 3, size=len(X_train))
    X_train = np.column_stack([X_train, cat_feat1, cat_feat2])

    cat_feat1_test = np.random.randint(0, 5, size=len(X_test))
    cat_feat2_test = np.random.randint(0, 3, size=len(X_test))
    X_test = np.column_stack([X_test, cat_feat1_test, cat_feat2_test])

    reg = TabNetRegressor(
        cat_idxs=[10, 11],  # Indices of categorical features
        cat_dims=[5, 3],    # Number of categories for each categorical feature
        cat_emb_dim=2       # Embedding dimension for categorical features
    )

    reg.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=3
    )

    y_pred = reg.predict(X_test)
    assert y_pred.shape == y_test.shape
