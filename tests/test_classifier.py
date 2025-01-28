import pytest
import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.utils import check_input
from pyspark.sql import DataFrame as SparkDataFrame
from typing import Dict, Union, Tuple, List
from numpy.typing import NDArray


@pytest.fixture
def binary_classification_data():
    """Create binary classification dataset"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)

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


@pytest.fixture
def multiclass_classification_data():
    """Create multiclass classification dataset"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    n_classes = 3

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, n_classes, size=n_samples)

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


def test_classifier_initialization():
    """Test classifier initialization with various parameters"""
    # Test default initialization
    clf = TabNetClassifier()
    assert clf._task == "classification"
    assert clf._default_loss == torch.nn.functional.cross_entropy
    assert clf._default_metric == "accuracy"

    # Test custom initialization
    clf = TabNetClassifier(
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
    assert clf.n_d == 8
    assert clf.n_a == 8
    assert clf.n_steps == 3
    assert clf.gamma == 1.5
    assert clf.n_independent == 2
    assert clf.n_shared == 2
    assert clf.lambda_sparse == 1e-3
    assert clf.momentum == 0.3
    assert clf.clip_value == 2.


def test_binary_classification_numpy(binary_classification_data):
    """Test binary classification with numpy arrays"""
    X_train = binary_classification_data["X_train"]
    X_test = binary_classification_data["X_test"]
    y_train = binary_classification_data["y_train"]
    y_test = binary_classification_data["y_test"]

    clf = TabNetClassifier(
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
    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=10,
        patience=5,
        batch_size=32,
        virtual_batch_size=16,
    )

    # Test predict
    y_pred = clf.predict(X_test)
    assert y_pred.shape == y_test.shape
    assert np.all(np.unique(y_pred) == np.array([0, 1]))

    # Test predict_proba
    y_pred_proba = clf.predict_proba(X_test)
    assert y_pred_proba.shape == (len(y_test), 2)
    assert np.allclose(y_pred_proba.sum(axis=1), 1.0)
    assert np.all((y_pred_proba >= 0) & (y_pred_proba <= 1))

    # Test feature importance
    feat_imp = clf.feature_importances_
    assert len(feat_imp) == X_train.shape[1]
    assert np.all(feat_imp >= 0)

    # Test explain
    explain_matrix, masks = clf.explain(X_test[:10])
    assert explain_matrix.shape == (10, X_test.shape[1])
    assert isinstance(masks, dict)
    assert all(isinstance(v, np.ndarray) for v in masks.values())


def test_multiclass_classification_numpy(multiclass_classification_data):
    """Test multiclass classification with numpy arrays"""
    X_train = multiclass_classification_data["X_train"]
    X_test = multiclass_classification_data["X_test"]
    y_train = multiclass_classification_data["y_train"]
    y_test = multiclass_classification_data["y_test"]

    clf = TabNetClassifier(
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
    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=3,
        patience=2,
        batch_size=128,
        virtual_batch_size=64,
    )

    # Test predict
    y_pred = clf.predict(X_test)
    assert y_pred.shape == y_test.shape
    assert np.all(np.isin(y_pred, np.array([0, 1, 2])))

    # Test predict_proba
    y_pred_proba = clf.predict_proba(X_test)
    assert y_pred_proba.shape == (len(y_test), 3)
    assert np.allclose(y_pred_proba.sum(axis=1), 1.0)
    assert np.all((y_pred_proba >= 0) & (y_pred_proba <= 1))


def test_classifier_weights():
    """Test classifier with sample weights"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randint(0, 2, size=n_samples)

    # Create imbalanced dataset
    mask = y == 0
    X = np.vstack([X[mask][:100], X[~mask]])
    y = np.hstack([y[mask][:100], y[~mask]])

    X_train = X[:800]
    X_test = X[800:]
    y_train = y[:800]
    y_test = y[800:]

    # Test with auto weights
    clf = TabNetClassifier()
    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=3,
        weights=1
    )
    assert clf.class_weights is not None

    # Test with custom weights
    custom_weights = {0: 2.0, 1: 1.0}
    clf = TabNetClassifier()
    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=3,
        weights=custom_weights
    )
    assert clf.class_weights == custom_weights


def test_classifier_input_validation():
    """Test input validation"""
    clf = TabNetClassifier()

    # Test invalid X shape
    with pytest.raises(ValueError):
        X = np.random.randn(100)  # 1D array
        y = np.random.randint(0, 2, size=100)
        clf.fit(X, y)

    # Test mismatched X, y shapes
    with pytest.raises(ValueError):
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, size=50)  # Wrong length
        clf.fit(X, y)

    # Test invalid weights
    with pytest.raises(ValueError):
        X = np.random.randn(100, 10)
        y = np.random.randint(0, 2, size=100)
        clf.fit(X, y, weights=2)  # weights should be 0, 1 or dict

    # Test non-finite values
    with pytest.raises(ValueError):
        X = np.random.randn(100, 10)
        X[0, 0] = np.inf
        y = np.random.randint(0, 2, size=100)
        clf.fit(X, y)


def test_classifier_save_load(tmp_path, binary_classification_data):
    """Test model saving and loading"""
    X_train = binary_classification_data["X_train"]
    X_test = binary_classification_data["X_test"]
    y_train = binary_classification_data["y_train"]
    y_test = binary_classification_data["y_test"]

    clf = TabNetClassifier()
    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=3
    )

    # Save model
    save_path = tmp_path / "model.zip"
    clf.save_model(save_path)

    # Load model
    loaded_clf = TabNetClassifier()
    loaded_clf.load_model(save_path)

    # Compare predictions
    y_pred = clf.predict(X_test)
    y_pred_loaded = loaded_clf.predict(X_test)
    assert np.array_equal(y_pred, y_pred_loaded)


@pytest.mark.parametrize("batch_size", [16, 32, 64])
@pytest.mark.parametrize("virtual_batch_size", [8, 16, 32])
def test_classifier_batch_sizes(batch_size, virtual_batch_size, binary_classification_data):
    """Test different batch size configurations"""
    X_train = binary_classification_data["X_train"]
    X_test = binary_classification_data["X_test"]
    y_train = binary_classification_data["y_train"]
    y_test = binary_classification_data["y_test"]

    clf = TabNetClassifier()
    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=3,
        batch_size=batch_size,
        virtual_batch_size=virtual_batch_size
    )

    y_pred = clf.predict(X_test)
    assert y_pred.shape == y_test.shape


def test_classifier_early_stopping(binary_classification_data):
    """Test early stopping functionality"""
    X_train = binary_classification_data["X_train"]
    X_test = binary_classification_data["X_test"]
    y_train = binary_classification_data["y_train"]
    y_test = binary_classification_data["y_test"]

    clf = TabNetClassifier()
    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=100,
        patience=2
    )

    # Check if training stopped early
    assert clf.best_epoch < 100


def test_classifier_feature_selection(binary_classification_data):
    """Test feature selection and importance"""
    X_train = binary_classification_data["X_train"]
    X_test = binary_classification_data["X_test"]
    y_train = binary_classification_data["y_train"]
    y_test = binary_classification_data["y_test"]
    feature_names = binary_classification_data["feature_names"]

    clf = TabNetClassifier()
    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=3
    )

    # Test feature importance shape and values
    feat_imp = clf.feature_importances_
    assert len(feat_imp) == len(feature_names)
    assert np.all(feat_imp >= 0)
    assert np.sum(feat_imp) > 0

    # Test explain matrix
    explain_matrix, masks = clf.explain(X_test)
    assert explain_matrix.shape == (len(X_test), len(feature_names))
    assert isinstance(masks, dict)


def test_classifier_categorical_features(binary_classification_data):
    """Test classifier with categorical features"""
    X_train = binary_classification_data["X_train"]
    X_test = binary_classification_data["X_test"]
    y_train = binary_classification_data["y_train"]
    y_test = binary_classification_data["y_test"]

    # Add categorical features
    cat_feat1 = np.random.randint(0, 5, size=len(X_train))
    cat_feat2 = np.random.randint(0, 3, size=len(X_train))
    X_train = np.column_stack([X_train, cat_feat1, cat_feat2])

    cat_feat1_test = np.random.randint(0, 5, size=len(X_test))
    cat_feat2_test = np.random.randint(0, 3, size=len(X_test))
    X_test = np.column_stack([X_test, cat_feat1_test, cat_feat2_test])

    clf = TabNetClassifier(
        cat_idxs=[10, 11],  # Indices of categorical features
        cat_dims=[5, 3],    # Number of categories for each categorical feature
        cat_emb_dim=2       # Embedding dimension for categorical features
    )

    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_test, y_test)],
        max_epochs=3
    )

    y_pred = clf.predict(X_test)
    assert y_pred.shape == y_test.shape
