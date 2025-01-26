import pytest
import torch
from pytorch_tabnet.data import TabularDataProvider, TabularDataBatch

class DummyProvider(TabularDataProvider):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        
    def __iter__(self):
        X = self.X.split(2)
        y = None if self.y is None else self.y.split(2)
        return (TabularDataBatch(x, y) for x, y in zip(X, y))
        
    def __len__(self):
        return (len(self.X) + 1) // 2

def test_data_provider_iter():
    X = torch.randn(10, 5)
    y = torch.randn(10)
    
    provider = DummyProvider(X, y)
    batches = list(iter(provider))
    
    assert len(batches) == 5
    assert all(batch.X.shape == (2, 5) for batch in batches)
    assert all(batch.y.shape == (2,) for batch in batches)
    
def test_data_provider_len():
    X = torch.randn(7, 3)
    provider = DummyProvider(X)
    assert len(provider) == 4
    
def test_pin_memory():
    X = torch.randn(4, 3)
    y = torch.randn(4)
    batch = TabularDataBatch(X, y)
    
    batch_pin = batch.pin_memory()
    assert batch_pin is not batch
    assert batch_pin.X.is_pinned() == torch.cuda.is_available()
    assert batch_pin.y.is_pinned() == torch.cuda.is_available()