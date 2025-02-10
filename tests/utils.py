import torch
from torch.nn import Module
import numpy as np


from pytorch_tabnet.tab_network import TabNet as RealTabNet

class DummyTabNet(torch.nn.Module):
    """A lightweight TabNet implementation for testing purposes.
    
    This implementation provides a minimal TabNet interface for testing,
    implementing only the essential methods needed for the tests.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
        group_attention_matrix=None,
    ):
        super(DummyTabNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.cat_idxs = cat_idxs
        self.cat_dims = cat_dims
        self.cat_emb_dim = cat_emb_dim
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.epsilon = epsilon
        self.virtual_batch_size = virtual_batch_size
        self.momentum = momentum
        self.mask_type = mask_type
        
        # Create minimal learnable parameters for state_dict
        self.dummy_param = torch.nn.Parameter(torch.randn(1))
        self.embedder = DummyEmbedder(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.feature_transformer = DummyFeatureTransformer(input_dim, output_dim)
        
    def forward(self, x):
        """Simplified forward pass returning dummy predictions."""
        batch_size = x.shape[0]
        # Use feature transformer for more realistic behavior
        predictions = self.feature_transformer(x)
        # Always return 0.0 for testing consistency
        predictions = torch.zeros_like(predictions)
        attention_loss = torch.tensor(0.0)
        return predictions, attention_loss
    
    def forward_masks(self, x):
        """Simplified mask generation."""
        batch_size = x.shape[0]
        dummy_masks = torch.ones((batch_size, self.input_dim))
        return dummy_masks, {"step0": dummy_masks}
    
    def state_dict(self, *args, **kwargs):
        """Override state_dict to ensure proper parameter handling."""
        state = super().state_dict(*args, **kwargs)
        # Add metadata that real TabNet would include
        state['input_dim'] = self.input_dim
        state['output_dim'] = self.output_dim
        state['n_d'] = self.n_d
        state['n_steps'] = self.n_steps
        state['cat_dims'] = self.cat_dims
        state['cat_emb_dim'] = self.cat_emb_dim
        state['mask_type'] = self.mask_type
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        """Override load_state_dict to handle test state."""
        # Extract metadata if present (for compatibility testing)
        if 'input_dim' in state_dict:
            self.input_dim = state_dict.pop('input_dim')
        if 'output_dim' in state_dict:
            self.output_dim = state_dict.pop('output_dim')
        if 'n_d' in state_dict:
            self.n_d = state_dict.pop('n_d')
        if 'n_steps' in state_dict:
            self.n_steps = state_dict.pop('n_steps')
        if 'cat_dims' in state_dict:
            self.cat_dims = state_dict.pop('cat_dims')
        if 'cat_emb_dim' in state_dict:
            self.cat_emb_dim = state_dict.pop('cat_emb_dim')
        if 'mask_type' in state_dict:
            self.mask_type = state_dict.pop('mask_type')
            
        return super().load_state_dict(state_dict, strict=False)


class DummyEmbedder(Module):
    """Minimal embedder implementation for testing."""
    def __init__(self, input_dim, cat_dims, cat_idxs, cat_emb_dim):
        super().__init__()
        self.input_dim = input_dim
        self.post_embed_dim = input_dim  # Simplified
        self.embedding_group_matrix = torch.eye(input_dim)  # Identity matrix
        self.dummy_embed = torch.nn.Parameter(torch.randn(1))
        
    def forward(self, x):
        """Pass through without embedding."""
        return x


class DummyFeatureTransformer(Module):
    """Feature transformer that mimics TabNet's feature processing."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Shared feature processing layers
        self.shared = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.BatchNorm1d(input_dim)
        ])
        # Feature normalization
        self.feature_bn = torch.nn.BatchNorm1d(input_dim)
        # Attention mechanism
        self.attention = torch.nn.Linear(input_dim, input_dim)
        # Output transformation
        self.output = torch.nn.Linear(input_dim, output_dim)
        # Learnable scale parameter
        self.scale = torch.nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        """More realistic feature transformation with attention."""
        # Shared feature processing
        features = x
        for layer in self.shared:
            features = layer(features)
            features = torch.relu(features)
        
        # Feature normalization
        features = self.feature_bn(features)
        
        # Attention mechanism
        attention_weights = torch.sigmoid(self.attention(features))
        features = features * attention_weights * self.scale
        
        # Output transformation
        return self.output(features)


def create_test_data(spark, n_samples=100, n_features=10):
    """Create test data for TabNet Spark integration tests."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    
    # Convert to Spark DataFrame
    data = [(X[i].tolist(), float(y[i])) for i in range(n_samples)]
    from pyspark.sql.types import ArrayType, DoubleType, StructField, StructType
    schema = StructType([
        StructField("features", ArrayType(DoubleType())),
        StructField("label", DoubleType())
    ])
    return spark.createDataFrame(data, schema)