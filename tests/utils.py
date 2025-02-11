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
        torch.nn.Module.__init__(self)
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
        self.post_embed_dim = self.embedder.post_embed_dim
        self.feature_transformer = DummyFeatureTransformer(self.post_embed_dim, output_dim)
        
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
        torch.nn.Module.__init__(self)
        self.input_dim = input_dim
        self.cat_dims = cat_dims or []
        self.cat_idxs = cat_idxs or []
        
        # Handle cat_emb_dim properly like real TabNet
        if isinstance(cat_emb_dim, int):
            self.cat_emb_dims = [cat_emb_dim] * len(self.cat_dims)
        else:
            self.cat_emb_dims = cat_emb_dim
            
        # Calculate post_embed_dim exactly like real TabNet
        if self.cat_dims and self.cat_idxs:
            self.post_embed_dim = int(input_dim + np.sum(self.cat_emb_dims) - len(self.cat_idxs))
        else:
            self.post_embed_dim = input_dim
            
        # Create embeddings for categorical features
        self.embeddings = torch.nn.ModuleList()
        if self.cat_dims:
            for cat_dim, emb_dim in zip(self.cat_dims, self.cat_emb_dims):
                self.embeddings.append(torch.nn.Embedding(cat_dim, emb_dim))
                
        self.embedding_group_matrix = torch.eye(self.post_embed_dim)
        
    def forward(self, x):
        """Process input with proper categorical embedding."""
        if not self.cat_dims:
            return x
            
        # Split categorical and continuous features
        cat_features = []
        cont_features = []
        
        for i in range(self.input_dim):
            if i in self.cat_idxs:
                cat_idx = self.cat_idxs.index(i)
                cat_dim = self.cat_dims[cat_idx]
                cat_emb = self.embeddings[cat_idx]
                cat_col = x[:, i].long()
                embedded_col = cat_emb(cat_col)
                cat_features.append(embedded_col)
            else:
                cont_features.append(x[:, i].unsqueeze(1))
                
        # Combine features
        if cont_features:
            cont_features = torch.cat(cont_features, dim=1)
        else:
            cont_features = torch.empty((x.shape[0], 0), device=x.device)
            
        if cat_features:
            cat_features = torch.cat(cat_features, dim=1)
            return torch.cat([cont_features, cat_features], dim=1)
        else:
            return cont_features


class DummyFeatureTransformer(Module):
    """Feature transformer that mimics TabNet's feature processing."""
    def __init__(self, input_dim, output_dim):
        torch.nn.Module.__init__(self)
        # Match real TabNet's feature processing
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.shared = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, input_dim),
            torch.nn.BatchNorm1d(input_dim)
        ])
        # Feature normalization matching TabNet's behavior
        self.feature_bn = torch.nn.BatchNorm1d(input_dim)
        # Attention mechanism with proper dimensions
        self.attention = torch.nn.Linear(input_dim, input_dim)
        # Output transformation with correct dimensions
        self.output = torch.nn.Linear(input_dim, output_dim)
        # Learnable scale parameter
        self.scale = torch.nn.Parameter(torch.ones(1))
        # Add post_embed_dim to match real TabNet
        self.post_embed_dim = input_dim
        
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