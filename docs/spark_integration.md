# TabNet Spark Integration

This document describes the architecture and usage of TabNet's Spark integration.

## Architecture Overview

The Spark integration enables TabNet to efficiently process large-scale tabular data using Apache Spark. The integration is built around the following key components:

### SparkDataProviderV2

The main class responsible for streaming data from Spark to TabNet. It implements the `TabularDataProvider` interface and provides:

- Memory-efficient data loading using Arrow optimization
- Configurable batch sizes and prefetching
- Proper tensor type conversion

```python
from pytorch_tabnet.spark_data_v2 import SparkDataProviderV2

provider = SparkDataProviderV2(
    spark_df,
    feature_cols=['feature1', 'feature2'],
    target_col='target',
    batch_size=1000
)
```

### Data Flow

1. **Initialization**:
   - Validate input DataFrame and columns
   - Configure Arrow optimization
   - Cache and optimize DataFrame partitioning

2. **Data Loading**:
   - Convert Spark DataFrame to pandas using Arrow
   - Process data in memory-efficient batches
   - Convert to PyTorch tensors

3. **Memory Management**:
   - Smart partition calculation based on data size
   - Efficient batch processing to bound memory usage
   - Proper cleanup of intermediate data

## Usage Guidelines

### Basic Usage

```python
from pyspark.sql import SparkSession
from pytorch_tabnet.spark_data_v2 import SparkDataProviderV2

# Create Spark DataFrame
spark = SparkSession.builder.getOrCreate()
df = spark.read.parquet("data.parquet")

# Create data provider
provider = SparkDataProviderV2(
    df,
    feature_cols=['feature1', 'feature2'],
    target_col='target',
    batch_size=1000
)

# Use with TabNet model
model.fit(provider)
```

### Memory Optimization

The provider automatically optimizes memory usage by:

1. Calculating optimal batch sizes based on available memory
2. Using Arrow for efficient data transfer
3. Processing data in streaming fashion

You can control memory usage through:

```python
provider = SparkDataProviderV2(
    df,
    feature_cols=feature_cols,
    target_col='target',
    batch_size=1000,          # Control batch size
    prefetch_batches=2,       # Control prefetch queue size
    arrow_max_records=10000   # Control Arrow batch size
)
```

### Performance Tuning

For optimal performance:

1. **Partition Sizing**: The provider automatically calculates optimal partitions, but you can influence this by:
   - Properly sizing your Spark cluster
   - Setting appropriate batch sizes

2. **Arrow Optimization**: Arrow is enabled by default for better performance. Ensure your Spark cluster has sufficient memory for Arrow processing.

3. **Data Caching**: The provider caches the optimized DataFrame. Ensure your Spark cluster has sufficient memory for caching.

## Migration Guide

If you're using the old `SparkDataProvider`, migrate to `SparkDataProviderV2` for:

- Better memory efficiency
- Improved performance
- More configuration options

The interface remains compatible:

```python
# Old code
from pytorch_tabnet.spark_data import SparkDataProvider
provider = SparkDataProvider(df, feature_cols, target_col)

# New code
from pytorch_tabnet.spark_data_v2 import SparkDataProviderV2
provider = SparkDataProviderV2(df, feature_cols, target_col)
```

## Best Practices

1. **Data Preparation**:
   - Properly handle missing values before creating the provider
   - Ensure consistent data types across columns
   - Consider partitioning large datasets appropriately

2. **Resource Management**:
   - Monitor memory usage during training
   - Adjust batch sizes based on your data and cluster size
   - Use appropriate Spark configurations for your cluster

3. **Error Handling**:
   - Validate input data before creating the provider
   - Handle potential out-of-memory conditions
   - Monitor Spark job progress

## API Reference

### SparkDataProviderV2

```python
class SparkDataProviderV2:
    """Memory-efficient data provider for PySpark DataFrames.
    
    Args:
        df: PySpark DataFrame containing features and target
        feature_cols: List of feature column names
        target_col: Optional target column name
        batch_size: Number of rows per batch (default: 1000)
        prefetch_batches: Number of batches to prefetch (default: 2)
        arrow_max_records: Maximum records per Arrow batch (default: 10000)
    """
```

For detailed API documentation, see the class docstrings.