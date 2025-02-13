"""TabNet model persistence utilities."""

import os
import json
import time
import logging
import tempfile
import shutil
import torch
import mlflow
from typing import Dict, Any

from pyspark.ml.util import MLWriter, MLReader, DefaultParamsWriter, DefaultParamsReader
from pytorch_tabnet.spark.factory import create_tabnet_model, validate_loaded_model
from pytorch_tabnet.spark.tabnet_pyspark import TabNetModel


class TabNetModelWriter(MLWriter):
    """MLWriter implementation for TabNetModel with improved persistence."""
    
    def __init__(self, instance: TabNetModel):
        super().__init__()
        self.instance = instance
    
    def saveImpl(self, path: str) -> None:
        """Implementation of save logic for TabNetModel.
        
        This implementation follows MLflow Spark conventions by:
        1. Creating a flat artifact structure with a top-level MLmodel file
        2. Storing model artifacts in the sparkml directory
        3. Following MLflow's expected directory structure for both standalone and pipeline models
        
        Args:
            path: Path to save the model
            
        Raises:
            ValueError: If there's an error during saving
        """
        try:
            # Prepare parameters
            params = {
                "input_dim": self.instance.input_dim,
                "output_dim": self.instance.output_dim,
                "inputCol": self.instance.getInputCol(),
                "outputCol": self.instance.getOutputCol(),
                **self.instance._get_model_params()
            }
            
            # Create base metadata
            metadata = {
                "class": "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel",
                "timestamp": int(time.time() * 1000),
                "sparkVersion": "3.4.0",
                "uid": self.instance.uid,
                "params": params
            }
            
            # Check if this is a pipeline stage
            is_pipeline_stage = "stages" in path.split(os.sep)
            
            # Save Spark-compatible metadata
            DefaultParamsWriter.saveMetadata(
                self.instance,
                path,
                self.sc,
                extraMetadata=metadata
            )
            
            # Create sparkml directory
            sparkml_dir = os.path.join(path, "sparkml")
            os.makedirs(sparkml_dir, exist_ok=True)
            
            # Save model artifacts in sparkml directory
            torch.save(self.instance._torch_model.state_dict(), os.path.join(sparkml_dir, "model.pt"))
            
            with open(os.path.join(sparkml_dir, "params.json"), "w") as f:
                json.dump(params, f, indent=2)
                
            with open(os.path.join(sparkml_dir, "metadata"), "w") as f:
                json.dump(metadata, f, indent=2)
            
            # Create MLmodel metadata with proper paths
            mlmodel_dict = {
                **metadata,
                "flavors": {
                    "spark": {
                        "model_data": "sparkml",
                        "spark_version": "3.4.0"
                    },
                    "python_function": {
                        "loader_module": "mlflow.spark",
                        "model_path": "sparkml"
                    }
                }
            }
            
            # Save MLmodel file at root level (for both standalone and pipeline stages)
            mlmodel_path = os.path.join(path, "MLmodel")
            with open(mlmodel_path, "w") as f:
                json.dump(mlmodel_dict, f, indent=2)
            
            # Save environment files only for standalone models
            if not is_pipeline_stage:
                with open(os.path.join(path, "requirements.txt"), "w") as f:
                    f.write("mlflow==2.12.2\n")
                    f.write("pyspark==3.4.0\n")
                    f.write("torch==2.2.1\n")
                
                with open(os.path.join(path, "python_env.yaml"), "w") as f:
                    f.write("python: 3.11.9\n")
                    f.write("build_dependencies:\n")
                    f.write("  - pip\n")
                    f.write("dependencies:\n")
                    f.write("  - python=3.11.9\n")
                    f.write("  - pip:\n")
                    f.write("    - mlflow==2.12.2\n")
                    f.write("    - pyspark==3.4.0\n")
                    f.write("    - torch==2.2.1\n")
            
        except Exception as e:
            msg = f"Error saving model: {str(e)}"
            logging.error(msg)
            raise ValueError(msg)
        finally:
            pass  # No cleanup needed


class TabNetModelReader(MLReader["TabNetModel"]):
    """MLReader implementation for TabNetModel with improved loading."""
    
    def __init__(self):
        super().__init__()
        
    def load(self, path: str) -> "TabNetModel":
        """Load TabNetModel from disk or MLflow artifact store.
        
        This implementation follows MLflow's expected directory structure:
        1. Reads metadata from top-level MLmodel file
        2. Loads model artifacts from sparkml directory
        3. Creates and validates the TabNet model
        
        Args:
            path: Path to the saved model or MLflow URI
            
        Returns:
            Loaded TabNetModel instance
            
        Raises:
            ValueError: If model cannot be loaded
        """
        try:
            # Handle MLflow URI
            if path.startswith("mlflow://"):
                try:
                    run_id = path.split("/")[2]
                    artifact_path = "/".join(path.split("/")[3:])
                    path = mlflow.artifacts.download_artifacts(
                        run_id=run_id,
                        artifact_path=artifact_path
                    )
                except mlflow.exceptions.MlflowException as e:
                    raise ValueError("Model path does not exist and artifact store fallback failed")
            
            # Check if path exists
            if not os.path.exists(path):
                raise ValueError("Model path does not exist")
            
            # Load MLmodel metadata
            mlmodel_path = os.path.join(path, "MLmodel")
            if not os.path.exists(mlmodel_path):
                raise ValueError("Model path does not exist and artifact store fallback failed")
            
            with open(mlmodel_path, "r") as f:
                metadata = json.load(f)
            
            # Verify metadata
            if metadata.get("class") != "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel":
                raise ValueError(f"Invalid model class in metadata: {metadata.get('class')}")
            
            params = metadata.get("params", {})
            if not params:
                raise ValueError("No parameters found in metadata")
            
            # Create TabNetModel instance
            model = TabNetModel(
                inputCol=params["inputCol"],
                outputCol=params["outputCol"],
                input_dim=params["input_dim"],
                output_dim=params["output_dim"]
            )
            
            model.setParams(**{k: v for k, v in params.items()
                             if k not in ["input_dim", "output_dim", "inputCol", "outputCol"]})
            
            # Create TabNet instance using factory
            model._torch_model = create_tabnet_model(params)
            
            # Load model state dict from sparkml directory
            model_path = os.path.join(path, "sparkml", "model.pt")
            if not os.path.exists(model_path):
                raise ValueError("Model state dict not found in sparkml directory")
            
            model._torch_model.load_state_dict(torch.load(model_path))
            validate_loaded_model(model._torch_model, params)
            
            return model
        
        except Exception as e:
            msg = f"Error loading model: {str(e)}"
            logging.error(msg)
            raise ValueError(msg)


def save_pipeline_model(
    pipeline_model: "PipelineModel",
    path: str,
    overwrite: bool = False
) -> None:
    """Save a pipeline model containing TabNet, ensuring proper metadata.
    
    This utility function ensures that when a pipeline containing TabNet
    is saved, all stages (including TabNet) follow MLflow's expected
    directory structure with proper metadata and artifact organization.
    
    Args:
        pipeline_model: Pipeline model to save
        path: Path to save the pipeline
        overwrite: Whether to overwrite existing files
    """
    from pyspark.ml import PipelineModel
    
    if not isinstance(pipeline_model, PipelineModel):
        raise ValueError("Input must be a PipelineModel instance")
    
    if os.path.exists(path):
        if overwrite:
            shutil.rmtree(path)
        else:
            raise ValueError(f"Path {path} already exists")
    
    os.makedirs(path)
    
    # Create stages directory at root level
    stages_path = os.path.join(path, "stages")
    os.makedirs(stages_path, exist_ok=True)
    
    # Create sparkml directory for pipeline
    sparkml_path = os.path.join(path, "sparkml")
    os.makedirs(sparkml_path, exist_ok=True)
    
    # Create pipeline metadata with MLflow flavors
    metadata = {
        "class": "pyspark.ml.pipeline.PipelineModel",
        "timestamp": int(time.time() * 1000),
        "sparkVersion": "3.4.0",
        "uid": pipeline_model.uid,
        "stages": [],
        "flavors": {
            "spark": {
                "model_data": "sparkml",
                "spark_version": "3.4.0"
            },
            "python_function": {
                "loader_module": "mlflow.spark",
                "model_path": "sparkml"
            }
        }
    }
    
    for i, stage in enumerate(pipeline_model.stages):
        # Create stage directory with simple numeric index
        stage_path = os.path.join(stages_path, str(i))
        os.makedirs(stage_path, exist_ok=True)
        
        if isinstance(stage, TabNetModel):
            # Create stage directory under sparkml/stages with index prefix
            stage_name = f"{i}_{stage.__class__.__name__}_{stage.uid.split('_')[-1]}"
            stage_path = os.path.join(stages_path, stage_name)
            os.makedirs(stage_path, exist_ok=True)
            
            # Create sparkml directory for stage artifacts
            stage_sparkml_path = os.path.join(stage_path, "sparkml")
            os.makedirs(stage_sparkml_path, exist_ok=True)
            
            # Prepare stage metadata
            stage_params = {
                "input_dim": stage.input_dim,
                "output_dim": stage.output_dim,
                "inputCol": stage.getInputCol(),
                "outputCol": stage.getOutputCol(),
                **stage._get_model_params()
            }
            
            stage_metadata = {
                "class": "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel",
                "timestamp": int(time.time() * 1000),
                "sparkVersion": "3.4.0",
                "uid": stage.uid,
                "params": stage_params,
                "flavors": {
                    "spark": {
                        "model_data": "sparkml",
                        "spark_version": "3.4.0"
                    },
                    "python_function": {
                        "loader_module": "mlflow.spark",
                        "model_path": "sparkml"
                    }
                }
            }
            
            # Save model artifacts in sparkml directory
            torch.save(stage._torch_model.state_dict(), os.path.join(stage_sparkml_path, "model.pt"))
            
            # Save metadata files in sparkml directory
            with open(os.path.join(stage_sparkml_path, "params.json"), "w") as f:
                json.dump(stage_params, f, indent=2)
            
            with open(os.path.join(stage_sparkml_path, "metadata"), "w") as f:
                json.dump(stage_metadata, f, indent=2)
            
            # Save MLmodel at stage root
            with open(os.path.join(stage_path, "MLmodel"), "w") as f:
                json.dump(stage_metadata, f, indent=2)
            
            logging.info(f"Saved TabNet stage to: {stage_path}")
        else:
            # For non-TabNet stages, use standard save
            stage.save(os.path.join(stages_path, str(i)))
        # Add stage to pipeline metadata with proper path
        stage_class = stage.__class__.__module__ + "." + stage.__class__.__name__
        metadata["stages"].append({
            "class": stage_class,
            "path": f"sparkml/stages/{i}"
        })
        
        # Update stage metadata to point to correct paths
        if isinstance(stage, TabNetModel):
            stage_metadata = {
                "class": "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel",
                "timestamp": int(time.time() * 1000),
                "sparkVersion": "3.4.0",
                "uid": stage.uid,
                "params": stage_params,
                "flavors": {
                    "spark": {
                        "model_data": "sparkml",
                        "spark_version": "3.4.0"
                    },
                    "python_function": {
                        "loader_module": "mlflow.spark",
                        "model_path": "sparkml"
                    }
                }
            }
            
            # Update MLmodel at stage root
            stage_mlmodel_path = os.path.join(stages_path, str(i), "MLmodel")
            with open(stage_mlmodel_path, "w") as f:
                json.dump(stage_metadata, f, indent=2)
    
    # Save pipeline MLmodel file
    with open(os.path.join(path, "MLmodel"), "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Save environment files
    with open(os.path.join(path, "requirements.txt"), "w") as f:
        f.write("mlflow==2.12.2\n")
        f.write("pyspark==3.4.0\n")
        f.write("torch==2.2.1\n")
    
    with open(os.path.join(path, "python_env.yaml"), "w") as f:
        f.write("python: 3.11.9\n")
        f.write("build_dependencies:\n")
        f.write("  - pip\n")
        f.write("dependencies:\n")
        f.write("  - python=3.11.9\n")
        f.write("  - pip:\n")
        f.write("    - mlflow==2.12.2\n")
        f.write("    - pyspark==3.4.0\n")
        f.write("    - torch==2.2.1\n")