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
        
        This implementation follows Spark ML conventions by:
        1. Saving metadata in MLmodel format
        2. Using a dedicated model directory for artifacts
        3. Maintaining clear separation between model data and metadata
        
        Args:
            path: Path to save the model
            
        Raises:
            ValueError: If there's an error during saving
        """
        try:
            # Create a temporary directory for local operations
            temp_dir = tempfile.mkdtemp()
            
            # Create model directory for artifacts
            model_dir = os.path.join(temp_dir, "model")
            os.makedirs(model_dir, exist_ok=True)
            
            # Save torch model state dict
            model_path = os.path.join(model_dir, "model.pt")
            torch.save(self.instance._torch_model.state_dict(), model_path)
            
            # Prepare parameters
            params = {
                "input_dim": self.instance.input_dim,
                "output_dim": self.instance.output_dim,
                "inputCol": self.instance.getInputCol(),
                "outputCol": self.instance.getOutputCol(),
                **self.instance._get_model_params()
            }
            
            # Save parameters
            params_path = os.path.join(model_dir, "params.json")
            with open(params_path, "w") as f:
                json.dump(params, f, indent=2)
            
            # Create MLmodel metadata following Spark ML conventions
            metadata = {
                "class": "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel",
                "timestamp": int(time.time() * 1000),
                "sparkVersion": "3.4.0",  # Use appropriate version
                "uid": self.instance.uid,
                "params": params,
                "modelData": {
                    "format": "pytorch",
                    "path": "model/model.pt"
                },
                "parameterData": {
                    "format": "json",
                    "path": "model/params.json"
                }
            }
            
            # Log the save path for debugging
            logging.info(f"Saving TabNet model to path: {path}")
            
            # Save MLmodel metadata in root directory and model directory
            mlmodel_paths = [
                os.path.join(path, "MLmodel"),  # Root MLmodel
                os.path.join(path, "model", "MLmodel")  # Model dir MLmodel
            ]
            for mlmodel_path in mlmodel_paths:
                os.makedirs(os.path.dirname(mlmodel_path), exist_ok=True)
                with open(mlmodel_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                logging.info(f"Saved MLmodel to: {mlmodel_path}")

            # Copy model files to target path
            target_model_dir = os.path.join(path, "model")
            os.makedirs(os.path.dirname(target_model_dir), exist_ok=True)
            shutil.copytree(model_dir, target_model_dir, dirs_exist_ok=True)
            
            # Handle pipeline stage paths
            path_parts = path.split(os.sep)
            if "stages" in path_parts:
                stages_idx = path_parts.index("stages")
                if stages_idx < len(path_parts) - 1:
                    # Extract stage number from directory name (e.g., "2_TabNetModel_xyz" -> "2")
                    stage_dir = path_parts[stages_idx + 1]
                    stage_num = stage_dir.split("_")[0]
                    try:
                        int(stage_num)  # Validate it's a number
                        # Create clean stage path without model name/uid
                        pipeline_root = os.path.dirname(os.path.dirname(path))  # Go up two levels from stage dir
                        stage_path = os.path.abspath(os.path.join(pipeline_root, "stages", stage_num))
                        
                        # Create stage directories
                        stage_model_dir = os.path.join(stage_path, "model")
                        os.makedirs(stage_model_dir, exist_ok=True)
                        
                        # Copy model files from the target model directory
                        model_files = ["model.pt", "params.json"]
                        for file in model_files:
                            src = os.path.join(target_model_dir, file)
                            dst = os.path.join(stage_model_dir, file)
                            logging.info(f"Copying model file from {src} to {dst}")
                            shutil.copy2(src, dst)
                            logging.info(f"Successfully copied to {dst}")
                        
                        # Save MLmodel in both stage root and model directories
                        stage_mlmodel_paths = [
                            os.path.join(stage_path, "MLmodel"),  # Stage root MLmodel
                            os.path.join(stage_path, "model", "MLmodel")  # Stage model dir MLmodel
                        ]
                        for stage_mlmodel in stage_mlmodel_paths:
                            os.makedirs(os.path.dirname(stage_mlmodel), exist_ok=True)
                            with open(stage_mlmodel, "w") as f:
                                json.dump(metadata, f, indent=2)
                            logging.info(f"Saved stage MLmodel to: {stage_mlmodel}")
                    except (ValueError, IndexError) as e:
                        logging.warning(f"Failed to save stage MLmodel: {str(e)}")
            
            # Save Spark-compatible metadata using DefaultParamsWriter
            DefaultParamsWriter.saveMetadata(
                self.instance,
                path,
                self.sc,
                extraMetadata={
                    "class": "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel",
                    "timestamp": metadata["timestamp"],
                    "sparkVersion": metadata["sparkVersion"],
                    "uid": self.instance.uid
                }
            )
            
        except Exception as e:
            msg = f"Error saving model: {str(e)}"
            logging.error(msg)
            raise ValueError(msg)
        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)


class TabNetModelReader(MLReader["TabNetModel"]):
    """MLReader implementation for TabNetModel with improved loading."""
    
    def __init__(self):
        super().__init__()
    def load(self, path: str) -> "TabNetModel":
        """Load TabNetModel from disk or MLflow artifact store.
        
        This implementation:
        1. Handles MLflow URIs by downloading artifacts
        2. Reads metadata from MLmodel file
        3. Uses factory function to create TabNet model
        4. Validates loaded model parameters
        
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
                except mlflow.exceptions.MlflowException:
                    msg = "Model path does not exist and artifact store fallback failed"
                    logging.error(msg)
                    raise ValueError(msg)
            
            # Check if path exists after potential MLflow download
            if not os.path.exists(path):
                msg = "Model path does not exist and artifact store fallback failed"
                logging.error(msg)
                raise ValueError(msg)
                
            # Load Spark metadata using DefaultParamsReader
            try:
                metadata = DefaultParamsReader.loadMetadata(path, self.sc)
            except Exception:
                # If Spark metadata doesn't exist, try to create it from MLmodel
                mlmodel_path = os.path.join(path, "MLmodel")
                if os.path.exists(mlmodel_path):
                    with open(mlmodel_path, "r") as f:
                        mlmodel_metadata = json.load(f)
                    
                    # Save as Spark metadata
                    DefaultParamsWriter.saveMetadata(
                        TabNetModel(),  # Temporary instance for metadata
                        path,
                        self.sc,
                        extraMetadata={
                            "class": mlmodel_metadata.get("class", "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel"),
                            "timestamp": mlmodel_metadata.get("timestamp", int(time.time() * 1000)),
                            "sparkVersion": mlmodel_metadata.get("sparkVersion", "3.4.0"),
                            "uid": mlmodel_metadata.get("uid", "")
                        }
                    )
                    metadata = DefaultParamsReader.loadMetadata(path, self.sc)
            
            # Load MLmodel metadata
            mlmodel_path = os.path.join(path, "MLmodel")
            if not os.path.exists(mlmodel_path):
                msg = "MLmodel file not found"
                logging.error(msg)
                raise ValueError(msg)

            with open(mlmodel_path, "r") as f:
                metadata = json.load(f)

            # Verify metadata
            if metadata.get("class") != "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel":
                msg = f"Invalid model class in metadata: {metadata.get('class')}"
                logging.error(msg)
                raise ValueError(msg)

            # Load parameters
            params = metadata.get("params", {})
            if not params:
                msg = "No parameters found in metadata"
                logging.error(msg)
                raise ValueError(msg)

            # Create TabNetModel instance
            model = TabNetModel(
                inputCol=params["inputCol"],
                outputCol=params["outputCol"],
                input_dim=params["input_dim"],
                output_dim=params["output_dim"]
            )

            # Set model parameters
            model.setParams(**{k: v for k, v in params.items()
                             if k not in ["input_dim", "output_dim", "inputCol", "outputCol"]})

            # Create TabNet instance using factory
            model._torch_model = create_tabnet_model(params)

            # Load model state dict
            model_path = os.path.join(path, "model", "model.pt")
            if not os.path.exists(model_path):
                msg = "Model state dict file not found"
                logging.error(msg)
                raise ValueError(msg)

            model._torch_model.load_state_dict(torch.load(model_path))

            # Validate loaded model
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
    is saved, all stages (including TabNet) have proper metadata and
    can be loaded correctly.
    
    Args:
        pipeline_model: Pipeline model to save
        path: Path to save the pipeline
        overwrite: Whether to overwrite existing files
    """
    from pyspark.ml import PipelineModel
    
    if not isinstance(pipeline_model, PipelineModel):
        raise ValueError("Input must be a PipelineModel instance")
    
    # Create target directory
    if os.path.exists(path):
        if overwrite:
            shutil.rmtree(path)
        else:
            raise ValueError(f"Path {path} already exists")
    
    os.makedirs(path)
    
    # Save pipeline metadata
    metadata = {
        "class": "pyspark.ml.pipeline.PipelineModel",
        "timestamp": int(time.time() * 1000),
        "sparkVersion": "3.4.0",
        "uid": pipeline_model.uid,
        "stages": []
    }
    
    # Save each stage
    for i, stage in enumerate(pipeline_model.stages):
        stage_path = os.path.join(path, f"stages/{i}")
        os.makedirs(stage_path)
        
        # Save stage and ensure MLmodel exists
        stage.save(stage_path)
        
        # For TabNet stages, ensure MLmodel is in the correct location
        if isinstance(stage, TabNetModel):
            # Get metadata from the stage
            stage_metadata = {
                "class": "pytorch_tabnet.spark.tabnet_pyspark.TabNetModel",
                "timestamp": int(time.time() * 1000),
                "sparkVersion": "3.4.0",
                "uid": stage.uid,
                "params": {
                    "input_dim": stage.input_dim,
                    "output_dim": stage.output_dim,
                    "inputCol": stage.getInputCol(),
                    "outputCol": stage.getOutputCol(),
                    **stage._get_model_params()
                }
            }
            
            # For TabNet stages, save files in both original and MLflow paths
            if isinstance(stage, TabNetModel):
                # Create temporary directory for stage files
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save files in temporary directory
                    model_dir = os.path.join(temp_dir, "model")
                    os.makedirs(model_dir, exist_ok=True)
                    
                    # Save torch model state dict
                    model_path = os.path.join(model_dir, "model.pt")
                    torch.save(stage._torch_model.state_dict(), model_path)
                    
                    # Save parameters
                    params_path = os.path.join(model_dir, "params.json")
                    with open(params_path, "w") as f:
                        json.dump(stage_metadata["params"], f, indent=2)
                    
                    # Save MLmodel file
                    mlmodel_path = os.path.join(temp_dir, "MLmodel")
                    with open(mlmodel_path, "w") as f:
                        json.dump(stage_metadata, f, indent=2)
                    
                    # Save files in MLflow sparkml path with correct naming pattern
                    stage_name = f"{i}_TabNetModel_{stage.uid}"
                    stage_path = os.path.join(path, "sparkml", "stages", stage_name)
                    os.makedirs(stage_path, exist_ok=True)
                    
                    # Save MLmodel in stage root
                    stage_mlmodel = os.path.join(stage_path, "MLmodel")
                    with open(stage_mlmodel, "w") as f:
                        json.dump(stage_metadata, f, indent=2)
                    logging.info(f"Saved stage MLmodel to: {stage_mlmodel}")
                    
                    # Create model directory and save files
                    model_dir = os.path.join(stage_path, "model")
                    os.makedirs(model_dir, exist_ok=True)
                    
                    # Save model files
                    torch.save(stage._torch_model.state_dict(), os.path.join(model_dir, "model.pt"))
                    with open(os.path.join(model_dir, "params.json"), "w") as f:
                        json.dump(stage_metadata["params"], f, indent=2)
                    with open(os.path.join(model_dir, "MLmodel"), "w") as f:
                        json.dump(stage_metadata, f, indent=2)
                    
                    logging.info(f"Saved stage files to: {stage_path}")
        
        # Add stage metadata
        stage_metadata = {
            "class": stage.__class__.__module__ + "." + stage.__class__.__name__,
            "path": f"stages/{i}"
        }
        metadata["stages"].append(stage_metadata)
    
    # Save pipeline metadata
    metadata_path = os.path.join(path, "MLmodel")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)