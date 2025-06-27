import json
import logging
import os
import tempfile

import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME", "")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
# Define a cache directory that will be mounted as a volume
MLFLOW_CACHE_DIR = os.getenv("MLFLOW_CACHE_DIR", os.path.join(tempfile.gettempdir(), "mlflow_algorithm_cache"))

class MlflowAlgorithmClient:
    """
    Client for managing algorithm definitions in MLflow
    
    This class provides functionality to:
    1. Load algorithm definitions from MLflow
    2. Register new algorithms in MLflow
    3. Access algorithms using dictionary-like syntax (e.g., client["algorithm_name"])
    """
    def __init__(
        self, 
        tracking_uri=None,
        username=None, 
        password=None,
        cache_dir=None
    ):
        """
        Initialize the MLflow client with connection parameters.
        
        Args:
            tracking_uri: MLflow tracking server URI
            username: MLflow authentication username
            password: MLflow authentication password
            cache_dir: Directory to store cached models
        """
        self.algorithms = {}
        self.algorithm_names = []
        self.modelname_list = []  # For backward compatibility with Models class
        
        # Setup MLflow connection parameters
        self.tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        self.username = username or os.getenv("MLFLOW_TRACKING_USERNAME", "")
        self.password = password or os.getenv("MLFLOW_TRACKING_PASSWORD", "")
        self.cache_dir = cache_dir or MLFLOW_CACHE_DIR
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set environment variables
        os.environ['MLFLOW_TRACKING_USERNAME'] = self.username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = self.password
        
        # Set tracking URI
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create client
        self.client = MlflowClient()
        
    def load_from_mlflow(self, algorithm_type=None):
        """
        Load algorithm definitions from MLflow
        
        Args:
            algorithm_type: Optional filter for algorithm type
            
        Returns:
            bool: True if algorithms were loaded successfully, False otherwise
        """
        try:
            # Search for models with the algorithm_definition tag
            filter_string = "tags.entity_type = 'algorithm_definition'"
            if algorithm_type:
                filter_string += f" AND tags.algorithm_type = '{algorithm_type}'"
                
            registered_models = self.client.search_registered_models(filter_string=filter_string)
            
            if not registered_models:
                logger.warning("No algorithm definitions found in MLflow")
                return False
                
            # Reset algorithm collections
            self.algorithms = {}
            self.algorithm_names = []
            
            for model in registered_models:
                # Get latest version
                versions = self.client.get_latest_versions(model.name)
                if not versions:
                    continue
                    
                version = versions[0]
                
                # Get run to access artifacts
                try:
                    run = self.client.get_run(version.run_id)
                    
                    # Download the config artifact
                    download_path = os.path.join(self.cache_dir, model.name)
                    os.makedirs(download_path, exist_ok=True)
                    artifact_path = os.path.join(download_path, "algorithm_config.json")
                    
                    self.client.download_artifacts(run.info.run_id, "algorithm_config.json", download_path)
                    with open(artifact_path, "r") as f:
                        algorithm_config = json.load(f)
                    
                    # Add to algorithms dict
                    self.algorithm_names.append(model.name)
                    self.algorithms[model.name] = algorithm_config
                    
                except Exception as e:
                    logger.warning(f"Error loading algorithm {model.name}: {e}")
                    continue
            
            # For backward compatibility with Models class
            self.modelname_list = self.algorithm_names
            return len(self.algorithms) > 0
            
        except Exception as e:
            logger.warning(f"Failed to load algorithms from MLflow: {e}")
            return False
                
    def register_algorithm(self, algorithm_config, overwrite=False):
        """
        Register an algorithm definition in MLflow
        
        Args:
            algorithm_config (dict): Algorithm configuration with GUI parameters
            overwrite (bool): Whether to overwrite if algorithm already exists
            
        Returns:
            dict: Registration result with model name and version
        """
        # Extract basic information
        model_name = algorithm_config.get("model_name")
        if not model_name:
            raise ValueError("Algorithm configuration must include 'model_name'")
        
        # Check if algorithm already exists
        try:
            existing_versions = self.client.get_latest_versions(model_name)
            if existing_versions and not overwrite:
                logger.info(f"Algorithm '{model_name}' already exists. Use overwrite=True to replace it.")
                return {
                    "status": "exists",
                    "model_name": model_name,
                    "version": existing_versions[0].version,
                    "message": "Algorithm already exists"
                }
        except Exception:
            # If we get an error, the model probably doesn't exist, so continue
            pass
        
        algorithm_type = algorithm_config.get("type")
        experiment_name = f"Algorithm Registry - {algorithm_type}"
        
        # Create or get experiment
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = self.client.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            logger.warning(f"Error creating experiment: {e}")
            experiment_id = "0"  # Default experiment
        
        # Load environment variables
        READ_DIR_MOUNT = os.getenv("READ_DIR_MOUNT", "")
        CONTAINER_NETWORK = os.getenv("CONTAINER_NETWORK", "")
        PARTITIONS_CPU = json.loads(os.getenv("PARTITIONS_CPU", "[]"))
        RESERVATIONS_CPU = json.loads(os.getenv("RESERVATIONS_CPU", "[]"))
        MAX_TIME_CPU = os.getenv("MAX_TIME_CPU", "1:00:00")
        SUBMISSION_SSH_KEY = os.getenv("SUBMISSION_SSH_KEY", "")
        FORWARD_PORTS = json.loads(os.getenv("FORWARD_PORTS", "[]"))
        
        # Start MLflow run to log algorithm definition
        with mlflow.start_run(experiment_id=experiment_id) as run:
            # Log basic algorithm metadata
            mlflow.set_tag("algorithm_type", algorithm_type)
            mlflow.set_tag("entity_type", "algorithm_definition")  # Important tag!
            mlflow.set_tag("version", algorithm_config.get("version", "0.0.1"))
            mlflow.set_tag("owner", algorithm_config.get("owner", "mlexchange team"))
            
            # Log container info
            mlflow.log_param("image_name", algorithm_config.get("image_name", ""))
            mlflow.log_param("image_tag", algorithm_config.get("image_tag", ""))
            mlflow.log_param("conda_env", algorithm_config.get("conda_env", ""))
            mlflow.log_param("network", CONTAINER_NETWORK)
            mlflow.log_param("volumes", json.dumps([f"{READ_DIR_MOUNT}:/tiled_storage"]))
            
            # Log Slurm info
            mlflow.log_param("num_nodes", 1)
            mlflow.log_param("partitions", json.dumps(PARTITIONS_CPU))
            mlflow.log_param("reservations", json.dumps(RESERVATIONS_CPU))
            mlflow.log_param("max_time", MAX_TIME_CPU)
            mlflow.log_param("submission_ssh_key", SUBMISSION_SSH_KEY)
            mlflow.log_param("forward_ports", json.dumps(FORWARD_PORTS))
            
            # Log file paths
            python_files = algorithm_config.get("python_file_name", {})
            if isinstance(python_files, dict):
                for op, path in python_files.items():
                    mlflow.log_param(f"python_file_{op}", path)
            else:
                mlflow.log_param("python_file", python_files)
            
            # Log applications
            applications = algorithm_config.get("application", [])
            mlflow.log_param("applications", json.dumps(applications))
            
            # Log description
            mlflow.log_param("description", algorithm_config.get("description", ""))
            
            # Save complete algorithm config
            temp_dir = os.path.join(self.cache_dir, "artifacts")
            os.makedirs(temp_dir, exist_ok=True)
            temp_file = os.path.join(temp_dir, "algorithm_config.json")
            with open(temp_file, "w") as f:
                json.dump(algorithm_config, f, indent=2)
            mlflow.log_artifact(temp_file)
            
            # Register the algorithm in the model registry
            try:
                model_details = mlflow.register_model(
                    f"runs:/{run.info.run_id}/algorithm_config.json", 
                    model_name
                )
                
                # Set tags on registered model 
                self.client.set_registered_model_tag(model_name, "entity_type", "algorithm_definition")
                self.client.set_registered_model_tag(model_name, "algorithm_type", algorithm_type)
                
                # Set tags on model version
                self.client.set_model_version_tag(
                    model_name, 
                    model_details.version, 
                    "entity_type", 
                    "algorithm_definition"
                )
                
                # Reload algorithms to include the newly registered one
                self.load_from_mlflow(algorithm_type)
                
                return {
                    "status": "success",
                    "model_name": model_name,
                    "version": model_details.version,
                    "run_id": run.info.run_id
                }
                
            except Exception as e:
                logger.error(f"Failed to register algorithm: {e}")
                return {
                    "status": "error",
                    "model_name": model_name,
                    "error": str(e)
                }
    
    def __getitem__(self, key):
        """
        Access algorithms by name using dictionary syntax.
        Example: client["algorithm_name"]
        
        Args:
            key: Name of the algorithm
            
        Returns:
            dict: Algorithm configuration
            
        Raises:
            KeyError: If the algorithm doesn't exist
        """
        try:
            return self.algorithms[key]
        except KeyError:
            raise KeyError(f"An algorithm with name '{key}' does not exist.")
    
    def is_connected(self):
        """
        Check if this client instance is still connected to MLflow.
        
        Returns:
            bool: True if MLflow server is reachable, False otherwise
        """
        try:
            # Perform a lightweight API call to verify connectivity
            self.client.search_experiments(max_results=1)
            logger.info("MLflow server is reachable")
            return True
        except Exception as e:
            logger.warning(f"MLflow server is not reachable: {e}")
            return False
            
    @staticmethod
    def check_mlflow_ready(tracking_uri=None, username=None, password=None):
        """
        Static method to check if MLflow is available and configured correctly
        
        Args:
            tracking_uri: MLflow tracking server URI
            username: MLflow authentication username
            password: MLflow authentication password
            
        Returns:
            bool: True if MLflow server is reachable, False otherwise
        """
        try:
            tracking_uri = tracking_uri or MLFLOW_TRACKING_URI
            username = username or MLFLOW_TRACKING_USERNAME
            password = password or MLFLOW_TRACKING_PASSWORD
            
            os.environ["MLFLOW_TRACKING_USERNAME"] = username
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient()
            # Try a simple operation
            client.search_experiments(max_results=1)
            logger.info("MLflow server is reachable")
            return True
        except Exception as e:
            logger.warning(f"MLflow server is not reachable: {e}")
            return False