import json
import mlflow
from mlflow.tracking import MlflowClient
import logging
from src.logger import logging
import os
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "lakshmanbunny"
repo_name = "MLOPS-CAPSTONE-PROJECT"
# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri('https://dagshub.com/lakshmanbunny/MLOPS-CAPSTONE-PROJECT.mlflow')
# dagshub.init(repo_owner='lakshmanbunny', repo_name='MLOPS-CAPSTONE-PROJECT', mlflow=True)
# -------------------------------------------------------------------------------------


def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry using MlflowClient."""
    try:
        run_id = model_info['run_id']
        model_path = model_info['model_path']
        
        client = MlflowClient()
        
        # 1. Fetch Run Info to get the exact artifact URI
        logging.info(f"Fetching Run Info for {run_id}...")
        run = client.get_run(run_id)
        artifact_uri = run.info.artifact_uri
        source = f"{artifact_uri}/{model_path}"
        
        logging.info(f"Source Artifact URI determined: {source}")

        # 2. Ensure the Registered Model container exists
        try:
            client.create_registered_model(model_name)
            logging.info(f"Created new registered model container: {model_name}")
        except Exception:
            logging.info(f"Registered model container '{model_name}' already exists.")

        # 3. Create the Version (Low-level method to bypass connectivity checks)
        logging.info("Sending 'Create Version' request to DagsHub...")
        version = client.create_model_version(
            name=model_name,
            source=source,
            run_id=run_id
        )
        
        logging.info(f"Model Version {version.version} created with status: {version.status}")
        
        # 4. Transition the model to "Staging" stage
        logging.info("Transitioning to Staging...")
        client.transition_model_version_stage(
            name=model_name,
            version=version.version,
            stage="Staging"
        )
        
        logging.debug(f'Model {model_name} version {version.version} registered and transitioned to Staging.')
        print(f"✅ Success! Model {model_name} version {version.version} is now in Staging.")
        
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()