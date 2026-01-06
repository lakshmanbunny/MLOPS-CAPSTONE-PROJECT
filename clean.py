import mlflow
from mlflow.tracking import MlflowClient
import os

# --- CONFIG ---
os.environ["MLFLOW_TRACKING_USERNAME"] = "lakshmanbunny"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "e4c0074dc1f21886d2eba50b20ca55c28dfeb3a1"
DAGSHUB_MLFLOW_URI = "https://dagshub.com/lakshmanbunny/MLOPS-CAPSTONE-PROJECT.mlflow"

mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
client = MlflowClient()
model_name = "my_model"

print(f"Attempting to delete {model_name}...")

try:
    # 1. Fetch all versions of the model
    versions = client.search_model_versions(f"name='{model_name}'")
    
    # 2. Delete every version first
    for version in versions:
        print(f"Deleting Version {version.version}...")
        client.delete_model_version(name=model_name, version=version.version)
        
    # 3. Delete the registered model container
    client.delete_registered_model(name=model_name)
    print(f"✅ Successfully deleted registered model: {model_name}")

except Exception as e:
    print(f"⚠️ Error: {e}")
    print("If the error says 'Resource not found', the model is already deleted!")