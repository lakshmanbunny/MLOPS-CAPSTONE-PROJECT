import unittest
import mlflow
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = "lakshmanbunny" # User's username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "lakshmanbunny"  # <-- I updated this for you
        repo_name = "MLOPS-CAPSTONE-PROJECT" # <-- I updated this for you

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the new model from MLflow model registry
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        
        if cls.new_model_version is None:
            raise ValueError(f"No model version found for '{cls.new_model_name}' in stage 'Staging'")

        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        print(f"Loading model from: {cls.new_model_uri}")
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load the vectorizer
        # Ensure 'models/vectorizer.pkl' exists via dvc pull or dvc repro
        cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))

        # Load holdout test data
        cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        # Get latest versions for the given stage
        latest_versions = client.get_latest_versions(model_name, stages=[stage])
        return latest_versions[0].version if latest_versions else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input for the model based on expected input shape
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

        # Predict using the new model to verify the input and output shapes
        prediction = self.new_model.predict(input_df)

        # Verify the input shape
        self.assertEqual(input_df.shape[1], len(self.vectorizer.get_feature_names_out()))

        # Verify the output shape (assuming binary classification with a single output)
        self.assertEqual(len(prediction), input_df.shape[0])
        # Check if output is a numpy array or list
        self.assertTrue(hasattr(prediction, '__len__')) 

    def test_model_performance(self):
        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.iloc[:,0:-1]
        y_holdout = self.holdout_data.iloc[:,-1]

        # Predict using the new model
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Define expected thresholds for the performance metrics
        # (Lowered slightly to 0.40 as per your previous request, adjust as needed)
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy_new:.4f}")
        print(f"Precision: {precision_new:.4f}")
        print(f"Recall: {recall_new:.4f}")
        print(f"F1 Score: {f1_new:.4f}")

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy {accuracy_new} < {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision {precision_new} < {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall {recall_new} < {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score {f1_new} < {expected_f1}')

if __name__ == "__main__":
    unittest.main()