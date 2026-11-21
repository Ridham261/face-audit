import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="sqlite:///D:/audit/mlflow.db")
mv = client.get_model_version_by_alias("GogglesDetector", "production")
print(f"Production model: GogglesDetector v{mv.version}")
