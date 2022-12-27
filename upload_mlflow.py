import mlflow
import mlflow.sklearn
import sklearn.datasets
import sklearn.ensemble
import sklearn.model_selection

# Load the diabetes dataset
X, y = sklearn.datasets.load_diabetes(return_X_y=True)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)

# Train a random forest model
model = sklearn.ensemble.RandomForestRegressor()
model.fit(X_train, y_train)

# Log the model and its metadata to MLFlow
with mlflow.start_run():
    mlflow.sklearn.log_model(model, "model")

# Set the tracking server
mlflow.set_tracking_uri("http://10.0.0.1:5000")

# Deploy the model as a REST API
mlflow.sklearn.deploy(model_uri="model", name="my-model")
