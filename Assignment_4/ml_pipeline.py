from prefect import flow, task
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

@task
def load_data():
    
    iris = load_iris()
    return iris.data, iris.target

@task
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

@task
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

@task
def save_model(model, filename="iris_model.pkl"):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

@flow
def iris_pipeline():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    save_model(model)

if __name__ == "__main__":
    iris_pipeline()
