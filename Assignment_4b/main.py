#Predict Whether a Person Buys a Product or Not
from prefect import task, flow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

#load data
@task
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data     

#normalize data
@task
def preprocess_data(data):
    data = data.dropna()
    X = data.drop('Purchased', axis=1)
    y = data['Purchased'].map({'Yes': 1, 'No': 0})

    X = pd.get_dummies(X, drop_first=True)
    return X, y 

#split data
@task
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)

#train model
@task
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

#evaluate model (accuracy)
@task
def evalute_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    return accuracy

#save model
@task   
def save_model(model, filename="logistic_model.pkl"):
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")


@flow
def ml_pipeline(file_path):
    data = load_data(file_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    accuracy = evalute_model(model, X_test, y_test)
    save_model(model)
    return accuracy

if __name__ == "__main__":
    file_path = "ecom_data.csv"  
    ml_pipeline(file_path)


