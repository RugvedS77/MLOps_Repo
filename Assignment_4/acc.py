import joblib
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# Load model
model = joblib.load("iris_model.pkl")

# Test it
iris = load_iris()
X, y = iris.data, iris.target
y_pred = model.predict(X)
print("Full dataset accuracy:", accuracy_score(y, y_pred))
