import numpy as np
from sklearn.linear_model import Perceptron

def run_perceptron(X_train, y_train, X_test):
    clf = Perceptron(max_iter=1000, tol=1e-3, random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred

# some example training data
X_train = np.array([
    [2, 1],
    [1, 1],
    [2, 0],
    [0, 1]
])


y_train = np.array([1, 1, -1, -1])
# example test data
X_test = np.array([
    [1.5, 1],
    [0.2, 0.8]
])

model, predictions = run_perceptron(X_train, y_train, X_test)

print("Predictions:", predictions)
