from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Standard 80/20 train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Feed forward NN w/ 40 neurons and one layer
clf = MLPClassifier(hidden_layer_sizes=(40,), random_state=41)

# Train the neural network
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate performance using accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)
