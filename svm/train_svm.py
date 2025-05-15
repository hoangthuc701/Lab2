# train_svm.py
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def train_svm(train_path="svm/features_train.npz", test_path="svm/features_test.npz"):
    # Load features
    train_data = np.load(train_path)
    test_data = np.load(test_path)

    X_train, y_train = train_data['X'], train_data['y']
    X_test, y_test = test_data['X'], test_data['y']

    print("Training SVM...")
    clf = SVC(kernel='linear', C=1.0)
    clf.fit(X_train, y_train)

    print("Evaluating on test set...")
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    train_svm()
