from sklearn.metrics import accuracy_score, classification_report


def evaluate(y_test, y_pred):

    accuracy = accuracy_score(y_test, y_pred)

    print("Model Accuracy:", accuracy)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))