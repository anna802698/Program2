from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocess1 import preprocess_data

def train_model():

    X_train, X_test, y_train, y_test = preprocess_data()

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Prediction
    predictions = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("Model Training Completed")
    print("Model Accuracy:", accuracy)

if __name__ == "__main__":
    train_model()
