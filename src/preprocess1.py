import pandas as pd
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    
    # Load dataset
    data = pd.read_csv("C:/Users/Dell/Desktop/MLOps Lab programs/Program2/src/data/Irisdataset.csv")
    # Create DataFrame
    df = pd.DataFrame(data)

    # Display the DataFrame
    print(df.head())
    # Separate features and target
    X = df.drop("Species", axis=1)
    y = df["Species"]
    
   

    print(X.head())

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("split has done")
    # # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_data()
    print("Data preprocessing completed successfully")

