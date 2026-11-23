import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data():
    
    # Load dataset
    dataset_path = (
        Path(os.environ["IRIS_DATA_PATH"])
        if "IRIS_DATA_PATH" in os.environ
        else Path(__file__).resolve().parent / "data" / "Irisdataset.csv"
    )
    data = pd.read_csv(dataset_path)
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

