import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Import your validation functions from preprocessor.py
from preprocessing import validate_path, seperate_target

def main():
    # 1. Configuration & Paths
    data_path = "Datasets\Student_Addiction.csv"   # Update to your dataset path
    target_column = "addicted_label"            # Update to your target column name
    model_path = "artifacts\model.pkl"
    
    print("==================================")
    print("Starting Model Evaluation Pipeline...")
    print("==================================")
    
    # 2. Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model not found at {model_path}. Please run train.py first!")
    
    # 3. Load Dataset & Test Split (Must match the split used in train.py)
    validate_path(data_path)
    df = pd.read_csv(data_path)
    X, y = seperate_target(df, target=target_column)
    
    # Use the exact same random_state and stratify to recreate the same test set
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 4. Load the Saved Master Pipeline
    print(f"Loading trained model from {model_path}...")
    model = joblib.load(model_path)
    
    # 5. Make Predictions on Test Data
    print("Running predictions on the test set...")
    y_pred = model.predict(X_test)
    
    # 6. Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print("\n==================================")
    print(f"Evaluation Results:")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print("==================================")
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()