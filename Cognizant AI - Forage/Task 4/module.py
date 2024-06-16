# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Function to load the dataset
def load_data(file_path):
    """
    Load the CSV data into a DataFrame.

    :param file_path: str, path to the CSV file
    :return: DataFrame
    """
    data = pd.read_csv(file_path)
    return data

# Function to split the dataset into training and testing sets
def split_data(data, test_size=0.2):
    """
    Split the dataset into training and testing sets.

    :param data: DataFrame, the dataset to split
    :param test_size: float, proportion of the dataset to include in the test split
    :return: tuple, (X_train, X_test, y_train, y_test)
    """
    X = data.drop('estimated_stock_pct', axis=1)
    y = data['estimated_stock_pct']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test

# Function to train the model
def train_model(X_train, y_train):
    """
    Train a RandomForest model on the training data.

    :param X_train: DataFrame, training features
    :param y_train: Series, training labels
    :return: trained model
    """
    model = RandomForestRegressor(random_state=42)

    model.fit(X_train, y_train)

    return model

# Function to evaluate the model
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test data.

    :param model: trained model
    :param X_test: DataFrame, test features
    :param y_test: Series, test labels
    :return: float, accuracy score
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return mae

# Function to save the trained model
def save_model(model, file_path):
    """
    Save the trained model to disk.

    :param model: trained model
    :param file_path: str, path to save the model
    """
    joblib.dump(model, file_path)

# Main function to execute the workflow
def main(file_path, model_save_path):
    """
    Execute the machine learning workflow: load data, split, train, evaluate, and save model.

    :param file_path: str, path to the CSV file
    :param model_save_path: str, path to save the trained model
    """
    # Load the data
    data = load_data(file_path)

    K = 10
    accuracy = []
    model = None

    for i in range(0, K):

      # Split the data
      X_train, X_test, y_train, y_test = split_data(data)
      scaler = StandardScaler()

      scaler.fit(X_train)
      X_train = scaler.transform(X_train)
      X_test = scaler.transform(X_test)

      # Train the model
      model = train_model(X_train, y_train)

      # Evaluate the model
      mae = evaluate_model(model, X_test, y_test)
      accuracy.append(mae)
      print(f"Model MAE for fold {i+1}: {mae:.3f}%")

    # Print average MAE across all folds
    average_mae = sum(accuracy) / len(accuracy)
    print(f"Average MAE: {average_mae:.2f}%")

    # Save the model
    save_model(model, model_save_path)
    print(f"Model saved to {model_save_path}")

# Entry point for the script
if __name__ == "__main__":
    # Example usage
    csv_file_path = "{file_path}"
    model_output_path = "{folder_path}"
    main(csv_file_path, model_output_path)
