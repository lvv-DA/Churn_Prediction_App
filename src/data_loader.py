import pandas as pd
import os

def load_data(data_path):
    """
    Loads the dataset from the specified path.
    Args:
        data_path (str): The path to the CSV data file.
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    try:
        # Try reading with a common alternative encoding like 'latin1'
        df = pd.read_csv(data_path, encoding='latin1')
        print("Data loaded successfully.")
        return df
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None
    except UnicodeDecodeError:
        print(f"Error: Could not decode data from {data_path} with 'latin1' encoding. Trying 'ISO-8859-1'...")
        try:
            df = pd.read_csv(data_path, encoding='ISO-8859-1')
            print("Data loaded successfully with 'ISO-8859-1' encoding.")
            return df
        except UnicodeDecodeError:
            print(f"Error: Could not decode data from {data_path} with 'ISO-8859-1' encoding. Trying 'cp1252'...")
            try:
                df = pd.read_csv(data_path, encoding='cp1252')
                print("Data loaded successfully with 'cp1252' encoding.")
                return df
            except Exception as e:
                print(f"Final attempt failed. An error occurred while loading data with multiple encodings: {e}")
                return None
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        return None

if __name__ == '__main__':
    # Example usage (assuming DATA_PATH is defined somewhere, e.g., in config.py)
    # For this example, let's create a dummy file or adjust the path
    # You would typically have a config.py to define DATA_PATH
    # For now, let's assume it's in the parent directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # This dummy path should now reflect the CSV file
    dummy_data_path = os.path.join(project_root, 'data', 'customer_churn.csv')

    # Create a dummy CSV file for testing if it doesn't exist
    if not os.path.exists(os.path.join(project_root, 'data')):
        os.makedirs(os.path.join(project_root, 'data'))
    if not os.path.exists(dummy_data_path):
        dummy_df = pd.DataFrame({
            'CallFailure': [8,0,10], 'Complains': [0,0,0], 'SubscriptionLength': [38,39,37],
            'ChargeAmount': [0,0,0], 'SecondsUse': [4370,318,2453], 'FrequencyUse': [71,5,60],
            'FrequencySMS': [5,7,359], 'DistinctCalls': [17,6,20], 'AgeGroup': [2,3,2],
            'TariffPlan': [1,2,1], 'Status': [1,1,1], 'Age': [30,45,28],
            'CustomerValue': [150.0, 200.0, 180.0], 'Churn': [0, 1, 0]
        })
        # Save with latin1 for testing this specific fix
        dummy_df.to_csv(dummy_data_path, index=False, encoding='latin1')
        print(f"Created dummy CSV file at {dummy_data_path}")

    df_test = load_data(dummy_data_path)
    if df_test is not None:
        print(f"Loaded DataFrame shape: {df_test.shape}")
        print(f"Loaded DataFrame columns: {df_test.columns.tolist()}")