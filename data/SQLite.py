import pandas as pd
import sqlite3
import os

# Get the directory where the script is located
# This will be 'd:/Vinu UK/Churn_app_01/data'
script_dir = os.path.dirname(__file__)

# Construct the path to the CSV files assuming they are IN THE SAME DIRECTORY as the script
customer_identifiers_path = os.path.join(script_dir,'customer_data_with_identifiers.csv')
customer_churn_path = os.path.join(script_dir, 'customer_churn.csv')

# Construct the path for the SQLite database file in the same directory
db_path = os.path.join(script_dir, 'churn_data.db')

# Load your CSVs using the constructed paths
try:
    df_identifiers = pd.read_csv(customer_identifiers_path)
    df_churn = pd.read_csv(customer_churn_path)

    # Create a SQLite database connection
    conn = sqlite3.connect(db_path)

    # Write DataFrames to SQLite tables
    df_identifiers.to_sql('customer_identifiers', conn, if_exists='replace', index=False)
    df_churn.to_sql('customer_churn', conn, if_exists='replace', index=False)

    # Close the connection
    conn.close()

    print("Data successfully loaded into churn_data.db in the 'data' directory!")

except FileNotFoundError:
    print(f"Error: One or both CSV files were not found.")
    print(f"Expected customer_data_with_identifiers.csv at: {customer_identifiers_path}")
    print(f"Expected customer_churn.csv at: {customer_churn_path}")
    print("Please ensure your CSV files are in the same directory as the SQLite.py script.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")