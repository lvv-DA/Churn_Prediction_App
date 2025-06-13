import sqlite3
import os
import pandas as pd

# Get the directory where the script is located
script_dir = os.path.dirname(__file__)

# Construct the path to the database file
db_path = os.path.join(script_dir,"..","data",'churn_data.db')

try:
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print(f"Successfully connected to database: {db_path}")

    # --- Check available tables ---
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("\nTables in the database:")
    for table in tables:
        print(f"- {table[0]}")

    # --- Read data from a table (e.g., customer_identifiers) ---
    print("\nFirst 5 rows from 'customer_identifiers' table:")
    df_identifiers = pd.read_sql_query("SELECT * FROM customer_identifiers LIMIT 5;", conn)
    print(df_identifiers.to_string()) # .to_string() for better console formatting

    # --- Read data from another table (e.g., customer_churn) ---
    print("\nFirst 5 rows from 'customer_churn' table:")
    df_churn = pd.read_sql_query("SELECT * FROM customer_churn LIMIT 5;", conn)
    print(df_churn.to_string())

    # --- Optional: Check row count ---
    cursor.execute("SELECT COUNT(*) FROM customer_identifiers;")
    count_identifiers = cursor.fetchone()[0]
    print(f"\nTotal rows in 'customer_identifiers': {count_identifiers}")

    cursor.execute("SELECT COUNT(*) FROM customer_churn;")
    count_churn = cursor.fetchone()[0]
    print(f"Total rows in 'customer_churn': {count_churn}")

except sqlite3.Error as e:
    print(f"Database error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    # Close the connection
    if conn:
        conn.close()
        print("\nDatabase connection closed.")