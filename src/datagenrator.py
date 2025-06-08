import pandas as pd
import numpy as np
import random
import io # Import io to capture string output

# Path to your original customer churn data (uploaded file is directly accessible by its name)
original_data_path = "D:\Vinu UK\DA_Projects\DA_Projects\churn_prediction_app\CFM KTP_Stage 1 task_churn dataset.xlsx"

try:
    df_churn = pd.read_excel(original_data_path)
    # print(f"Original data loaded successfully from: {original_data_path}") # Removed print for clean CSV output
except FileNotFoundError:
    print(f"Error: Original data file not found at {original_data_path}")
    print("Please ensure 'df.xlsx - Customer Churn.csv' is correctly provided.")
    exit()

# --- Generate Synthetic Identifying Data ---

num_customers = len(df_churn)

# CustomerID
df_churn['CustomerID'] = range(100001, 100001 + num_customers)

# CustomerName (simple synthetic names)
first_names = ["Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy", "Karl", "Linda", "Mike", "Nancy", "Oscar", "Pamela", "Quinn", "Rachel", "Steve", "Tina"]
last_names = ["Smith", "Jones", "Williams", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez", "Robinson", "Clark"]

synthetic_names = []
for _ in range(num_customers):
    first = random.choice(first_names)
    last = random.choice(last_names)
    synthetic_names.append(f"{first} {last}")
df_churn['CustomerName'] = synthetic_names

# PhoneNumber (random 10-digit numbers, realistic looking UK mobile numbers)
synthetic_phone_numbers = []
for _ in range(num_customers):
    phone_prefix = random.choice(["077", "078", "079"]) # Common UK mobile prefixes
    remaining_digits = ''.join([str(random.randint(0, 9)) for _ in range(7)])
    synthetic_phone_numbers.append(f"{phone_prefix}{remaining_digits}")
df_churn['PhoneNumber'] = synthetic_phone_numbers

# Address (simple synthetic addresses)
street_names = ["Main St", "Oak Ave", "Pine Ln", "Elm Dr", "Maple Rd", "High St", "Church Rd"]
cities = ["London", "Manchester", "Birmingham", "Leeds", "Glasgow", "Bristol", "Liverpool"]
postcodes = ["SW1A 0AA", "M1 1AA", "B1 1BB", "LS1 1BA", "G1 1AE", "BS1 1AA", "L1 1AP"] # Simplified UK postcodes

synthetic_addresses = []
for i in range(num_customers):
    house_num = random.randint(1, 150)
    street = random.choice(street_names)
    city = random.choice(cities)
    postcode = random.choice(postcodes)
    synthetic_addresses.append(f"{house_num} {street}, {city}, {postcode}")
df_churn['Address'] = synthetic_addresses

# Reorder columns to have identifiers at the front
identifier_cols = ['CustomerID', 'CustomerName', 'PhoneNumber', 'Address']
# Ensure all other original columns are included
other_cols = [col for col in df_churn.columns if col not in identifier_cols]
df_churn = df_churn[identifier_cols + other_cols]

# Print the CSV content directly to stdout
# Use io.StringIO to capture the CSV output to a string
output = io.StringIO()
df_churn.to_csv(output, index=False)
print(output.getvalue())