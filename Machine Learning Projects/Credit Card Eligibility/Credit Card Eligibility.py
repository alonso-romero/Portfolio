#%%
# Credit Card Eligibility Data Analysis

# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the dataset
df = pd.read_csv('dataset.csv')

# Display the first 5 rows of the dataset
print(df.head())

# Display information about the dataset
print(df.info())
#%%

"""
Features:
- ID: An identifier for each individual (customer)
- Gender: The gender of the individual {0 for male, 1 for female}
- Own_car: A binary feature indicating whether the individual owns a car {0 for no, 1 for yes}
- Own_property: A binary feature indicating whether the individual owns property {0 for no, 1 for yes}
- Work_phone: A binary feature indicating whether the individual has a work phone {0 for no, 1 for yes}
- Phone: A binary feature indicating whether the individual has a phone {0 for no, 1 for yes}
- Email: A binary feature indicating whether the individual has provided an email address {0 for no, 1 for yes}
- Unemployed: A binary feature indicating whether the individual is unemployed {0 for no, 1 for yes}
- `Num_children`: The number of children the individual has
- `Num_family`: The total number of family members
- `Account_length`: The length of the individual's account with a bank or financial institution
- `Total_income`: The total income of the individual
- `Age`: The age of the individual
- `Years_employed`: The number of years the individual has been employed
- `Income_type`: The type of income {object}
- `Education_type`: The education level of the individual {object}
- `Family_status`: The family status of the individual {object}
- `Housing_type`: The type of housing the individual lives in {object}
- `Occupation_type`: The type of occupation the individual is engaged in {object}
- `Target`: The target variable for the classification task, whether the individual is eligible for a credit card or not {0 for Not eligible, 1 for Eligible}
"""

# --------------- Data Analysis Question ---------------
# What factors contribute to credit card eligibility?


# ------------- Cleaning and Preprocessing -------------

# Check for missing values
print("\nNumber of Missing Values:", df.isnull().sum().sum()) 
# The dataset contains no null values

# Check for duplicate rows
print("\nNumber of Duplicate Rows:", df.duplicated().sum()) 
# The dataset contains no duplicate rows

# Display unique values of each feature for encoding
categorical_features = ['Income_type', 'Education_type', 'Family_status', 'Housing_type', 'Occupation_type']

for feature in categorical_features:
    unique_values = df[feature].unique()
    print(f"\nUnique values for {feature}: {unique_values}")

# initialize the LabelEncoder
label_encoder = LabelEncoder()

# Encoding the categorical features
for feature in categorical_features:
    df[feature] = label_encoder.fit_transform(df[feature])

print("\nFirst 5 rows of dataset with categorical features encoded:")
print(df.head())
#%%

#%%
# -------------- Exploratory Data Analysis -------------

#%%