#%%
# Credit Card Eligibility Data Analysis

# import libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Load the dataset
df = pd.read_csv('dataset.csv')

# Display the first 5 rows of the dataset
print(df.head())

# Display information about the dataset
print(df.info())
#%%