# Explore the data as follows: 

import pandas as pd

# Downloading the file and loading it into a Pandas dataframe
url = 'https://e.centennialcollege.ca/content/enforced/1010634-COMP309401_2023F/Misissagua_dealer.txt?_&d2lSessionVal=AJOR2OlmrARDauUz5Y2Dwd7cB'
df2_fatimah = pd.read_csv(url, delimiter='\t')

# Explore the data
# 1. Print the names of columns
print("Column Names:")
print(df2_fatimah.columns)

# 2. Print the types of columns
print("\nColumn Types:")
print(df2_fatimah.dtypes)

# 3. Print statistics count, min, mean, std, 1st quartile, median, 3rd quartile, max of numeric columns
print("\nNumeric Column Statistics:")
print(df2_fatimah.describe())

# 4. Print the first three records
print("\nFirst Three Records:")
print(df2_fatimah.head(3))

# 5. Print a summary of all missing values in all columns
print("\nSummary of Missing Values:")
print(df2_fatimah.isnull().sum())

# Preprocess the data:    

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Step 1: Convert categorical columns into numeric values using get_dummies and drop original columns
df2_fatimah_numeric = pd.get_dummies(df2_fatimah, drop_first=True)

# Step 2: Check for missing values
if df2_fatimah_numeric.isnull().sum().sum() == 0:
    print("\nNo Missing Values in the DataFrame.")

# Step 3: Save the new numeric dataset
df2_fatimah_numeric.to_csv('df2_fatimah_numeric.csv', index=False)

# Step 4: Standardize the numeric dataframe
scaler = StandardScaler()
df2_fatimah_standardized = pd.DataFrame(scaler.fit_transform(df2_fatimah_numeric), columns=df2_fatimah_numeric.columns)

# Step 5: Print statistics for the standardized dataframe
statistics_df = df2_fatimah_standardized.describe()
print("\nStatistics for Standardized DataFrame:")
print(statistics_df)

# Step 6: Save the standardized dataframe
df2_fatimah_standardized.to_csv('df2_fatimah_standardized.csv', index=False)

# Build a model    

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the standardized dataframe
df2_fatimah_standardized = pd.read_csv('df2_fatimah_standardized.csv')

# Perform K-means clustering with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)
df2_fatimah_standardized['cluster_fatimah'] = kmeans.fit_predict(df2_fatimah_standardized)

# Plot a histogram of the clusters
plt.figure(figsize=(8, 6))
plt.hist(df2_fatimah_standardized['cluster_fatimah'], bins=range(6), align='left', edgecolor='black', alpha=0.7)
plt.title("fatimah_clusters")
plt.xlabel("Cluster Number")
plt.ylabel("Number of Observations")
plt.xticks(range(5))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Print the number of observations in each cluster
cluster_counts = df2_fatimah_standardized['cluster_fatimah'].value_counts()
print("\nNumber of Observations in Each Cluster:")
print(cluster_counts)