import pandas as pd
import os

path = 'C:/Users/yasin/Desktop/COMP-309/Final Test'
file = 'AA3.csv'

fullpath = os.path.join(path, file)

df_fatimah = pd.read_csv(fullpath)

# Explore the data

# Print the names of columns
print(df_fatimah.columns.values)

# Print the types of columns
print(df_fatimah.dtypes)

# Print the unique values in each column.
for cols in df_fatimah.columns:
    unique = df_fatimah[cols].unique()
    print(unique)
    
# Print the statistics count, min, mean, standard deviation, 1st quartile, 
# median, 3rd quartile max of all the numeric columns(use one command).
print(df_fatimah.describe())

# Print the first three records.
print(df_fatimah.head(3))

# Print a summary of all missing values in all columns (use one command).
print(df_fatimah.isnull().sum())

# Print the total number (count) of each unique value in the following categorical columns: 
# Model
print(df_fatimah['model'].value_counts())

# Color
print(df_fatimah['color'].value_counts())

# Visualize the data 

# Plot a histogram for the millage use 12 bins, name the x and y axis’ 
# appropriately, give the plot a title "firstname_millage".
import matplotlib.pyplot as plt

plt.figure(figsize=(10,9))
plt.hist(df_fatimah['millage'], bins=12)
plt.title('fatimah_millage')
plt.xlabel('Millage')
plt.ylabel('Stolen')
plt.show()

# Create a scatterplot showing "millage" versus "value", name the x and y axis’
# appropriately, give the plot a title "firstname_millage_scatter".

plt.figure(figsize=(10,9))
plt.scatter(df_fatimah['millage'], df_fatimah['value'])
plt.title('fatimah_millage_scatter')
plt.xlabel('Millage')
plt.ylabel('Value')
plt.show()

# Plot a "scatter matrix" showing the relationship between all columns of the 
# dataset on the diagonal of the matrix plot the kernel density function.
import seaborn as sns

sns.pairplot(data=df_fatimah)
plt.suptitle('fatimah_Scatter_Matrix')
plt.show()


# Pre-process the data (15 marks)

# Remove (drop) properly the column with the most missing values. 
# (hint: make sure you review and set the right arguments)
print("Missing Values Before Pre-processing:")
print(df_fatimah.isnull().sum())

#the only column that has most missing value is 'motor'
df_fatimah = df_fatimah.drop(columns='motor')

# Replace the missing values in the "millage" column with the mean average of 
# the column value.  
df_fatimah['millage'].fillna(df_fatimah['millage'].mean(), inplace=True)

# Check that there are no missing values.
print("Missing Values After Pre-processing:")
print(df_fatimah.isnull().sum())


# Convert the all the categorical columns into numeric values and drop/delete 
# the original columns. (hint:  use get dummies) Make sure your new data frame
# is completely numeric, name it df_firstname_numeric.
# Specify categorical columns
# Specify categorical columns
# Specify categorical columns
cat_vars = ['model', 'type', 'damage', 'color']

# Iterate through each categorical column
for var in cat_vars:
    # Apply one-hot encoding with a custom suffix
    cat_list = pd.get_dummies(df_fatimah[var], prefix=var)
    
    # Drop the original categorical column
    df_fatimah = df_fatimah.drop(columns=[var])
    
    # Join the new columns to the original DataFrame
    df_fatimah = df_fatimah.join(cat_list)

# Extract column names after one-hot encoding
data1_vars = df_fatimah.columns.values.tolist()

# Create a list of columns to keep (excluding original categorical columns)
to_keep = [i for i in data1_vars if i not in cat_vars]

# Create the final DataFrame with only numeric columns
df_fatimah_numeric = df_fatimah[to_keep]

# Display the first few rows of the final numeric DataFrame
print(df_fatimah_numeric.head())


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Split the data into features (X) and target variable (y)
columns = df_fatimah.columns.values.tolist()

X = df_fatimah[columns] 
Y = df_fatimah['stolen']

# Split the data into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)

# Build the decision tree model
dt_fatimah = DecisionTreeClassifier(criterion='entropy', max_depth=6, min_samples_split=13, random_state=42)
dt_fatimah.fit(X_train, y_train)

# Validate using 6-fold cross-validation
cross_val_accuracy = cross_val_score(dt_fatimah, X_train, y_train, cv=6, scoring='accuracy')
print("Mean Accuracy on Validation Data:", cross_val_accuracy.mean())

# Test the model on the testing data
y_pred = dt_fatimah.predict(X_test)

# Print accuracy and confusion matrix
test_accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy on Test Data:", test_accuracy)
print("\nConfusion Matrix:", conf_matrix)

# Save confusion matrix plot
plt.savefig("fatimah_screenshotAA3.png")

# Prune the tree: Vary the maximum depth of your predictive model from 1 to 8
# and print the mean accuracy of the k-flod of each run on the training data.

depths = list(range(1, 9))
mean_accuracies = []

for depth in depths:
    dt_pruned = DecisionTreeClassifier(criterion='entropy', max_depth=depth, min_samples_split=13, random_state=42)
    accuracy = cross_val_score(dt_pruned, X_train, y_train, cv=6, scoring='accuracy').mean()
    mean_accuracies.append(accuracy)

# Print mean accuracies for different depths
for depth, accuracy in zip(depths, mean_accuracies):
    print(f"Depth: {depth}, Mean Accuracy: {accuracy}")