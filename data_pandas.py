import pickle
import pandas as pd

# Load the data from the .pkl file
with open("classifications_with_names.pkl", "rb") as file:
    data = pickle.load(file)

# Convert the data into a DataFrame
df = pd.DataFrame(data).T

# If the DataFrame contains nested dictionaries, normalize it
if isinstance(df.iloc[0, 0], dict):
    df = pd.json_normalize(df.to_dict())

# Ensure all columns are accessible
df.columns = df.columns.map(lambda x: x.split('.')[-1])

# Count the occurrences of each classification
class_counts = df['class_name'].value_counts()

# Convert the Series to a DataFrame for better readability
class_counts_df = class_counts.reset_index()
class_counts_df.columns = ['Class Name', 'Count']

# Extract all unique classifications and convert to a list
unique_classifications = class_counts_df['Class Name'].unique().tolist()

# Display the unique classifications
#print("Unique Classifications:")
#print(unique_classifications)

# Save the table to a CSV file for further use
class_counts_df.to_csv("classification_counts.csv", index=False)