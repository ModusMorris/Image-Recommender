import pickle
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx

# Load the data from the pkl file
with open('pkl_files/classifications_with_names.pkl', 'rb') as file:
    data = pickle.load(file)

# Extract class names from the dictionary
classifications = [value['class_name'] for value in data.values()]

# Count occurrences of each class
class_counts = pd.Series(classifications).value_counts()

# Limit to top 30 classes for better visualization
top_n = 50
filtered_class_counts = class_counts.head(top_n)

# Plot the data as a horizontal bar chart
plt.figure(figsize=(12, 8))
filtered_class_counts.plot(kind='barh', color=plt.cm.viridis_r(filtered_class_counts / max(filtered_class_counts)))

# Add titles and labels
plt.title(f'Top {top_n} Image Classification Counts', fontsize=16)
plt.xlabel('Number of Images', fontsize=14)
plt.ylabel('Class Name', fontsize=14)

# Add counts on the bars
for index, value in enumerate(filtered_class_counts):
    plt.text(value, index, str(value), va='center', ha='left', fontsize=12, color='black')

# Improve layout
plt.tight_layout()

# Show the plot
plt.show()