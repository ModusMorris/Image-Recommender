import pickle
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Load the data from the pkl file
with open("classifications_with_names.pkl", "rb") as file:
    data = pickle.load(file)

# Extract class names from the dictionary
classifications = [value["class_name"] for value in data.values()]

# Count occurrences of each class
class_counts = pd.Series(classifications).value_counts()

# Limit to top 50 classes for better visualization
top_n = 1000
filtered_class_counts = class_counts.head(top_n)

# Prepare data for the 3D bar plot
x = np.arange(len(filtered_class_counts))  # Class names as categorical indices
y = filtered_class_counts.values  # Class counts
z = np.zeros_like(x)  # Base of the bars (all zeros)

# Plot the data as a 3D bar chart
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')

# Create 3D bars
ax.bar3d(x, z, z, dx=1, dy=1, dz=y, color=plt.cm.viridis_r(y / max(y)))

# Set the labels
ax.set_xlabel('Class Index', fontsize=14)
ax.set_ylabel('Z-axis', fontsize=14)
ax.set_zlabel('Number of Images', fontsize=14)

# Add title
ax.set_title(f"Top {top_n} Image Classification Counts in 3D", fontsize=16)

# Set tick labels on the x-axis as class names, but only show every 100th label
ax.set_xticks(x[::50])
ax.set_xticklabels(filtered_class_counts.index[::50], rotation=90, ha='center', fontsize=10)

plt.show()
