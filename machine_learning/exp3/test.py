import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Example data
x_labels = ['A', 'B', 'C', 'D']
y_values = np.random.rand(len(x_labels))

# Use the "magma" color palette
colors = sns.color_palette("Set2", len(x_labels))

# Plotting
bars = plt.bar(x_labels, y_values, color=colors)

# Optionally, you can customize other plot settings
plt.xlabel('X Labels')
plt.ylabel('Y Values')
plt.title('Bar Plot with "magma" Color Scheme')

# Show the plot
plt.show()
