import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pycolormap_2d import ColorMap2DBremm  # Import the ColorMap2DZiegler colormap
from pycolormap_2d import ColorMap2DZiegler  # Import the ColorMap2DZiegler colormap
import matplotlib.colors as mcolors


# Load the data from "turtle.csv"
data = pd.read_csv("all_2nd_turtle.csv")

# Extract X and Y columns
X = data["X"]
Y = data["Y"]

# Find the minimum and maximum values of X and Y
min_x, max_x = X.min(), X.max()
min_y, max_y = Y.min(), Y.max()

# Determine the size of each grid cell
x_range = max_x - min_x
y_range = max_y - min_y
grid_size_x = x_range / 4
grid_size_y = y_range / 4

# Create labels for each grid cell (A, B, C, ..., P)
labels = [chr(ord('A') + i) for i in range(16)]

# Initialize an empty list to store labels for each point
point_labels = []

# Load the new data from "turtle2.csv"
data2 = pd.read_csv("4_2nd_turtle.csv")

# Extract X and Y columns for the new trip
X2 = data2["X"]
Y2 = data2["Y"]

# Create a 4x4 grid and assign labels to each point in the new trip
for i in range(len(X2)):
    x = X2[i]
    y = Y2[i]
    x_index = min(int((x - min_x) / grid_size_x), 3)
    y_index = min(int((y - min_y) / grid_size_y), 3)
    grid_label = labels[y_index * 4 + x_index]
    point_labels.append(grid_label)

# Add labels to the new data
data2["Label"] = point_labels

# Save the labeled data from the new trip to "label2.csv"
data2.to_csv("4_label.csv", index=False)

# Define the landmarks
landmarks = {
    "Center": {"TX": -372.943634, "TY": -1109.714722, "color": "red"},
    "Wall-Door": {"TX": -364.122894, "TY": -3034.825684, "color": "blue"},
    "Wall-Kitech": {"TX": 869.19812, "TY": -1165.539063, "color": "green"},
    "Wall-superT": {"TX": -1612.344604, "TY": -1163.421631, "color": "orange"},
    "Wall-Window": {"TX": -378.007507, "TY": 1306.378662, "color": "purple"},
}

# Create a plot of the new trip's trajectory
# plt.figure(figsize=(8, 6))
# plt.plot(X2, Y2, label="X-Y Curve (New Trip)")

# Create a colormap plot of the X-Y curve
plt.figure(figsize=(6, 8))
cmap = plt.get_cmap("cool")  # You can change the colormap as desired
colors = np.linspace(0, 1, len(X2))
plt.scatter(X2, Y2, c=colors, cmap=cmap, marker='.', label="Robot Path")

# Add colorbar
cbar = plt.colorbar()
cbar.set_label("Start-End")

# Create a dictionary to store the colors used for each cell
cell_colors = {}

# Plot the grid lines using the previous grid
for i in range(1, 4):
    plt.axvline(x=min_x + i * grid_size_x, color='gray', linestyle='--')
    plt.axhline(y=min_y + i * grid_size_y, color='gray', linestyle='--')

#################################################################################
# # Add labels to the grid cells using the previous grid
# for i, label in enumerate(labels):
#     x_center = min_x + (i % 4 + 0.5) * grid_size_x
#     y_center = min_y + (i // 4 + 0.5) * grid_size_y

#     # Create a heatmap effect with colors from blue to red
#     color_value = i / 15  # Ranges from 0 to 1
#     color = plt.cm.coolwarm(color_value)

#     # Display the cell label
#     plt.text(x_center, y_center, label, fontsize=12, ha='center', va='center', color = 'black')

#     # Fill the cell with the heatmap color
#     plt.fill_between([x_center - grid_size_x / 2, x_center + grid_size_x / 2],
#                      [y_center - grid_size_y / 2, y_center - grid_size_y / 2],
#                      [y_center + grid_size_y / 2, y_center + grid_size_y / 2],
#                      color=color, alpha=0.5)

#     # Store the color in the dictionary
#     cell_colors[label] = color
#######################################################################################
# Add labels to the grid cells
    # Create a colormap using ColorMap2DZiegler
cmap = ColorMap2DZiegler(range_x=(min_x, max_x), range_y=(min_y, max_y))

for i, label in enumerate(labels):
    x_center = min_x + (i % 4 + 0.5) * grid_size_x
    y_center = min_y + (i // 4 + 0.5) * grid_size_y

    # Create a heatmap effect with colors from ColorMap2DBremm
    color = cmap(x_center, y_center)

    # Normalize the RGB values to the 0-1 range
    color = [c / 255.0 for c in color]

    # Display the cell label
    plt.text(x_center, y_center, label, fontsize=12, ha='center', va='center', color='black')

    # Fill the cell with the heatmap color
    plt.fill_between([x_center - grid_size_x / 2, x_center + grid_size_x / 2],
                     [y_center - grid_size_y / 2, y_center - grid_size_y / 2],
                     [y_center + grid_size_y / 2, y_center + grid_size_y / 2],
                     color=color, alpha=0.7)

    # Store the color in the dictionary
    cell_colors[label] = color
#####################################################################################33
# Print the color of each cell in the format "A - #7f7f7f"
for label, color in cell_colors.items():
    hex_color = "#{:02x}{:02x}{:02x}".format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    print(f"{label} - {hex_color}")

# Plot landmarks with different icons and colors
for landmark, coords in landmarks.items():
    plt.scatter(coords["TX"], coords["TY"], marker='s', label=landmark, s=100, c=coords["color"])

# Add labels to the plot
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()

# Show the plot
plt.show()
