import pandas as pd

# Load the motion capture data from the original file with proper header handling
data = pd.read_csv("4_2nd.csv", skiprows=4)

# Extract only columns 36 and 37, starting from row six
# 12, 13
selected_columns = data.columns[11:13]
turtlebot_data = data.iloc[:, 11:13]

# Rename the columns for better readability (optional)
turtlebot_data.columns = ["X", "Y"]

# Save the extracted data to a new CSV file named "turtle.csv"
turtlebot_data.to_csv("4_2nd_turtle.csv", index=False)
