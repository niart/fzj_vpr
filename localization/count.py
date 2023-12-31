import pandas as pd

# Read the CSV file
my_filtered_csv = pd.read_csv("5 Seq similarity_results.csv", usecols=['Distance'])

# Calculate the percentage of distances less than 300
distances_less_than_300 = my_filtered_csv[my_filtered_csv['Distance'] < 300]
percentage_less_than_300 = (len(distances_less_than_300) / len(my_filtered_csv)) * 100

print(f"The percentage of distances less than 300 is: {percentage_less_than_300:.2f}%")
