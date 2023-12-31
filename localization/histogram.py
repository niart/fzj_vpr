import pandas as pd
import matplotlib.pyplot as plt

my_filtered_csv = pd.read_csv("5 Seq similarity_results.csv", usecols=['Distance'])

plt.hist(my_filtered_csv['Distance'], bins=100, color='orange')
plt.title('Occurrence Distribution of Distances')
plt.xlabel("Distance between prediction and ground truth")
plt.ylabel("Number of occurrences among 1521 testing samples")
plt.savefig('orange Occurrence-Distance-in-Seq5.png')
plt.show()
