import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt

seq_len = 5
# Load reference and query dictionaries from pickle files
with open('seq_reference1609.pkl', 'rb') as file:
    reference = pickle.load(file)

with open('seq_query1521.pkl', 'rb') as file:
    query = pickle.load(file)

#print(type(reference), type(query))
# Initialize lists to store results
results = []
distances = []
# Iterate through query latent codes

# Iterate through the dictionary to read the values
for key, value in query.items():
    if key < 1521 - seq_len + 1:
        #print(f"The value for key {key} is: {value}")
        query_codes = []
        query_coordinates = []
        for i in range(seq_len):
            query_codes.append(query[key + i][0])
            #print(type(query_codes))
            a = list(query[key + i][1])
            query_coordinates.append([float(a[0]), float(a[1])])    
        sequence_1 = np.array([query_codes[0],
                            query_codes[1], 
                            query_codes[2], 
                            query_codes[3],
                            query_codes[4]]) 
        
        max_similarity = 0 
        similar_place = None
        similar_codes = None
        distance = None

        for key_1, value_1 in reference.items():
            if key_1 < 1609 - seq_len + 1:
                reference_codes = []
                reference_coordinates = []
                for j in range(seq_len):
                    reference_codes.append(reference[key_1 + j][0])
                    b = list(reference[key_1 + j][1])
                    reference_coordinates.append([float(b[0]), float(b[1])])
                sequence_2 = np.array([reference_codes[0],
                                    reference_codes[1], 
                                    reference_codes[2], 
                                    reference_codes[3],
                                    reference_codes[4]])    
                # Reshape the arrays if necessary (ensure the shape is (n_samples, n_features))
                #print('after', type(sequence_2))
                # Compute cosine similarity
                sequence_1 = sequence_1.reshape(-1)
                sequence_2 = sequence_2.reshape(-1)
                dot_product = np.dot(sequence_1, sequence_2)
                magnitude_1 = np.linalg.norm(sequence_1)
                magnitude_2 = np.linalg.norm(sequence_2)
                cosine_similarity = dot_product / (magnitude_1 * magnitude_2)

                if cosine_similarity > max_similarity:                   
                    max_similarity = cosine_similarity
                    similar_place = reference_coordinates[3]
                    similar_codes = reference_codes
        distance = np.sqrt((query_coordinates[3][0] - similar_place[0])**2 + (query_coordinates[3][1] - similar_place[1])**2)
        distances.append(distance)

        # Append the results
        results.append((max_similarity, distance, query_coordinates[3], similar_place))
        print(max_similarity, distance, query_coordinates[3], similar_place)

plt.figure(figsize=(10, 6))  # Adjust the figure size if needed
indices = list(range(1, 1518))
plt.plot(indices, distances, marker='.', linestyle='-')
# Labeling the axes and title
plt.xlabel('Index')
plt.ylabel('Distance')
plt.title('Seq Plot of Distances')
plt.savefig("5 Seq Plot of Distances")

# Save results to a CSV file
with open('5 Seq similarity_results.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([ 'Highest Cosine Similarity', 'Distance', 'Query Coordinate', 'Reference Coordinate'])
    writer.writerows(results)
