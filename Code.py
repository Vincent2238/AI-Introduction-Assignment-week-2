# Clustering Student Learning Patterns (SDG 4: Quality Education)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load data (sample data)
data = {
    'student_id': ['s1', 's2', 's3', 's4', 's5'],
    'final_result': ['Pass', 'Fail', 'Pass', 'Withdrawn', 'Pass'],
    'date_registration': [1, 2, 3, 4, 5],
    'num_of_prev_attempts': [0, 1, 0, 2, 1],
    'studied_credits': [60, 30, 60, 90, 30],
    'disability': ['N', 'N', 'Y', 'N', 'Y'],
    'score': [85, 45, 78, 33, 90]
}
df = pd.DataFrame(data)

# Step 2: Select features for clustering
features = df[['score', 'num_of_prev_attempts', 'studied_credits']]

# Step 3: Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 4: Elbow Method to determine optimal k
inertia = []
k_range = range(1, 6)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 5: Apply KMeans with k=3
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_features)
df['cluster'] = clusters

# Step 6: Visualize Clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(x='score', y='studied_credits', hue='cluster', palette='Set1', data=df, s=100)
plt.title('Student Clusters Based on Score and Studied Credits')
plt.xlabel('Score')
plt.ylabel('Studied Credits')
plt.legend(title='Cluster')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7: Display results
print(df[['student_id', 'score', 'studied_credits', 'num_of_prev_attempts', 'cluster']])
