import random
import pandas as pd
import numpy as np
from faker import Faker
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import hdbscan
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull

# Setup
TOTAL_POINTS = 60000
DELHI_BOUNDS = {'min_lat': 28.55, 'max_lat': 28.75, 'min_lon': 77.00, 'max_lon': 77.30}
fake = Faker()

# Step 1: Set type distribution
res_ratio = random.uniform(0.65, 0.75)
vac_ratio = random.uniform(0.03, 0.05)
com_ratio = 1 - (res_ratio + vac_ratio)

type_distribution = {
    'Residential': int(res_ratio * TOTAL_POINTS),
    'Commercial': int(com_ratio * TOTAL_POINTS),
    'Vacant': TOTAL_POINTS - int(res_ratio * TOTAL_POINTS) - int(com_ratio * TOTAL_POINTS)
}

cluster_config = {
    'Residential': 5,
    'Commercial': 3,
    'Vacant': 2
}

# Step 2: Generate data
property_data = []
property_id_counter = 100000

for prop_type, total_count in type_distribution.items():
    clusters = cluster_config[prop_type]
    
    base = total_count // clusters
    points_per_cluster = [base] * clusters
    for i in random.sample(range(clusters), total_count % clusters):
        points_per_cluster[i] += 1

    for cluster_id in range(clusters):
        n_points = points_per_cluster[cluster_id]
        if n_points <= 0:
            continue

        center_lat = random.uniform(DELHI_BOUNDS['min_lat'], DELHI_BOUNDS['max_lat'])
        center_lon = random.uniform(DELHI_BOUNDS['min_lon'], DELHI_BOUNDS['max_lon'])
        std_dev = random.uniform(0.02, 0.06)
        cov = [[std_dev ** 2, 0], [0, std_dev ** 2]]

        points = np.random.multivariate_normal((center_lat, center_lon), cov, n_points)

        for lat, lon in points:
            property_data.append({
                'Property_ID': f'DL-{property_id_counter}',
                'Property_Type': prop_type,
                'Latitude': round(lat, 6),
                'Longitude': round(lon, 6)
            })
            property_id_counter += 1

df = pd.DataFrame(property_data)

# Step 3: Plot raw points
plt.figure(figsize=(12, 10))
colors = {'Residential': 'green', 'Commercial': 'blue', 'Vacant': 'orange'}
for t in df['Property_Type'].unique():
    plt.scatter(
        df[df['Property_Type'] == t]['Longitude'],
        df[df['Property_Type'] == t]['Latitude'],
        s=2, alpha=0.4, label=t, c=colors[t]
    )
plt.title("Raw Property Points by Type")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)
plt.show()

# Step 4: Cluster by type with HDBSCAN
clustered_df = []

for prop in ['Residential', 'Commercial', 'Vacant']:
    subset = df[df['Property_Type'] == prop].copy()
    if len(subset) < 10:
        continue

    X_scaled = StandardScaler().fit_transform(subset[['Latitude', 'Longitude']])
    labels = hdbscan.HDBSCAN(min_cluster_size=200).fit_predict(X_scaled)
    subset['Cluster_ID'] = labels
    clustered_df.append(subset)

df_clustered = pd.concat(clustered_df)

# Step 5: Plot clusters and boundaries
plt.figure(figsize=(14, 12))

for prop_type in df_clustered['Property_Type'].unique():
    data = df_clustered[df_clustered['Property_Type'] == prop_type]
    for cluster_id in sorted(data['Cluster_ID'].unique()):
        if cluster_id == -1:
            continue
        cluster_points = data[data['Cluster_ID'] == cluster_id]
        plt.scatter(
            cluster_points['Longitude'],
            cluster_points['Latitude'],
            s=1,
            alpha=0.4,
            label=f'{prop_type} Cluster {cluster_id}'
        )

        if len(cluster_points) >= 3:
            try:
                coords = cluster_points[['Longitude', 'Latitude']].values
                hull = ConvexHull(coords)
                polygon = Polygon(coords[hull.vertices])
                x, y = polygon.exterior.xy
                plt.plot(x, y, color='black', linewidth=1)
            except:
                pass  # Skip invalid clusters

plt.title("HDBSCAN Clusters with Boundaries by Property Type")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend(markerscale=10)
plt.grid(True)
plt.show()