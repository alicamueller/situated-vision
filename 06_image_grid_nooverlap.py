import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

embeddings = np.load('embeddings.npy')
df = pd.read_csv('online_collections_with_images.csv')

reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

n = len(embedding_2d)
grid_size = int(np.ceil(np.sqrt(n)))

scaler = MinMaxScaler(feature_range=(0, grid_size-1))
embedding_norm = scaler.fit_transform(embedding_2d)

grid_positions = np.array([[i, j] for i in range(grid_size) for j in range(grid_size)])

distances = cdist(embedding_norm, grid_positions)
assigned = set()
assignments = []

for i in range(n):
    sorted_positions = np.argsort(distances[i])
    for pos in sorted_positions:
        if pos not in assigned:
            assigned.add(pos)
            assignments.append(grid_positions[pos])
            break

assignments = np.array(assignments)

thumb_size = 50
canvas_size = grid_size * thumb_size

fig, ax = plt.subplots(figsize=(20, 20), facecolor='black')
ax.set_facecolor('black')

for i, row in df.iterrows():
    accession = row['Accession number']
    image_path = f"images/{accession}.jpg"
    
    try:
        img = Image.open(image_path).resize((thumb_size, thumb_size))
        x, y = assignments[i]
        ax.imshow(img, extent=[x, x+1, y, y+1], aspect='auto')
    except:
        pass

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.axis('off')
plt.tight_layout()
plt.savefig('umap_grid.png', dpi=200)
plt.show()
