import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import MinMaxScaler
import umap
from scipy.spatial.distance import cdist

embeddings = np.load('embeddings.npy')
df = pd.read_csv('online_collections_with_images.csv')

reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

n = len(embedding_2d)
grid_size = int(np.ceil(np.sqrt(n)))

# Normalisiere auf Grid
scaler = MinMaxScaler(feature_range=(0, grid_size-1))
embedding_norm = scaler.fit_transform(embedding_2d)

# Erstelle Grid-Positionen
grid_x, grid_y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
grid_positions = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1).astype(float)

# Berechne Distanzmatrix und löse Assignment
cost = cdist(embedding_norm, grid_positions[:n])
row_ind, col_ind = linear_sum_assignment(cost)
assigned = grid_positions[col_ind]

fig, ax = plt.subplots(figsize=(40, 40), facecolor='black')
ax.set_facecolor('black')

for i, row in df.iterrows():
    accession = row['Accession number']
    image_path = f"images_nobg/{accession}.png"
    
    try:
        img = Image.open(image_path).convert('RGBA').resize((50, 50))
        x, y = assigned[i]
        ax.imshow(img, extent=[x, x+1, y, y+1], aspect='auto')
    except:
        pass

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.axis('off')
plt.tight_layout()
plt.savefig('umap_grid_form.png', dpi=200)
plt.show()

