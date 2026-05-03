import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

embeddings = np.load('embeddings.npy')
df = pd.read_csv('online_collections_with_images.csv')

reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

fig, ax = plt.subplots(figsize=(40, 30), facecolor='black')
ax.set_facecolor('black')

for i, row in df.iterrows():
    accession = row['Accession number']
    
    try:
        image_path = f"images_nobg/{accession}.png"
        img = Image.open(image_path).convert('RGBA').resize((50, 50))
        x, y = embedding_2d[i]
        ax.imshow(img, extent=[x-0.5, x+0.5, y-0.5, y+0.5], aspect='auto')
    except:
        pass

ax.set_xlim(embedding_2d[:, 0].min() - 1, embedding_2d[:, 0].max() + 1)
ax.set_ylim(embedding_2d[:, 1].min() - 1, embedding_2d[:, 1].max() + 1)

plt.savefig('umap_images.png', dpi=300)
plt.show()
