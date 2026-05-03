import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from PIL import Image

image_embeddings = np.load('embeddings.npy')
text_embeddings = np.load('text_embeddings.npy')
df = pd.read_csv('online_collections_with_gap.csv')

combined = np.vstack([image_embeddings, text_embeddings])

reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(combined)

n = len(image_embeddings)
img_coords = embedding_2d[:n]
txt_coords = embedding_2d[n:]

fig, ax = plt.subplots(figsize=(40, 30), facecolor='black')
ax.set_facecolor('black')

# Linien zuerst (damit sie hinter den Bildern sind)
for i in range(n):
    ax.plot([img_coords[i, 0], txt_coords[i, 0]], 
            [img_coords[i, 1], txt_coords[i, 1]], 
            'white', alpha=0.1, linewidth=0.3)

# Textpunkte
ax.scatter(txt_coords[:, 0], txt_coords[:, 1], c='red', alpha=0.6, s=8, zorder=2)

print("Images:", img_coords.min(axis=0), img_coords.max(axis=0))
print("Texts:", txt_coords.min(axis=0), txt_coords.max(axis=0))

# Bilder
for i, row in df.iterrows():
    accession = row['Accession number']
    image_path = f"images_nobg/{accession}.png"
    try:
        img = Image.open(image_path).convert('RGBA').resize((20, 20))
        x, y = img_coords[i]
        ax.imshow(img, extent=[x-0.8, x+0.8, y-0.8, y+0.8], aspect='auto', zorder=3)
    except:
        pass

ax.set_xlim(embedding_2d[:, 0].min() - 1, embedding_2d[:, 0].max() + 1)
ax.set_ylim(embedding_2d[:, 1].min() - 1, embedding_2d[:, 1].max() + 1)
plt.savefig('text_image_lines.png', dpi=200)
plt.show()

# 3D Version
reducer3d = umap.UMAP(n_components=3, random_state=42)
embedding_3d = reducer3d.fit_transform(combined)

img_coords_3d = embedding_3d[:n]
txt_coords_3d = embedding_3d[n:]

fig3d = go.Figure()

fig3d.add_trace(go.Scatter3d(
    x=img_coords_3d[:, 0], y=img_coords_3d[:, 1], z=img_coords_3d[:, 2],
    mode='markers', marker=dict(size=3, color='blue', opacity=0.6), name='image'))

fig3d.add_trace(go.Scatter3d(
    x=txt_coords_3d[:, 0], y=txt_coords_3d[:, 1], z=txt_coords_3d[:, 2],
    mode='markers', marker=dict(size=3, color='red', opacity=0.6), name='text'))

fig3d.update_layout(
    paper_bgcolor='black',
    scene=dict(bgcolor='black')
)

fig3d.write_html('text_image_3d.html')
print("3D Done!")
