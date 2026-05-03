import clip
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from numpy.linalg import norm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_embeddings = np.load('embeddings.npy')
df = pd.read_csv('online_collections_with_images.csv')

query = "face"

tokens = clip.tokenize([query]).to(device)
with torch.no_grad():
    query_embedding = model.encode_text(tokens).cpu().numpy()

similarities = []
for i in range(len(image_embeddings)):
    img_vec = image_embeddings[i].flatten()
    q_vec = query_embedding.flatten()
    sim = float(np.dot(q_vec, img_vec) / (norm(q_vec) * norm(img_vec)))
    similarities.append(sim)

similarities = np.array(similarities)
valid_idx = [i for i, s in enumerate(similarities) if not np.isnan(s)]
valid_similarities = np.array([similarities[i] for i in valid_idx])
top_10_idx = np.array(valid_idx)[np.argsort(valid_similarities)[-10:][::-1]]

fig, axes = plt.subplots(2, 5, figsize=(20, 10), facecolor='black')
fig.suptitle(f'CLIP Zero-Shot: "{query}"', color='white', fontsize=16)

for plot_idx, idx in enumerate(top_10_idx.tolist()):
    ax = axes[plot_idx // 5][plot_idx % 5]
    ax.set_facecolor('black')
    row = df.iloc[idx]
    accession = row['Accession number']
    image_path = f"images_nobg/{accession}.png"
    try:
        img = Image.open(image_path).convert('RGBA')
        ax.imshow(img)
    except:
        ax.text(0.5, 0.5, 'No image', color='white', ha='center', va='center')
    ax.set_title(f"{similarities[idx]:.3f}", color='white', fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig(f'zero_shot_{query.replace(" ", "_")}.png', dpi=150)
plt.show()
