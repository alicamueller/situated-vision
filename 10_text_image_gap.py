import clip
import torch
import pandas as pd
from PIL import Image
import numpy as np
import os
from numpy.linalg import norm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

df = pd.read_csv('online_collections_with_images.csv')
image_embeddings = np.load('embeddings.npy')

text_embeddings = []

for i, row in df.iterrows():
    text = str(row['Description'])
    tokens = clip.tokenize([text], truncate=True).to(device)
    
    with torch.no_grad():
        embedding = model.encode_text(tokens)
    
    text_embeddings.append(embedding.cpu().numpy())
    print(f"{i}/1000")

text_embeddings = np.vstack(text_embeddings)
np.save('text_embeddings.npy', text_embeddings)
print("Done!")

gaps = []
for i in range(len(df)):
    img_vec = image_embeddings[i]
    txt_vec = text_embeddings[i]
    
    # Kosinus-Distanz
    similarity = np.dot(img_vec, txt_vec) / (norm(img_vec) * norm(txt_vec))
    gaps.append(similarity)

df['clip_gap'] = gaps
df.to_csv('online_collections_with_gap.csv', index=False)

# Head or Tail, if you want 10 furthest away (head) or closest together (tail).
print(df[['Accession number', 'Description', 'clip_gap']].sort_values('clip_gap').tail(10))
