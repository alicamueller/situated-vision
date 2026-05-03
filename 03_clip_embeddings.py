import clip
import torch
import pandas as pd
from PIL import Image
import numpy as np
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

df = pd.read_csv('online_collections_with_images.csv')
print(device)

embeddings = []

for index, row in df.iterrows():
    accession = row['Accession number']
    image_path = f"images/{accession}.jpg"
    try:
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(image)
        embeddings.append(embedding.cpu().numpy())
    except:
        print(f"Error: {accession}")
        embeddings.append(np.zeros((1, 512)))
    print(f"{index}/993")

np.save('embeddings.npy', np.vstack(embeddings))
print("Done!")
