import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_csv('online_collections_with_gap.csv')
df_sorted = df.dropna(subset=['clip_gap']).sort_values('clip_gap')

furthest = df_sorted.head(10)
closest = df_sorted.tail(10)

fig, axes = plt.subplots(10, 2, figsize=(20, 30), facecolor='black')
fig.suptitle('CLIP Gap: Image vs. Text', color='white', fontsize=16)

for idx, (i, row) in enumerate(furthest.iterrows()):
    accession = row['Accession number']
    image_path = f"images_nobg/{accession}.png"
    
    # Bild
    ax_img = axes[idx, 0]
    ax_img.set_facecolor('black')
    try:
        img = Image.open(image_path).convert('RGBA')
        ax_img.imshow(img)
    except:
        ax_img.text(0.5, 0.5, 'No image', color='white', ha='center', va='center')
    ax_img.axis('off')
    
    # Text
    ax_txt = axes[idx, 1]
    ax_txt.set_facecolor('black')
    desc = str(row['Description'])[:150]
    ax_txt.text(0.05, 0.5, f"Gap: {row['clip_gap']:.3f}\n{desc}", 
                color='white', ha='left', va='center', wrap=True, fontsize=8)
    ax_txt.axis('off')

plt.tight_layout()
plt.savefig('gap_furthest.png', dpi=150)
plt.show()
