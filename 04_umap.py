import umap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

embeddings = np.load('embeddings.npy')
df = pd.read_csv('online_collections_with_images.csv')

reducer = umap.UMAP(n_components=2, random_state=42)
embedding_2d = reducer.fit_transform(embeddings)
print(embedding_2d.shape)

df['region'] = df['Geographical reference'].apply(
    lambda x: x.split('->')[0].strip() if isinstance(x, str) else 'Unknown'
)

main_regions = ['Africa', 'Oceania', 'Asia', 
                'Latin America and the Caribbean', 
                'Europe', 'Northern America']

df['region_clean'] = df['region'].apply(
    lambda x: x if x in main_regions else 'Other'
)

colors = {'Africa': 'red', 'Oceania': 'blue', 'Asia': 'green',
          'Latin America and the Caribbean': 'orange', 
          'Europe': 'purple', 'Northern America': 'brown', 'Other': 'grey'}

plt.figure(figsize=(14, 10))
for region, color in colors.items():
    mask = df['region_clean'] == region
    plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                c=color, label=region, alpha=0.6, s=15)

plt.legend()
plt.title('CLIP Embeddings - nach Region')
plt.savefig('umap_regions.png', dpi=150)
plt.show()
