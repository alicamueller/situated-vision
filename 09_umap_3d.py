import umap
import numpy as np
import pandas as pd
import plotly.express as px

embeddings = np.load('embeddings.npy')
df = pd.read_csv('online_collections_with_images.csv')

reducer = umap.UMAP(n_components=3, random_state=42)
embedding_3d = reducer.fit_transform(embeddings)

df['x'] = embedding_3d[:, 0]
df['y'] = embedding_3d[:, 1]
df['z'] = embedding_3d[:, 2]

df['region'] = df['Geographical reference'].apply(
    lambda x: x.split('->')[0].strip() if isinstance(x, str) else 'Unknown'
)
main_regions = ['Africa', 'Oceania', 'Asia', 
                'Latin America and the Caribbean', 
                'Europe', 'Northern America']

df['region_clean'] = df['region'].apply(
    lambda x: x if x in main_regions else 'Other'
)

fig = px.scatter_3d(df, x='x', y='y', z='z', 
                    color='region_clean',
                    hover_data=['Accession number', 'Description'],
                    opacity=0.7)

fig.update_layout(
    paper_bgcolor='black',
    plot_bgcolor='black',
    scene=dict(
        bgcolor='black'
    )
)

fig.write_html('umap_3d.html')
print("Done!")
