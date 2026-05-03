import pandas as pd
import requests
import time 

with open('online_collections.csv', 'r') as f:
    content = f.read()

content = content.replace('undefined', '\n')

with open('online_collections_clean.csv', 'w') as f:
    f.write(content)

df = pd.read_csv('online_collections_clean.csv')

image_urls = []

for accession in df['Accession number']:
    url = f"https://dams.prm.ox.ac.uk/iiif/{accession}/manifest"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            manifest = response.json()
            image_url = manifest['items'][0]['items'][0]['items'][0]['body']['id']
            image_urls.append(image_url)
        else:
            image_urls.append(None)
    except:
        image_urls.append(None)
    time.sleep(0.5)

df['image_url'] = image_urls
df.to_csv('online_collections_with_images.csv', index=False)
print("Done!")
print(f"Images found: {df['image_url'].notna().sum()} von {len(df)}")
