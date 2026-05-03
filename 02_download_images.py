import pandas as pd
import requests
import os
import time

df = pd.read_csv('online_collections_with_images.csv')
os.makedirs('images', exist_ok=True)

for index, row in df.iterrows():
    url = row['image_url']
    accession = row['Accession number']
    filename = f"images/{accession}.jpg"
    try:
        response = requests.get(url, timeout=10)
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"{index}/993 - {accession}")
    except:
        print(f"Error: {accession}")
    time.sleep(0.3)
