from rembg import remove
from PIL import Image
import os

os.makedirs('images_nobg', exist_ok=True)

files = os.listdir('images')

for i, filename in enumerate(files):
    if filename.endswith('.jpg'):
        input_path = f"images/{filename}"
        output_path = f"images_nobg/{filename.replace('.jpg', '.png')}"
        
        try:
            img = Image.open(input_path)
            result = remove(img)
            result.save(output_path)
            print(f"{i}/{len(files)} - {filename}")
        except Exception as e:
            print(f"Error: {filename} - {e}")

            