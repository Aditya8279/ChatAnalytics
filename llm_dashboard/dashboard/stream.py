import os
import json
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
import numpy as np
import faiss
from transformers import CLIPProcessor, CLIPModel

# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
image_dir = images

# Step 1: Load data from JSON Line
# ldjson_path = "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"
# records = [json.loads(line) for line in open(ldjson_path, "r")]

# df = pd.DataFrame(records)

image_embeddings = []
metadata_store = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    uniq_id = row.get("uniq_id")
    product_title = row.get("product_title", "")
    metadata = {
        "uniq_id": uniq_id,
        "product_title": product_title,
        "brand": row.get("brand", ""),
        "color": row.get("color", ""),
        "product_type": row.get("product_type", ""),
        "price": row.get("price", ""),
    }

    image_path = os.path.join(image_dir, f"{uniq_id}.jpg")
    if not os.path.exists(image_path):
        print(f"Image not found for {image_path}, skipping.")
        continue  # skip if image doesn't exist

    try:

        image = Image.open(image_path).convert("RGB")


        with torch.no_grad():

            text_inputs = clip_processor(text=str(metadata), return_tensors="pt").to(device)
            text_emb = clip_model.get_text_features(**text_inputs)
            image_inputs = clip_processor(images=image, return_tensors="pt").to(device)
            image_emb = clip_model.get_image_features(**image_inputs)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
            combined_emb = (text_emb + image_emb) / 2
            combined_emb = combined_emb.cpu().numpy().astype("float32")

            image_embeddings.append(combined_emb[0])
            metadata_store.append(metadata)

    except Exception as e:
        print(f"Failed on {uniq_id}: {e}")
        os.remove(image_path)  # Remove the image if processing fails
        continue

embedding_dim = image_embeddings[0].shape[0]
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(image_embeddings))

print(f"Stored {len(image_embeddings)} items in FAISS.")

faiss.write_index(index, f"clip_fashion_index_{version}.faiss")
with open(f"clip_fashion_metadata_{version}.json", "w") as f:
    json.dump(metadata_store, f)



def search_by_text(text_query, top_k=5):

    inputs = clip_processor(text=[text_query], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        text_embedding = clip_model.get_text_features(**inputs)
    text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
    text_vector = text_embedding.cpu().numpy().astype("float32")


    D, I = index.search(text_vector, top_k)
    results = [(metadata_store[i], float(D[0][idx])) for idx, i in enumerate(I[0])]
    return results






def show_results(results, image_folder="./images"):
    plt.figure(figsize=(15, 3))
    for idx, (filename, score) in enumerate(results):
        image_path = os.path.join(image_folder, filename['uniq_id'] + '.jpg')
        image = Image.open(image_path).convert("RGB")
        
        plt.subplot(1, len(results), idx + 1)
        plt.imshow(image)
        plt.title(f"recommendations")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# Example usage:


query = "blue t shirt for females"
results = search_by_text(query, top_k=5)
for fname, score in results:
    print(f"{fname} (similarity: {score:.4f})")
show_results(results, image_folder=images)

##########################################################

import json
import pandas as pd
datapath= "marketing_sample_for_amazon_com-amazon_fashion_products__20200201_20200430__30k_data.ldjson"

data = pd.read_json(datapath, lines=True)

data.columns

df= data[['uniq_id',
    'product_name', 
    'large'
    ]]
df = df.sample(1000)
print(df.shape)


import requests
import os
from tqdm import tqdm_notebook as tqdm
images= os.listdir("images")
urls= df.values
for url in tqdm(urls):
    if url[0] + '.jpg' not in images:
        if 'http' in str(url[2]):
            response = requests.get(url[2])
            with open(f"images/{url[0]}.jpg", "wb") as f:
                f.write(response.content)
    # break
response.content

dict(zip(data.columns, data.iloc[1].values))