import torch
import glob
import time
import sys

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa

from torch import nn

from PIL import Image

from transformers import CLIPModel, AutoProcessor, AutoTokenizer

# device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

# model_name = "openai/clip-vit-base-patch32"
# model_name = "openai/clip-vit-large-patch14"ip

if len(sys.argv) > 1:
    device = sys.argv[1]
    max_number_of_images = int(sys.argv[2])
    model_name = sys.argv[3]
    path = "data/test2017/*.jpg"
    print(f"Setting device {device} and max_number_of_images {max_number_of_images}")
else:
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_name = "openai/clip-vit-base-patch32"
    max_number_of_images = 55000
    path = "data/test2017/*.jpg"
    print("No arguments specified, defaulting to device {device} and max_number_of_images {max_number_of_images}")

modulo = 100

print(f"Model loading... Device used: {device} - max_number_of_images: {max_number_of_images} - modulo: {modulo}")

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
# tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

model = CLIPModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# move the model to the device
model.to(device)

print("Model loaded and shipped to device...")

jpg_files = glob.glob(path)

# import requests
# image_paths = [
#     "http://images.cocodataset.org/val2017/000000039769.jpg"
# ]

image_paths = jpg_files[:max_number_of_images]
number_of_embeddings = len(image_paths)

# establish embeddings tensor
# embeddings = torch.empty(size=(number_of_embeddings, 512))
# embeddings.to(device)
embeddings = []

print(f"Embeddings count: {len(embeddings)}")

queries = [
    "a photo of an animal",
    "a photo of a vegetable",
    "a photo of a mineral",
    "a photo of a person",
]

text_inputs = tokenizer(queries, padding=True, return_tensors="pt").to(device)
text_features = model.get_text_features(**text_inputs)
text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

print(text_features.shape)

softmax_model = nn.Softmax(dim=0)

start_time = time.perf_counter()

for image_path_index, image_path in enumerate(image_paths):
    # image = Image.open(requests.get(image_path, stream=True).raw)
    image = Image.open(image_path)

    inputs = processor(images=image, return_tensors="pt").to(device)

    image_features = model.get_image_features(**inputs)

    # print(image_features)

    # add embeddings to the embeddings tensor
    # embeddings[image_path_index] = image_features
    embeddings.append([image_path, image_features.detach().cpu().numpy()[0]])
    # print(embeddings[image_path_index].size()[0])
    # print(embeddings[image_path_index].element_size())
    # embeddings.append(image_features)
    # print(f"Embeddings Individual Length: {len(embeddings[image_path_index])}")
    # print(f"Embeddings Length: {len(embeddings)}")

    if image_path_index % modulo == 0:
        print(f"Size of image {image_path_index} features embedding: {len(image_features[0])}")


    #
    # image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
    #
    # logit_scale = model.logit_scale.exp()
    #
    # torch.matmul(text_features, image_features.t()) * logit_scale
    #
    # similarity = torch.nn.functional.cosine_similarity(text_features, image_features) * logit_scale
    #
    # if image_path_index % modulo == 0:
    #     print(f"Similarity Tensor: {similarity}")
    #
    # probs = softmax_model(similarity)
    #
    # for query_index, prob in enumerate(queries):
    #     # print(f"Image: {image_path_index} - Image Path: {image_path}\n\tQuery: {queries[query_index]}\n\tProb: {probs[i]}")
    #     if image_path_index % modulo == 0:
    #         print(f"image_{image_path_index}_{image_path}_{query_index}_{probs[query_index]}")

    # outputs = model(**inputs)
    # logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    # probs = logits_per_image.softmax(dim=1)
    image.close()

# print(embeddings)

end_time = time.perf_counter()

generate_embeddings_timing = end_time - start_time

print(
    f"Time taken to generate embeddings for {len(embeddings)} records: {generate_embeddings_timing} seconds"
)

print("Writing parquet file.")

df = pd.DataFrame(embeddings, columns=['name', 'embedding'], dtype=float)

# print(df)

arrow_table = pa.Table.from_pandas(df)

pq.write_table(arrow_table, "output.snappy.parquet")

pq.write_table(
    arrow_table,
    "output.snappy.parquet",
    use_dictionary=False,
    compression="SNAPPY"
)