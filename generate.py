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
    max_number_of_images = 50000
    path = "data/test2017/*.jpg"
    print("No arguments specified, defaulting to device {device} and max_number_of_images {max_number_of_images}")

chunk_size = 50

modulo = 10


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

image_paths = jpg_files[:max_number_of_images]
number_of_embeddings = len(image_paths)

embeddings = []

print(f"Embeddings count: {len(embeddings)}")
#
# queries = [
#     "a photo of an animal",
#     "a photo of a vegetable",
#     "a photo of a mineral",
#     "a photo of a person",
# ]
#
# text_inputs = tokenizer(queries, padding=True, return_tensors="pt").to(device)
# text_features = model.get_text_features(**text_inputs)
# text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
#
# print(text_features.shape)
#
# softmax_model = nn.Softmax(dim=0)

start_time = time.perf_counter()


def chunks(list, n):
    """ Yield successive n-sized chunks from list.
    """
    for i in range(0, len(list), n):
        yield list[i:i+n]


chunked_image_paths = chunks(image_paths, chunk_size)

processed_count = 0

for chunk in chunked_image_paths:
    # image = Image.open(requests.get(image_path, stream=True).raw)
    images = [Image.open(image_path) for image_path in chunk]

    inputs = processor(images=images, return_tensors="pt").to(device)

    image_features_for_partition = model.get_image_features(**inputs)

    # print(image_features_for_partition)

    for index, image_features in enumerate(image_features_for_partition):
        embeddings.append([chunk[index], image_features.detach().cpu().numpy()])

    processed_count += len(image_features_for_partition)

    if processed_count % modulo == 0:
        print(f"Processed images: {processed_count}.")

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
    #
    # outputs = model(**inputs)
    # logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    # probs = logits_per_image.softmax(dim=1)
    for image in images:
        image.close()
        del image
    del images

# print(embeddings)

end_time = time.perf_counter()

generate_embeddings_timing = end_time - start_time

print(
    f"Time taken to generate embeddings for {len(embeddings)} records: {generate_embeddings_timing} seconds"
)

print("Writing parquet file.")

df = pd.DataFrame(embeddings, columns=['name', 'embedding'], dtype=float)

arrow_table = pa.Table.from_pandas(df)

pq.write_table(
    arrow_table,
    "output.snappy.parquet",
    use_dictionary=False
)