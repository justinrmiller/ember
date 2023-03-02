import time
import sys
import torch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from torch import nn

from transformers import CLIPModel, AutoProcessor, AutoTokenizer

# model_name = "openai/clip-vit-base-patch32"
# model_name = "openai/clip-vit-large-patch14"ip

if len(sys.argv) > 1:
    device = sys.argv[1]
    model_name = sys.argv[2]
    input_filename = sys.argv[3]
    print(f"Setting device {device} and input_filename {input_filename}")
else:
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model_name = "openai/clip-vit-base-patch32"
    input_filename = "output.snappy.parquet"
    print(f"No arguments specified, defaulting to device {device} and input file {input_filename}")

modulo = 1000


print(f"Model loading... Device used: {device} - input_filename: {input_filename} - modulo: {modulo}")

model = CLIPModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# move the model to the device
model.to(device)

print("Model loaded and shipped to device...")

print(f"Reading embeddings from parquet file: {input_filename}")

limit = 100
name_embedding_df = pq.read_table(input_filename)

print(f"Embeddings count: {len(name_embedding_df)}")

# queries = [
#     "a photo of an animal",
#     "a photo of a vegetable",
#     "a photo of a mineral",
#     "a photo of a person",
# ]

queries = [
    "a photo of a horse"
]

text_inputs = tokenizer(queries, padding=True, return_tensors="pt").to(device)
text_features = model.get_text_features(**text_inputs)
text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

print(text_features.shape)

softmax_model = nn.Softmax(dim=0)

start_time = time.perf_counter()

processed_count = 0
import array

results = []

for name_embedding in name_embedding_df.to_batches():
    d = name_embedding.to_pydict()
    for name, embedding in zip(d['name'], d['embedding']):
        a = array.array('f', embedding)

        image_features = torch.frombuffer(a, dtype=torch.float32).to(device)

        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

        logit_scale = model.logit_scale.exp()

        torch.matmul(text_features, image_features.t()) * logit_scale

        similarity = torch.nn.functional.cosine_similarity(text_features, image_features) * logit_scale

        # print(f"Name: {name} - Query: {queries[0]} - probs: {similarity}")


#
# for name_embedding in name_embedding_df.to_batches():
#     d = name_embedding.to_pydict()
#     for name, embedding in zip(d['name'], d['embedding']):
#         a = array.array('f', embedding)
#
#         image_features = torch.frombuffer(a, dtype=torch.float32).to(device)
#
#         image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
#
#         logit_scale = model.logit_scale.exp()
#
#         torch.matmul(text_features, image_features.t()) * logit_scale
#
#         similarity = torch.nn.functional.cosine_similarity(text_features, image_features) * logit_scale
#         #
#         # if processed_count % modulo == 0:
#         #     print(f"Similarity Tensor: {similarity}")
#
#         probs = softmax_model(similarity)
#
#         if processed_count % modulo == 0:
#             probs_joined = ",".join([f"{prob:.2%}" for prob in probs])
#             print(f"{processed_count}|{name}|{probs_joined}\n")
#
#             results.append([processed_count, name] + probs.tolist())
#
#         #
#         # outputs = model(**inputs)
#         # logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
#         # probs = logits_per_image.softmax(dim=1)
#
#         processed_count += 1

# print(embeddings)

end_time = time.perf_counter()

generate_embeddings_timing = end_time - start_time

print(
    f"Time taken to query embeddings for {len(name_embedding_df)} records: {generate_embeddings_timing} seconds"
)
#
print("Writing parquet file.")

df = pd.DataFrame(results, columns=['processed_count', 'name'] + queries, dtype=float)

arrow_table = pa.Table.from_pandas(df)

pq.write_table(
    arrow_table,
    "queries.snappy.parquet",
    use_dictionary=False
)