# import re

# from simple_tokenizer import SimpleTokenizerV1, SimpleTokenizerV2

# text = "Hello, world. This, is a test."
# result = re.split(r'([,.]|\s)', text)
# print(result)
# print(len(result))

# result = [item for item in result if item.strip()]
# print(result)

# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

# print("Total number of characters:", len(raw_text))
# print(raw_text[:99])

# text = "Hello, world. Is this-- a test?"
# result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
# result = [item for item in result if item.strip()]
# print(result)

# preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
# preprocessed = [item.strip() for item in preprocessed if item.strip()]
# print(len(preprocessed))
# print(preprocessed[:30])

# all_words = sorted(set(preprocessed))
# vocab_size = len(all_words)
# print("Vocabulary size:", vocab_size)

# vocab = {token: integer for integer, token in enumerate(all_words)}
# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i > 50:
#         break

# tokenizer = SimpleTokenizerV1(vocab)

# text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable
# pride."""
# ids = tokenizer.encode(text)
# print(ids)
# print(tokenizer.decode(ids))

# text = "Hello, do you like tea?"
# # pop KeyError: 'Hello' not in vocab
# # print(tokenizer.encode(text))

# all_tokens = sorted(list(set(preprocessed)))
# all_tokens.extend(["<|endoftext|>", "<|unk|>"])

# vocab = {token:integer for integer,token in enumerate(all_tokens)}

# print(len(vocab.items()))

# for i, item in enumerate(list(vocab.items())[-5:]):
#     print(item)

# text1 = "Hello, do you like tea?"
# text2 = "In the sunlit terraces of the palace."
# text = " <|endoftext|> ".join((text1, text2))
# print(text)

# tokenizer = SimpleTokenizerV2(vocab)
# print(tokenizer.encode(text))
# print(tokenizer.decode(tokenizer.encode(text)))

# from importlib.metadata import version
# import tiktoken
# print("tiktoken version:", version("tiktoken"))

# tokenizer = tiktoken.get_encoding("gpt2")

# text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)

# strings = tokenizer.decode(integers)
# print(strings)

# text = "Akwirw ier"
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)

# strings = tokenizer.decode([33901])
# print(strings)
# strings = tokenizer.decode([86])
# print(strings)
# strings = tokenizer.decode([343])
# print(strings)
# strings = tokenizer.decode([86])
# print(strings)
# strings = tokenizer.decode([220])
# print(strings)
# strings = tokenizer.decode([959])
# print(strings)

# import tiktoken

# tokenizer = tiktoken.get_encoding("gpt2")

# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()
# enc_text = tokenizer.encode(raw_text)
# print(len(enc_text))

# enc_sample = enc_text[50:]

# context_size = 4
# x = enc_sample[:context_size]
# y = enc_sample[1:context_size + 1]

# print(f"x: {x}")
# print(f"y:      {y}")

# for i in range(1, context_size + 1):
#     context = enc_sample[:i]
#     desired = enc_sample[i]
#     print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

from dataset import create_dataloader_v1

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)


import torch 

input_ids = torch.tensor([2, 3, 5, 1])
vocab_size = 6
output_dim = 3

torch.manual_seed(123)

embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

print(embedding_layer(torch.tensor([3])))

print(embedding_layer(input_ids))

vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token IDs:\n", inputs)
print("\nInputs shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)