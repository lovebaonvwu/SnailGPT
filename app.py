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

# from dataset import create_dataloader_v1

# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#     raw_text = f.read()

# dataloader = create_dataloader_v1(
#     raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
# data_iter = iter(dataloader)
# first_batch = next(data_iter)
# print(first_batch)


# import torch 

# input_ids = torch.tensor([2, 3, 5, 1])
# vocab_size = 6
# output_dim = 3

# torch.manual_seed(123)

# embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# print(embedding_layer.weight)

# print(embedding_layer(torch.tensor([3])))

# print(embedding_layer(input_ids))

# vocab_size = 50257
# output_dim = 256
# token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

# max_length = 4
# dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
# data_iter = iter(dataloader)
# inputs, targets = next(data_iter)
# print("Token IDs:\n", inputs)
# print("\nInputs shape:\n", inputs.shape)

# token_embeddings = token_embedding_layer(inputs)
# print(token_embeddings.shape)

# context_length = max_length
# pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# pos_embeddings = pos_embedding_layer(torch.arange(context_length))
# print(pos_embeddings.shape)

# input_embeddings = token_embeddings + pos_embeddings
# print(input_embeddings.shape)



# ########################### #
# Coding Attention Mechanisms #
# ########################### #

# ########################################################### #
# Simple Self Attention Mechanisms for a single input element #
# ########################################################### #

import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     
   [0.55, 0.87, 0.66], # journey
   [0.57, 0.85, 0.64], # starts
   [0.22, 0.58, 0.33], # with
   [0.77, 0.25, 0.10], # one
   [0.05, 0.80, 0.55]] # step
)

## first calculate attention scores 'w'

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print("attn_scores_2: ", attn_scores_2)

## normalize attention stores to get attention weights 'a'

attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("tmp Attention weights: ", attn_weights_2_tmp)
print("tmp Sum: ", attn_weights_2_tmp.sum())

def softmax_naive(x):
    print("torch.exp(x): ", torch.exp(x))
    print("torch.exp(x).sum(dim=0): ", torch.exp(x).sum(dim=0))
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("naive Attention weights:", attn_weights_2_naive)
print("naive Sum:", attn_weights_2_naive.sum())

## pytorch softmax 

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("pytorch Attention weights:", attn_weights_2)
print("pytorch Sum:", attn_weights_2.sum())

## calculate context vector for x_2

query = inputs[1]
print("query shape: ", query.shape)
context_vec_2 = torch.zeros(query.shape)
print("context_vec_2 shape: ", context_vec_2.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i] * x_i
print("context_vec_2: ", context_vec_2)

## attention scores for all 6 inputs, each input should have 6 scores about other inputs in the current seqence

attn_scores = torch.empty(6, 6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print("attn_scores: ", attn_scores)

## pytorch accelerate

print("inputs.T: ", inputs.T)
attn_scores = inputs @ inputs.T
print("@ attn_scores: ", attn_scores)

## normalize attention scores

attn_weights = torch.softmax(attn_scores, dim=-1)
print("attn_weights: ", attn_weights)

## check if attention weights of each inputs row can be summed up to 1.0

row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)
print("All row sums:", attn_weights.sum(dim=-1))

## calculate all context vectors for inputs

all_context_vecs = attn_weights @ inputs
print("all_context_vecs: ", all_context_vecs)

# ############################### #
# Computing the attention weights #
# ############################### #

x_2 = inputs[1]
print("inputs: ")
print(inputs)
d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
print("W_query: ")
print(W_query)

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print("x_2:")
print(x_2)
print("query_2: ")
print(query_2)

keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)

keys_2 = keys[1] 
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)

a = torch.tensor(1)
b = torch.tensor([1])

print(a.shape)
print(b.shape)
print(a == b)

t = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
print("invert t: ", t.T)
