## Encoding input data text by hand
with open('input.txt', 'r') as input_text_file:
    input_text = input_text_file.read()
print(len(input_text))
input_text_sorted = sorted(list(set(input_text)))
print("Loaded chars:")
print(input_text_sorted)
print(len(input_text_sorted))

# Char to in
stoi = {ch: i for i, ch in enumerate(input_text_sorted)}
print(stoi)
itos = {i: ch for i, ch in enumerate(input_text_sorted)}
print(itos)

encode = lambda string: [stoi[c] for c in string]
decode = lambda list_of_ints: [itos[_int] for _int in list_of_ints]

print(encode("Hello, world!"))
print(''.join(decode(encode("Hello, world!"))))


## Encoding input data text with PyTorch
import torch

data = torch.tensor(encode(input_text), dtype=torch.long) # The dtype=torch.long is usually added for efficiency reasons
# data = torch.tensor(encode(input_text))
print(data.shape, data.dtype)
print(data[:1000])
# Numerical representation of the first 1000 characters. For example 0 may be a '\n' character, 1 space, etc.

# Splitting the data
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
print(train_data[:block_size+1])

# Creating sequential blocks. We want to predict the next character based on the n previous chars
x = train_data[:block_size]
# for the next characters we want to predict
for t in range(block_size):
    context = x[:t+1]
    target = x[t]
    print(f"Based on {context}, the target is: {target}")

# "PyTorching" it
TORCH_SEED = 356556
torch.manual_seed(TORCH_SEED) # setting the seed for random numbers generation

block_size = 8
batch_size = 4  # This allows parallel processing


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Shifting the matrix left in the 4x8 matrix
    # for every n items on position jth in the matrix on the ith rows
    x = torch.stack([data[i:i+block_size] for i in ix])
    # we have one particular target value in the jth column and ith row
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f"when input is {context.tolist()} the target: {target}")
