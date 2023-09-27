from itertools import zip_longest

from matplotlib import pyplot as plt

words = open("names.txt", 'r').read().splitlines()

print(words[:10])
print("Shortest name: ", min(len(word) for word in words))
print("Longest name: ", max(len(word) for word in words))

for w in words[:1]:
    print("w: ", w)
    print("w[1:]: ", w[1:])  # Thanks to this used in zip, we will no longer continue when we reach the end stop
    print("zip(w, w[1:]): ", zip(w, w[1:]))
    for ch1, ch2 in zip(w, w[1:]):
        print(ch1, ch2)

# Adding the special characters for marking the beginning and the end of the word
for w in words[:3]:
    print("w: ", w)
    print("w[1:]: ", w[1:])  # Thanks to this used in zip, we will no longer continue when we reach the end stop
    names_with_special_chars = ['.'] + list(w) + ['.']
    print("zip(w, names_with_special_chars[1:]): ", zip(names_with_special_chars, names_with_special_chars[1:]))
    for ch1, ch2 in zip(names_with_special_chars, names_with_special_chars[1:]):
        print(ch1, ch2)

## Basic approaches and techniques
# Counting the bigrams to find out which bigrams are the most common

bigrams_counts = {}
for w in words:
    names_with_special_chars = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(names_with_special_chars, names_with_special_chars[1:]):
        bigram = (ch1, ch2)
        """
        # Less efficient, equivalent
        if bigram in bigrams_counts:
            bigrams_counts[bigram] += 1
        else:
            bigrams_counts[bigram] = 1 
        """
        # More efficient
        # With the second argument of 0, we handle the 'None' values
        bigrams_counts[bigram] = bigrams_counts.get(bigram, 0) + 1

print("bigrams_counts")
print(bigrams_counts)
print(sorted(bigrams_counts.items(), key=lambda key_value: key_value[1], reverse=True))

# But even with the small efficiency improvement, this is not very effective. In the practise, special
# datastructures like Tensors from the Tensorflow are used

import torch

a = torch.zeros((3, 5), dtype=torch.int32)
print("a: ", a)
a[2, 4] += 1
print("a: ", a)

ALPHABET_SIZE_WITH_STOP_SYMBOL = 28
## Basic approach of mapping chars to tensors
N = torch.zeros((ALPHABET_SIZE_WITH_STOP_SYMBOL + 1, ALPHABET_SIZE_WITH_STOP_SYMBOL + 1))  # +1 for the start/stop sign
characters = list(set(''.join(words)))
characters_sorted = sorted(characters)
characters_to_int = {s: i for i, s in enumerate(characters_sorted)}
print(characters_to_int['a'])
# print(characters_to_int['<S>']) # This would not work, we need to define this
characters_to_int['<S>'] = 27
characters_to_int['<E>'] = 28

for w in words:
    names_with_special_chars = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(names_with_special_chars, names_with_special_chars[1:]):
        index_1 = characters_to_int[ch1]
        index_2 = characters_to_int[ch2]
        N[index_1, index_2] += 1

print("N:")
print(N)

## Tensors and its operations
print("type(N): ", type(N))
print("type(N[2][1]): ", type(N[2][1]))
# If we want a concrete value:
print("type(N[2][1]): ", type(N[2][1].item()))
print(N[4][1].item())

characters_to_int = {i: s for i, s in enumerate(characters_sorted)}


def plot_bigrams():
    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap='Blues')
    for i in range(27):
        for j in range(27):
            chstr = integer_of_string[i] + integer_of_string[j]
            plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
            plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
    plt.axis('off')
    plt.show()


# plot_bigrams()

# Edits for a better standards: using the dots instead of <S> and <E>
# We need to shift the entire set of characters
characters = sorted(list(set(''.join(words))))
characters_to_int = {s: i + 1 for i, s in enumerate(characters)}
characters_to_int['.'] = 0
integer_of_string = {i: s for s, i in characters_to_int.items()}

for w in words:
    names_with_special_chars = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(names_with_special_chars, names_with_special_chars[1:]):
        index_1 = characters_to_int[ch1]
        index_2 = characters_to_int[ch2]
        N[index_1, index_2] += 1

# plot_bigrams()


print("# Normalization")
p = N[0]
print("p before: ", p)
# after
p = N[0].float()
p = p / p.sum()
print("p after: ", p)

SAMPLES_SEED_VALUE = 2485785757  # random, but if we use this also later in the generator,
# we should end up with the same results
# one sample example
g = torch.Generator().manual_seed(SAMPLES_SEED_VALUE)
prob_distribution = torch.rand(3, generator=g)
prob_distribution = prob_distribution / prob_distribution.sum()
print("Probability distribution:")
print(prob_distribution)
print(prob_distribution.shape)

for i in range(20):
    g = torch.Generator().manual_seed(SAMPLES_SEED_VALUE)
    prob_distribution = torch.rand(3, generator=g)
    prob_distribution = prob_distribution / prob_distribution.sum()
    print("Probability distribution")
    print(prob_distribution)
    # we have the same samples here thanks to using the same seed

# we will use this to generate a prob distribution
tensors_samples = torch.multinomial(prob_distribution, num_samples=100, replacement=True, generator=g)
print("sample generated by the probability above:")
print(tensors_samples)

out = []
index = 0
while True:
    prob_distribution = N[index].float()
    # problematic part: normalizing inside of the cycle
    prob_distribution = prob_distribution / prob_distribution.sum()  # very ineffective
    index = torch.multinomial(prob_distribution, num_samples=1, replacement=True, generator=g).item()
    out.append(integer_of_string[index])
    if index == 0:
        break
print("".join(out))

"""
chars = sorted(list(set(''.join(words))))
string_to_integer = {s:i+1 for i,s in enumerate(chars)}
string_to_integer['.'] = 0
integer_of_string = {i:s for s,i in string_to_integer.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = string_to_integer[ch1]
        ix2 = string_to_integer[ch2]
        N[ix1, ix2] += 1
"""
g = torch.Generator().manual_seed(SAMPLES_SEED_VALUE)

print("shape of prob_distribution: ", prob_distribution.shape)


def sample_dataset():
    g = torch.Generator().manual_seed(SAMPLES_SEED_VALUE)

    print("shape of prob_distribution: ", prob_distribution.shape)

    for i in range(5):

        out = []
        ix = 0
        while True:
            p = N[ix].float()
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(integer_of_string[ix])
            if ix == 0:
                break
        print(''.join(out))


# # effectivity
# better for computational effectiveness
print(prob_distribution.sum())

# Populating frequencies matrix again
N = torch.zeros((27, 27), dtype=torch.int32)

chars = sorted(list(set(''.join(words))))
string_to_integer = {s: i + 1 for i, s in enumerate(chars)}
string_to_integer['.'] = 0
integer_of_string = {i: s for s, i in string_to_integer.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs, chs[1:]):
        ix1 = string_to_integer[ch1]
        ix2 = string_to_integer[ch2]
        N[ix1, ix2] += 1

print("N[0]: ", N[0])

PROB_MATRIX = N.float()
print("PROB_MATRIX: ", PROB_MATRIX)
# This is not what we want!!! We want to divide all the rows by their sums
print("PROB_MATRIX.sum(): ", PROB_MATRIX.sum())
PROB_MATRIX = PROB_MATRIX / PROB_MATRIX.sum()
print("PROB_MATRIX: ", PROB_MATRIX)
print("PROB_MATRIX.sum(): ", PROB_MATRIX.sum())

print("PROB_MATRIX.shape: ", PROB_MATRIX.shape)
print("PROB_MATRIX.sum(0): ", PROB_MATRIX.sum(0))
print("PROB_MATRIX.sum(0).shape: ", PROB_MATRIX.sum(0).shape)
# This is what we really want!
print("vs:")
# Columns
print("PROB_MATRIX.sum(0, keepdim=True): ", PROB_MATRIX.sum(0, keepdim=True))
print("PROB_MATRIX.sum(0, keepdim=True).shape: ", PROB_MATRIX.sum(0, keepdim=True).shape)
# Rows counts sum
print("PROB_MATRIX.sum(1, keepdim=True).shape: ", PROB_MATRIX.sum(1, keepdim=True))
print("PROB_MATRIX.sum(1, keepdim=True).shape: ", PROB_MATRIX.sum(1, keepdim=True).shape)
# We have finally found what do we want, now we can finally go back to normalizing again
# However, not that fast! We need to be sure that the operations on Tensors re valid
print("Is it ok to do the Tenzors operation with these two tensors?")
print("PROB_MATRIX.shape():", PROB_MATRIX.shape)
print("PROB_MATRIX.sum(1).shape: ", PROB_MATRIX.sum(1).shape)
print("PROB_MATRIX.sum(1, keepdim=True).shape: ", PROB_MATRIX.sum(1, keepdim=True).shape)
print("These dimensions are OK for the tensors operations")
PROB_MATRIX = N.float()
PROB_MATRIX = PROB_MATRIX / PROB_MATRIX.sum(1)

print("PROB_MATRIX[0].sum() ", PROB_MATRIX[0].sum())  # row not summing up to 1
print("PROB_MATRIX[:,0]", PROB_MATRIX[:, 0].sum())  # column summing up to 1
# This is the cause of torch adding the dimension internally while doing operations on (27,27) / (27) => (27/27) / (1,27)
PROB_MATRIX = N.float()
PROB_MATRIX = PROB_MATRIX / PROB_MATRIX.sum(1, keepdim=True)

print("PROB_MATRIX[0].sum() ", PROB_MATRIX[0].sum())  # row summing up to 1
print("PROB_MATRIX[:,0]", PROB_MATRIX[:, 0].sum())  # column not summing up to 1
print("Normalized prob. matrix of bigrams(PROB_MATRIX): ", PROB_MATRIX)
sample_dataset()

## PROBABILITIES AND LIKELIHOODS
# Probabilities of bigrams and likelihood:
log_likelihood = 0.0
n = 0
probs_log_probs = {'P': [], 'log(P)': []}
for w in words[:3]:
    chars = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chars, chars[1:]):
        index_1 = string_to_integer[ch1]
        index_2 = string_to_integer[ch2]
        probability = PROB_MATRIX[index_1, index_2]
        log_prob = torch.log(probability)

        probs_log_probs['P'].append(probability.item())
        probs_log_probs['log(P)'].append(log_prob.item())

        log_likelihood += log_prob  # Likelihood = sum of logarithms of probabilities
        print(f"{ch1}{ch2}: {probability:.4f} {log_prob:.4f}")
        n += 1

print("Log likelihood: ", log_likelihood)
negative_log_likelihood = -log_likelihood
print("Negative log likelihood: ", negative_log_likelihood)  # Because it is loss, we want to minimize it
normalized_log_likelihood = negative_log_likelihood / n
print("Normalized log likelihood: ", normalized_log_likelihood)

print("probs_log_probs:")
print(probs_log_probs)

keys = list(probs_log_probs.keys())

# Get the number of items in the lists (assuming they have the same length)
num_items = len(probs_log_probs[keys[0]])


# Create pairs of 'P' and 'log(P)' values
pairs = list(zip(probs_log_probs['P'], probs_log_probs['log(P)']))

# Sort the pairs by 'P' in descending order
sorted_pairs_by_P = sorted(pairs, key=lambda x: x[0], reverse=True)

# Print the sorted pairs
for p, log_p in sorted_pairs_by_P:
    print(round(p, 4), round(log_p, 4))

print("--------------")
sorted_pairs_by_log_P = sorted(pairs, key=lambda x: x[1], reverse=True)

# Print the sorted pairs
for p, log_p in sorted_pairs_by_P:
    print(round(p, 4), round(log_p, 4))

print('We see, that these sorts are identical. By log we are trying to "penalize" low probability values.')

## NEURAL NETWORK FOR BIGRAM BASED LANG MODEL
# We are now ready to use the NN. Things will be similar to the spelled_out_intro_to_nn_and_backprop tutorial

# NN inputs

xs, ys = [], []
print("Bigrams for the first name in the dataset")
for w in words[:1]:
    chars = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chars, chars[1:]):
        index_1 = string_to_integer[ch1]
        index_2 = string_to_integer[ch2]
        print(ch1, ch2)
        xs.append(index_1)
        ys.append(index_2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)

print("xs: ", xs)
print("ys: ", ys)

print("Often, inserting int into NN is not really good idea (int is ordinal by its nature, "
      "which is certainly not what we want in the case of int representation of strings). "
      "We need to encode the inputs.")

import torch.nn.functional as nn_functional

x_encoded = nn_functional.one_hot(xs, num_classes=ALPHABET_SIZE_WITH_STOP_SYMBOL - 1).float()
print("x_encoded:")
print(x_encoded)
print(x_encoded.shape)
plt.imshow(x_encoded)
plt.show()

W = torch.randn((27, 1))  # What happens if we use for example (27, 27), what if we set it to 1x1?
print("Example of matrix multiplication in torch:")
print(x_encoded @ W)

