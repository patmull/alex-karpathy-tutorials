from matplotlib import pyplot as plt

words = open("spelled_out_intro_to_language_modeling/names.txt", 'r').read().splitlines()

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

ALPHABET_SIZE = 28
## Basic approach of mapping chars to tensors
N = torch.zeros((ALPHABET_SIZE + 1, ALPHABET_SIZE + 1))  # +1 for the start/stop sign
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
PROB_MATRIX = N.float()
PROB_MATRIX = PROB_MATRIX / PROB_MATRIX.sum(1, keepdim=True)

print("PROB_MATRIX[0].sum() ", PROB_MATRIX[0].sum())  # row summing up to 1
print("PROB_MATRIX[:,0]", PROB_MATRIX[:, 0].sum())  # column not summing up to 1
print("Normalized prob. matrix of bigrams(PROB_MATRIX): ", PROB_MATRIX)
sample_dataset()
