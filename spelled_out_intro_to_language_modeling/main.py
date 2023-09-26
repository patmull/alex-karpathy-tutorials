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
N = torch.zeros((ALPHABET_SIZE+1, ALPHABET_SIZE+1))  # +1 for the start/stop sign
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

plt.figure(figsize=(16,16))
plt.imshow(N, cmap='Blues')
for i in range(26):
    for j in range(26):
        chstr = characters_to_int[i] + characters_to_int[j]
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')
plt.show()

