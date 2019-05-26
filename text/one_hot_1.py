import numpy as np

samples = ['The cat sat on the mat', 'The dog ate my homework.']

token_index = {}
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

max_length = 15
results = np.zeros(shape=(len(samples), max_length,
                          max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    words = list(enumerate(sample.split()))[:max_length]
    for j, word in words:
        index = token_index.get(word)
        results[i, j, index] = 1

print(token_index)
print(results)
print(results.shape)
