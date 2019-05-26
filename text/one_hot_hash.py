import numpy as np

samples = ['The cat sat on the mat', 'The dog ate my homework.']

dimensionality = 1000
max_length = 10

results = np.zeros((len(samples), max_length, dimensionality))

for i, sample in enumerate(samples):
    words = list(enumerate(sample.split()))[:max_length]
    for j, word in words:
        index = abs(hash(word)) % dimensionality
        print(word)
        print(i, j, index)
        results[i, j, index] = 1.

print(results.shape)
print(results)
