import numpy as np
import pandas as pd

foo = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
print(foo)
print(foo.shape)
print(foo.mean(axis=0))
print(foo.std(axis=0))  # 默认ddof = 0， np的标准差默认是总体标准差

s = pd.Series([1, 2, 3, 4])
print(s.head())
print('========================')
print(s.describe())  # 默认ddof = 1, pandas的标准差默认是样本标准差

dataset_original = pd.read_csv('test-data-standard.csv')
print(dataset_original.head())

dataset = dataset_original.copy()
data_stats = dataset.describe()
print(data_stats)
data_stats = data_stats.transpose()
print(data_stats)

print('=========================================================')
npdata1 = dataset_original.copy().to_numpy()
print(npdata1)
mean1 = npdata1.mean(axis=0)
std1 = npdata1.std(axis=0)
print('mean1', mean1)
print('std1', std1)
npdata1 -= mean1
npdata1 /= std1
print(npdata1)
print('=========================================================')

print('=========================================================')
npdata2 = dataset_original.copy().to_numpy()
print(npdata2)
mean2 = npdata2.mean(axis=0)
print('mean2', mean2)
npdata2 -= mean2
std2 = npdata2.std(axis=0)
print('std2', std2)
npdata2 /= std2
print(npdata2)
print('=========================================================')

print('=========================================================')
dataset = ((dataset - data_stats['mean']) / data_stats['std'])
print(dataset.head())
print('=========================================================')