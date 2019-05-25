import os

original_dataset_dir = '/home/jp/workspace/kaggle_data/dogs_vs_cats/train'
base_dir = '/home/jp/deep-learning/dogs_vs_cats/local_data'

train_dir = os.path.join(base_dir, 'train')
train_dogs_dir = os.path.join(train_dir, 'dogs')
train_cats_dir = os.path.join(train_dir, 'cats')

validation_dir = os.path.join(base_dir, 'validation')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')

test_dir = os.path.join(base_dir, 'test')
test_dogs_dir = os.path.join(test_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')

# small
train_sample_size = 1000
validation_sample_size = 500
test_sample_size = 500

# full
# train_sample_size = 10000
# validation_sample_size = 2000
# test_sample_size = 500

# 评估程序
# train_sample_size = 80
# validation_sample_size = 40
# test_sample_size = 40