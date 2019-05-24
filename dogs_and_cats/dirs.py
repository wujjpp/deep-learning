import os

original_dataset_dir = '/home/jp/workspace/kaggle_data/dogs_vs_cats/train'
base_dir = '/home/jp/deep-learning/dogs_and_cats/local_data'

train_dir = os.path.join(base_dir, 'train')
train_dogs_dir = os.path.join(train_dir, 'dogs')
train_cats_dir = os.path.join(train_dir, 'cats')

validation_dir = os.path.join(base_dir, 'validation')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')

test_dir = os.path.join(base_dir, 'test')
test_dogs_dir = os.path.join(test_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
