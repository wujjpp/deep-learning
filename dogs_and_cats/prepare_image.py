import os
import shutil

original_dataset_dir = '/home/jp/deep-learning/dogs_and_cats/local_data/kaggle_original_data'
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


def mkdir_if_not_exists(path, remove_first=False):
    if not os.path.exists(path):
        os.mkdir(path)
    elif remove_first:
        shutil.rmtree(path)
        os.mkdir(path)


# prepare dirs
for path in [
        train_dir, train_dogs_dir, train_cats_dir, validation_dir,
        validation_dogs_dir, validation_cats_dir, test_dir, test_dogs_dir,
        test_cats_dir
]:
    mkdir_if_not_exists(path)


def batch_copy(src_dir, dst_dir, fnames):
    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        shutil.copyfile(src, dst)


fnames = ('dog.{0}.jpg'.format(i) for i in range(1000))
batch_copy(original_dataset_dir, train_dogs_dir, fnames)

fnames = ('dog.{0}.jpg'.format(i) for i in range(1000, 1500))
batch_copy(original_dataset_dir, validation_dogs_dir, fnames)

fnames = ('dog.{0}.jpg'.format(i) for i in range(1500, 2000))
batch_copy(original_dataset_dir, test_dogs_dir, fnames)

fnames = ('cat.{0}.jpg'.format(i) for i in range(1000))
batch_copy(original_dataset_dir, train_cats_dir, fnames)

fnames = ('cat.{0}.jpg'.format(i) for i in range(1000, 1500))
batch_copy(original_dataset_dir, validation_cats_dir, fnames)

fnames = ('cat.{0}.jpg'.format(i) for i in range(1500, 2000))
batch_copy(original_dataset_dir, test_cats_dir, fnames)
