import os
import shutil
from config import base_dir, original_dataset_dir
from config import train_dir, train_dogs_dir, train_cats_dir
from config import validation_dir, validation_dogs_dir, validation_cats_dir
from config import test_dir, test_dogs_dir, test_cats_dir
from config import train_sample_size, validation_sample_size, test_sample_size


def mkdir_if_not_exists(path, remove_first=True):
    if not os.path.exists(path):
        os.mkdir(path)
    elif remove_first:
        shutil.rmtree(path)
        os.mkdir(path)


# prepare dirs
for path in [
        base_dir, train_dir, train_dogs_dir, train_cats_dir, validation_dir,
        validation_dogs_dir, validation_cats_dir, test_dir, test_dogs_dir,
        test_cats_dir
]:
    mkdir_if_not_exists(path)


def batch_copy(src_dir, dst_dir, fnames):
    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        shutil.copyfile(src, dst)


train_sample_start_index = 0
train_sample_end_index = train_sample_start_index + train_sample_size

validation_sample_start_index = train_sample_end_index
validation_sample_end_index = validation_sample_start_index + validation_sample_size

test_sample_start_index = validation_sample_end_index
test_sample_end_index = test_sample_start_index + test_sample_size

fnames = ('dog.{0}.jpg'.format(i)
          for i in range(train_sample_start_index, train_sample_end_index))
batch_copy(original_dataset_dir, train_dogs_dir, fnames)

fnames = (
    'dog.{0}.jpg'.format(i)
    for i in range(validation_sample_start_index, validation_sample_end_index))
batch_copy(original_dataset_dir, validation_dogs_dir, fnames)

fnames = ('dog.{0}.jpg'.format(i)
          for i in range(test_sample_start_index, test_sample_end_index))
batch_copy(original_dataset_dir, test_dogs_dir, fnames)

fnames = ('cat.{0}.jpg'.format(i)
          for i in range(train_sample_start_index, train_sample_end_index))
batch_copy(original_dataset_dir, train_cats_dir, fnames)

fnames = (
    'cat.{0}.jpg'.format(i)
    for i in range(validation_sample_start_index, validation_sample_end_index))
batch_copy(original_dataset_dir, validation_cats_dir, fnames)

fnames = ('cat.{0}.jpg'.format(i)
          for i in range(test_sample_start_index, test_sample_end_index))
batch_copy(original_dataset_dir, test_cats_dir, fnames)
