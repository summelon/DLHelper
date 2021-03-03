import os
import itertools
import pandas as pd
import numpy as np
from scipy.io import loadmat


def food101_reader(
        is_train: bool = True,
        data_dir: str = '/home/data/food-101'
        ):
    meta_path = os.path.join(data_dir, 'meta')
    image_path = os.path.join(data_dir, 'images')

    class_names = pd.read_csv(
        os.path.join(meta_path, 'classes.txt'), header=None).values.flatten()
    class_names = list(class_names)

    if is_train:
        dataframe = pd.read_json(os.path.join(meta_path, 'train.json'))
    else:
        dataframe = pd.read_json(os.path.join(meta_path, 'test.json'))

    label_list = [class_names.index(p.split('/')[0])
                  for p in dataframe.values.flatten()]
    image_list = dataframe\
        .applymap(lambda x: os.path.join(image_path, x) + '.jpg')\
        .values.flatten()

    return image_list, label_list, class_names


def stanford_dogs_reader(
        is_train: bool = True,
        data_dir: str = '/home/data/stanford_dogs'
        ):
    image_path = os.path.join(data_dir, 'Images')

    if is_train:
        mat_file = loadmat(os.path.join(data_dir, 'train_list.mat'))
    else:
        mat_file = loadmat(os.path.join(data_dir, 'test_list.mat'))

    image_list = [os.path.join(image_path, f) for f
                  in itertools.chain(*mat_file['file_list'].flatten())]
    class_names = sorted(set([f.split('/')[-2] for f in image_list]))
    label_list = [class_names.index(p.split('/')[-2]) for p in image_list]

    return image_list, label_list, class_names


def caltech101_reader(
        is_train: bool = True,
        data_dir: str = '/home/data/caltech101'
        ):
    image_path = os.path.join(data_dir, '101_ObjectCategories')
    np.random.seed(1234)
    _TRAIN_POINTS_PER_CLASS = 30

    walker = os.walk(image_path)
    train_list, test_list = list(), list()

    _, class_names, _ = next(walker)
    for root, dirs, files in walker:
        train_sublist = np.random.choice(
                files, _TRAIN_POINTS_PER_CLASS, replace=False)
        if is_train:
            train_list += [os.path.join(root, f) for f in train_sublist]
        else:
            test_sublist = set(files).difference(train_sublist)
            test_list += [os.path.join(root, f) for f in test_sublist]
    class_names = sorted(set(class_names))
    if is_train:
        image_list = train_list
    else:
        image_list = test_list
    label_list = [class_names.index(p.split('/')[-2]) for p in image_list]

    return image_list, label_list, class_names


def diabetic_reader(
        is_train: bool = True,
        data_dir: str = '/home/data/diabetic/'
        ):
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    if is_train:
        csv_path = os.path.join(data_dir, 'trainLabels.csv')
        img_root_path = os.path.join(data_dir, 'train')
    else:
        csv_path = os.path.join(data_dir, 'retinopathy_solution.csv')
        img_root_path = os.path.join(data_dir, 'test')

    dataframe = pd.read_csv(csv_path)
    image_list = [os.path.join(img_root_path, img_name+'.jpeg')
                  for img_name in dataframe.image.to_list()]
    label_list = dataframe.level.to_list()

    return image_list, label_list, class_names


def main():
    print("--- Food101 ---")
    food_train_set, _, class_names = food101_reader(True)
    food_test_set, _, _ = food101_reader(False)
    print(len(food_train_set))
    print(len(food_test_set))
    print('Class num: ', len(class_names))

    print("--- Stanford dogs ---")
    dog_train_set, _, class_names = stanford_dogs_reader(True)
    dog_test_set, _, _ = stanford_dogs_reader(False)
    print(len(dog_train_set))
    print(len(dog_test_set))
    print('Class num: ', len(class_names))

    print("--- Caltech101 ---")
    caltech_train_set, _, class_names = caltech101_reader(True)
    caltech_test_set, _, _ = caltech101_reader(False)
    print(len(caltech_train_set))
    print(len(caltech_test_set))
    print('Class num: ', len(class_names))

    print("--- Diabetic Retinopathy Detection---")
    diabetic_train_set, _, class_names = diabetic_reader(True)
    diabetic_test_set, _, _ = diabetic_reader(False)
    print(len(diabetic_train_set))
    print(len(diabetic_test_set))
    print('Class num: ', len(class_names))


if __name__ == "__main__":
    main()
