import os
import glob
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
        data_dir: str = '/home/data/diabetic_retinopathy_detection/'
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


def food11_reader(
        is_train: bool = True,
        data_dir: str = '/home/data/food11'):
    class_names = [
            'Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat',
            'Noodles/Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable/Fruit']
    if is_train:
        image_path = os.path.join(data_dir, 'training/*/*')
    else:
        image_path = os.path.join(data_dir, 'validation/*/*')
    image_list = glob.glob(image_path)
    label_list = [int(p.split('/')[-2]) for p in image_list]

    return image_list, label_list, class_names


def imagenette_reader(
        is_train: bool = True,
        data_dir: str = '/home/data/imagenette/imagenette2-320'):
    class_names_dict = {
        'n01440764': 'tench',
        'n02102040': 'English_springer',
        'n02979186': 'cassette_player',
        'n03000684': 'chain_saw',
        'n03028079': 'church',
        'n03394916': 'French_horn',
        'n03417042': 'garbage_truck',
        'n03425413': 'gas_pump',
        'n03445777': 'golf_ball',
        'n03888257': 'parachute'}
    if is_train:
        image_path = os.path.join(data_dir, 'train/*/*')
    else:
        image_path = os.path.join(data_dir, 'val/*/*')
    image_list = glob.glob(image_path)
    class_dirs = list(class_names_dict.keys())
    label_list = [class_dirs.index(p.split('/')[-2]) for p in image_list]

    return image_list, label_list, list(class_names_dict.values())


def main():
    print("--- Food101 ---")
    food101_train_set, _, class_names = food101_reader(True)
    food101_test_set, _, _ = food101_reader(False)
    print(len(food101_train_set))
    print(len(food101_test_set))
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

    print("--- Food11 ---")
    food11_train_set, labels, class_names = food11_reader(True)
    food11_test_set, _, _ = food11_reader(False)
    print(len(food11_train_set))
    print(len(food11_test_set))
    print('Class num: ', len(class_names))

    print("--- Imagenette ---")
    imagenette_train_set, labels, class_names = imagenette_reader(True)
    imagenette_test_set, _, _ = imagenette_reader(False)
    print(len(imagenette_train_set))
    print(len(imagenette_test_set))
    print('Class num: ', len(class_names))

    return


if __name__ == "__main__":
    main()
