import os
import glob
import itertools
import pandas as pd
import numpy as np
from scipy.io import loadmat


imagenette_names = {
    # Imagenette
    'n01440764': 'tench',
    'n02102040': 'English_springer',
    'n02979186': 'cassette_player',
    'n03000684': 'chain_saw',
    'n03028079': 'church',
    'n03394916': 'French_horn',
    'n03417042': 'garbage_truck',
    'n03425413': 'gas_pump',
    'n03445777': 'golf_ball',
    'n03888257': 'parachute',
}

imagewoof_names = {
    # Imagewoof
    "n02093754": "Australian terrier",
    "n02089973": "Border terrier",
    "n02099601": "Samoyed",
    "n02087394": "Beagle",
    "n02105641": "Shih-Tzu",
    "n02096294": "English foxhound",
    "n02088364": "Rhodesian ridgeback",
    "n02115641": "Dingo",
    "n02111889": "Golden retriever",
    "n02086240": "Old English sheepdog",
}


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


def _diabetic_reader(is_train: bool, data_dir: str):
    class_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    if is_train:
        csv_path = os.path.join(data_dir, 'trainLabels.csv')
        img_root_path = os.path.join(data_dir, 'train')
    else:
        csv_path = os.path.join(data_dir, 'retinopathy_solution.csv')
        img_root_path = os.path.join(data_dir, 'test')

    dataframe = pd.read_csv(csv_path)
    image_list = [os.path.join(img_root_path, img_name+'.jpg')
                  for img_name in dataframe.image.to_list()]
    label_list = dataframe.level.to_list()

    return image_list, label_list, class_names


def diabetic_250k_reader(
        is_train: bool = True,
        data_dir: str = '/home/data/diabetic_retinopathy_detection_250K'
        ):
    return _diabetic_reader(
        is_train=is_train,
        data_dir=data_dir
    )


def diabetic_btgraham_reader(
        is_train: bool = True,
        data_dir: str = '/home/data/diabetic_retinopathy_detection_btgraham300'
        ):
    return _diabetic_reader(
        is_train=is_train,
        data_dir=data_dir
    )


def food11_reader(
        is_train: bool = True,
        data_dir: str = '/home/data/food11'
        ):
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


def _imagenet_reader(class_names_dict: dict, is_train: bool, data_dir: str):
    if is_train:
        image_path = os.path.join(data_dir, 'train/*/*')
    else:
        image_path = os.path.join(data_dir, 'val/*/*')
    image_list = glob.glob(image_path)
    class_dirs = list(class_names_dict.keys())
    label_list = [class_dirs.index(p.split('/')[-2]) for p in image_list]

    return image_list, label_list, list(class_names_dict.values())


def imagenette_reader(
        is_train: bool = True,
        data_dir: str = '/home/data/imagenette'
        ):
    return _imagenet_reader(
        class_names_dict=imagenette_names,
        is_train=is_train,
        data_dir=data_dir,
    )


def imagewoof_reader(
        is_train: bool = True,
        data_dir: str = '/home/data/imagewoof'
        ):
    return _imagenet_reader(
        class_names_dict=imagewoof_names,
        is_train=is_train,
        data_dir=data_dir,
    )


def imagewang_reader(
        is_train: bool = True,
        data_dir: str = '/home/data/imagewang'
        ):
    names = imagewoof_names
    if is_train:
        names.update(imagenette_names)

    return _imagenet_reader(
        class_names_dict=names,
        is_train=is_train,
        data_dir=data_dir,
    )


def _show_fn(name: str, reader_fn: callable):
    print(f"--- {name} ---")
    train_set, _, class_names = reader_fn(True)
    test_set, _, _ = reader_fn(False)
    print(len(train_set))
    print(len(test_set))
    print('Class num: ', len(class_names))

    return


def main():
    _show_fn("Food101", food101_reader)
    _show_fn("Food11", food11_reader)
    _show_fn("Stanford dogs", stanford_dogs_reader)
    _show_fn("Caltech101", caltech101_reader)
    _show_fn("Imagenette", imagenette_reader)
    _show_fn("Imagewoof", imagewoof_reader)
    _show_fn("Imagewang", imagewang_reader)
    _show_fn(
        "Diabetic Retinopathy Detection(250K)",
        diabetic_250k_reader,
    )
    _show_fn(
        "Diabetic Retinopathy Detection(btgraham-300)",
        diabetic_btgraham_reader,
    )

    return


if __name__ == "__main__":
    main()
