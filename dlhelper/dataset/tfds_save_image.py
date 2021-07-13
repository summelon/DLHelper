import os
import tqdm
import shutil
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds


def main():
    dataset_name = "diabetic_retinopathy_detection"
    root_path = "/home/project/ssl/" + dataset_name
    dataset_path = "/home/data/"

    if os.path.exists(root_path):
        shutil.rmtree(root_path)
    os.mkdir(root_path)

    augmentaiton = ["250K", "btgraham-300"]
    for aug in augmentaiton:
        save_path = os.path.join(root_path, aug)
        os.mkdir(save_path)
        ds = tfds.load(
            os.path.join(dataset_name, aug),
            data_dir=dataset_path,
            shuffle_files=False,
        )
        for subset in ["train", "validation", "test"]:
            subset_path = os.path.join(save_path, subset)
            os.mkdir(subset_path)
            loader = ds[subset].prefetch(tf.data.experimental.AUTOTUNE)
            pbar = tqdm.tqdm(loader)
            for data in pbar:
                name = data["name"].numpy().decode("UTF-8")
                image = data["image"].numpy()
                image = Image.fromarray(image.astype("uint8"))
                image_path = os.path.join(subset_path, name+".jpg")
                image.save(image_path)
    return


if __name__ == "__main__":
    main()
