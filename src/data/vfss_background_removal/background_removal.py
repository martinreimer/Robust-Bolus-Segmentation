import os
import json
import numpy as np
import cv2
import argparse
import pathlib
import sys
import imageio as io
import tensorflow as tf
from tensorflow.keras.optimizers import (
    Adam,
    Nadam,
    RMSprop,
    SGD,
    Adadelta,
    Adagrad,
    Adamax,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from data_generation import data_augmentation_alb, data_generator_padded
import unetnah

SCRIPT_DIR = pathlib.Path(os.path.realpath(os.path.dirname(__file__)))


def train_net(data_dir, out_dir, batch_size, filter_sizes, size_img):

    (
        train_image,
        train_mask,
        val_image,
        val_mask,
        test_image,
        test_mask,
    ) = data_augmentation_alb(
        data_dir,
        batch_size,
        size_img,
        "",
    )

    final_json = {}
    for i in filter_sizes:
        result = {}
        unet.main(
            out_dir,
            batch_size,
            train_image,
            train_mask,
            val_image,
            val_mask,
            test_image,
            test_mask,
            result,
            Adam,
            "Adam",
            "",
            size_img,
            i,
        )
        final_json[f"Adam_{i}"] = result
        with open(out_dir / "result_fb.json", "w") as f:
            json.dump(final_json, f, indent=4)


def predict(model_path, data_dir, output_dir, size_img):
    data_gen_test_args = dict(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(**data_gen_test_args)

    crop_image = []
    for i in range(len(os.listdir(data_dir))):
        crop_image.append(
            tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img(
                    data_dir / f"{i}.png",
                    color_mode="grayscale",
                    target_size=size_img[0:2],
                )
            )
        )

    crop_image = np.array(crop_image)
    crop_image = test_datagen.flow(crop_image, batch_size=1, shuffle=False, seed=42)
    unet.predict_fb(model_path, crop_image, output_dir)


def crop_image(input_dir, path_crop_mask, out_dir: pathlib.Path):
    out_dir_img = out_dir / "image"
    out_dir_msk = out_dir / "mask"
    out_dir_img.mkdir(parents=True, exist_ok=True)
    out_dir_msk.mkdir(parents=True, exist_ok=True)
    for i in range(len(os.listdir(input_dir / "image"))):
        img = cv2.imread(
            (input_dir / "image" / f"{i}.png").as_posix(), cv2.IMREAD_GRAYSCALE
        )
        mask = cv2.imread(
            (input_dir / "mask" / f"{i}.png").as_posix(), cv2.IMREAD_GRAYSCALE
        )
        crop_mask = cv2.imread(
            (path_crop_mask / f"{i}.png").as_posix(), cv2.IMREAD_GRAYSCALE
        )

        cropped_img, cropped_mask, _ = cropping(crop_mask, img, mask)

        io.imsave(out_dir_img / f"{i}.png", cropped_img)
        io.imsave(out_dir_msk / f"{i}.png", cropped_mask)


def cropping(mask, img_to_crop, mask_to_crop):
    # tried parameter
    puffer = 10

    # Data is an image [0 255]
    if mask.shape == (768, 1024):
        multiplied_img = cv2.bitwise_and(img_to_crop, mask)
        crop_mask_norm = np.interp(
            multiplied_img, (multiplied_img.min(), multiplied_img.max()), (0, 1)
        )
        profile_y = np.sum(crop_mask_norm, axis=1)
        crop_y = np.where(profile_y > np.ceil(profile_y.min() + puffer), 1, 0)
        crop_y_ind = np.where(crop_y[:-1] != crop_y[1:])[0]

        profile_x = np.sum(crop_mask_norm, axis=0)
        crop_x = np.where(profile_x > np.ceil(profile_x.min() + puffer), 1, 0)
        crop_x_ind = np.where(crop_x[:-1] != crop_x[1:])[0]

        return (
            multiplied_img[
                np.min(crop_y_ind) : np.max(crop_y_ind),
                np.min(crop_x_ind) : np.max(crop_x_ind),
            ],
            mask_to_crop[
                np.min(crop_y_ind) : np.max(crop_y_ind),
                np.min(crop_x_ind) : np.max(crop_x_ind),
            ],
            (crop_y_ind[0], crop_y_ind[1], crop_x_ind[0], crop_x_ind[1]),
        )
    else:
        # Data is an tensor [0 1]
        crop_mask = mask[0]
        image = img_to_crop[0]
        mask = mask_to_crop[0]
        multiplied_img = cv2.multiply(image, crop_mask)

        crop_mask_norm = np.interp(
            multiplied_img, (multiplied_img.min(), multiplied_img.max()), (0, 1)
        )

        profile_y = np.sum(crop_mask_norm, axis=1)
        crop_y = np.where(profile_y > np.ceil(profile_y.min() + puffer), 1, 0)
        crop_y_ind = np.where(crop_y[:-1] != crop_y[1:])[0]

        profile_x = np.sum(crop_mask_norm, axis=0)
        crop_x = np.where(profile_x > np.ceil(profile_x.min() + puffer), 1, 0)
        crop_x_ind = np.where(crop_x[:-1] != crop_x[1:])[0]
        return (
            multiplied_img[
                np.min(crop_y_ind) : np.max(crop_y_ind),
                np.min(crop_x_ind) : np.max(crop_x_ind),
            ],
            mask[
                np.min(crop_y_ind) : np.max(crop_y_ind),
                np.min(crop_x_ind) : np.max(crop_x_ind),
            ],
            (crop_y_ind[0], crop_y_ind[1], crop_x_ind[0], crop_x_ind[1]),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-dir", required=True, help="Path to data directory"
    )
    default_out = pathlib.Path(SCRIPT_DIR / "output")
    parser.add_argument(
        "-o", "--out-dir", default=default_out, help="Path to output dir"
    )
    parser.add_argument("-m", "--model", help="Path to model file.")
    parser.add_argument(
        "--action",
        default="train",
        const="train",
        nargs="?",
        choices=["train", "predict"],
        help="Type of action.",
    )
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit("data_dir is not a valid path!")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.action == "train":
        out_train = pathlib.Path(out_dir / "data")
        out_train.mkdir(parents=True, exist_ok=True)
        out_model = pathlib.Path(out_dir / "model")
        out_model.mkdir(parents=True, exist_ok=True)
    elif args.action == "predict":
        if not args.model:
            sys.exit("--model is a requried argument when using --action predict")
        model_file = pathlib.Path(args.model)
        if not model_file.is_file():
            sys.exit("model file is not a valid file!")
        out_padded = pathlib.Path(out_dir / "data_padded")
        out_padded.mkdir(parents=True, exist_ok=True)
        out_prediction = pathlib.Path(out_dir / "prediction")
        out_prediction.mkdir(parents=True, exist_ok=True)
        out_cropped = pathlib.Path(out_dir / "cropped")
        out_cropped.mkdir(parents=True, exist_ok=True)

    # Variable definition
    size_img = (768, 1024, 1)
    batch_size = 8
    filter_sizes = [4, 8, 16]

    if args.action == "train":
        # Generation of train data
        # Seperation of videos in train, val and test data
        subset_videos = np.unique([i.split("_0")[0] for i in os.listdir(data_dir)])
        data_generator_padded(
            data_dir, pathlib.Path(out_train), size_img[0:2], subset_videos
        )
        # Training of model
        train_net(out_train, out_model, batch_size, filter_sizes, size_img)
    elif args.action == "predict":
        # Generation of all data to crop finally
        data_generator_padded(data_dir, out_padded, size_img[0:2], "")

        # Predicting crop mask
        predict(model_file, out_padded / "image", out_prediction, size_img)

        # Crop images for image and masks
        crop_image(out_padded, out_prediction, out_cropped)