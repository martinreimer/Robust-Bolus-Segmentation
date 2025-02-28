import argparse
import sys
import os
import pathlib
import numpy as np
import pandas as pd
import cv2
import imageio as io
import flammkuchen as fl
import regex as re
from tensorflow.keras.utils import image_dataset_from_directory
import albumentations as A

FOLDER_LIST = ["image", "mask"]
SCRIPT_DIR = pathlib.Path(os.path.realpath(os.path.dirname(__file__)))


def data_generator(input_dir, output_dir):
    image_dir = output_dir / "image"
    mask_dir = output_dir / "mask"

    image_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    # Seed, otherwise there always different IDs
    np.random.seed(42)
    # PNG name

    running_id = 0
    running_id_image = 0
    running_id_mask = 0

    video_frame_list = pd.DataFrame(columns=["Video", "Frame", "Type", "Image"])
    video_frame_list_renamed = pd.DataFrame(columns=["Video", "Frame", "Type", "Image"])

    input_videos = os.listdir(input_dir)
    video_names = np.unique([os.path.splitext(i)[0] for i in input_videos])

    for video in video_names:
        # Iterate over video and mask
        try:
            video_images = io.mimread(input_dir / f"{video}.mp4", memtest=False)
            file = fl.load(input_dir / f"{video}.mask")
            mask = file["mask"]
        except (FileNotFoundError, IOError):
            continue

        if len(video_images) != len(mask):
            print("NOT EQUAL:", video, len(video_images), len(mask), "NOT EQUAL:")
            continue

        for i, mask in enumerate(mask):
            mask = cv2.resize(
                np.uint8(mask.transpose()),
                (224, 224),
            )
            io.imsave(
                (output_dir / "mask" / f"{running_id_mask}.png").as_posix(),
                cv2.convertScaleAbs(mask, alpha=(255.0)),
            )

            video_frame_list_renamed.loc[running_id] = [
                video,
                i,
                "mask",
                running_id_mask,
            ]
            running_id_mask += 1
            running_id += 1

        for i, image in enumerate(video_images):
            resized_image = cv2.resize(image, (224, 224))[..., 0]
            io.imsave(
                (output_dir / "image" / f"{running_id_image}.png").as_posix(),
                resized_image,
            )

            video_frame_list_renamed.loc[running_id] = [
                video,
                i,
                "image",
                running_id_image,
            ]
            running_id_image += 1
            running_id += 1

    video_frame_list_renamed.to_csv(
        output_dir / "video_frame_list.csv",
        index=False,
        na_rep="Unknown",
    )


def pad(img, size_img):
    difference = np.subtract(size_img, img.shape)
    top_bottom = int(difference[0] / 2)
    left_right = int(difference[1] / 2)
    if (top_bottom % 2) == 0 and (left_right % 2) == 0:
        padded_img = np.pad(img, ((top_bottom,), (left_right,)), "constant")
    elif (top_bottom % 2) == 0 and (left_right % 2) != 0:
        padded_img = np.pad(img, ((top_bottom,), (left_right + 1,)), "constant")
    elif (top_bottom % 2) != 0 and (left_right % 2) == 0:
        padded_img = np.pad(img, ((top_bottom + 1,), (left_right,)), "constant")
    else:
        padded_img = np.pad(img, ((top_bottom + 1,), (left_right + 1,)), "constant")
    return padded_img


def pad_cropped(img, size_img):
    difference = np.subtract(size_img, img.shape)
    if (difference[0] % 2) == 0 and (difference[1] % 2) == 0:
        val1 = int(difference[0] / 2)
        val2 = int(difference[1] / 2)
        padded_img = np.pad(img, ((val1,), (val2,)), "constant")
    elif (difference[0] % 2) == 0 and (difference[1] % 2) != 0:
        val1 = int(difference[0] / 2)
        val2 = int((difference[1] - 1) / 2)
        padded_img = np.pad(img, ((val1,), (val2,)), "constant")
        padded_img = np.pad(padded_img, [(0, 0), (0, 1)], mode="constant")
    elif (difference[0] % 2) != 0 and (difference[1] % 2) == 0:
        val1 = int((difference[0] - 1) / 2)
        val2 = int(difference[1] / 2)
        padded_img = np.pad(img, ((val1,), (val2,)), "constant")
        padded_img = np.pad(padded_img, [(0, 1), (0, 0)], mode="constant")
    else:
        val1 = int((difference[0] - 1) / 2)
        val2 = int((difference[1] - 1) / 2)
        padded_img = np.pad(img, ((val1,), (val2,)), "constant")
        padded_img = np.pad(padded_img, [(0, 1)], mode="constant")

    return padded_img


def data_generator_padded(input_dir, output_dir, size, video_array):
    # Setting the paths to the folder according to generated Dataset
    image_dir = output_dir / "image"
    mask_dir = output_dir / "mask"
    size_img = size

    image_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    # Seed, otherwise there always different IDs
    np.random.seed(42)
    # PNG name

    running_id = 0
    running_id_image = 0
    running_id_mask = 0

    video_frame_list = pd.DataFrame(columns=["Video", "Frame", "Type", "Image"])
    video_frame_list_renamed = pd.DataFrame(columns=["Video", "Frame", "Type", "Image"])

    input_videos = os.listdir(input_dir)
    video_names = np.unique([os.path.splitext(i)[0] for i in input_videos])

    for video in video_names:
        # Iterate over video and mask and
        try:
            video_images = io.mimread(input_dir / f"{video}.mp4", memtest=False)
            file = fl.load(input_dir / f"{video}.mask")
            mask = file["mask"]
        except (FileNotFoundError, IOError):
            # print(f"Mask file for video {video} not available!")
            continue

        # Check if length of video and mask is equal
        if len(video_images) != len(mask):
            print("NOT EQUAL:", video, len(video_images), len(mask), "NOT EQUAL:")
            continue

        # Mask files transposed, checking if padding possible
        if (mask[0].transpose()).shape > size_img:
            print("FAILED SiZE: ", (mask[0].transpose()).shape, size_img, video)
            continue
        else:
            # Saving masks
            for i, mask in enumerate(mask):
                io.imsave(
                    (mask_dir / f"{running_id_mask}.png").as_posix(),
                    cv2.convertScaleAbs(
                        np.uint8(pad(mask.transpose(), size)), alpha=(255.0)
                    ),
                )
                video_frame_list_renamed.loc[running_id] = [
                    video,
                    i,
                    "mask",
                    running_id_mask,
                ]
                running_id_mask += 1
                running_id += 1

            # Saving images
            for i, image in enumerate(video_images):
                io.imsave(
                    (image_dir / f"{running_id_image}.png").as_posix(),
                    pad(image[..., 0], size),
                )

                video_frame_list_renamed.loc[running_id] = [
                    video,
                    i,
                    "image",
                    running_id_image,
                ]
                running_id_image += 1
                running_id += 1

    video_frame_list_renamed.to_csv(
        output_dir / "video_frame_list_padded.csv",
        index=False,
        na_rep="Unknown",
    )


def data_generator_cropped(input_dir, output_dir, size_img, resizing=True):
    video_frames_cropped_pad = pd.DataFrame(columns=["Source_Frame", "Dest_Frame"])
    running_id = 0

    for i in range(len(os.listdir(input_dir / "image"))):
        image = io.imread(input_dir / "image" / f"{i}.png")
        mask = io.imread(input_dir / "mask" / f"{i}.png")
        if image.shape[0] > size_img[0] or image.shape[1] > size_img[1]:
            continue
        else:
            pad_crop_img = pad_cropped(image, size_img)
            pad_crop_mask = pad_cropped(mask, size_img)
            if pad_crop_img.shape != size_img:
                print(input_dir / "image" / f"{i}.png", pad_crop_img.shape)
            if resizing:
                io.imsave(
                    (output_dir / "image" / f"{running_id}.png").as_posix(),
                    cv2.resize(pad_crop_img, (256, 256)),
                )
                io.imsave(
                    (output_dir / "mask" / f"{running_id}.png").as_posix(),
                    cv2.resize(pad_crop_mask, (256, 256)),
                )
            else:
                io.imsave(
                    (output_dir / "image" / f"{running_id}.png").as_posix(),
                    pad_crop_img,
                )
                io.imsave(
                    (output_dir / "mask" / f"{running_id}.png").as_posix(),
                    pad_crop_mask,
                )

        video_frames_cropped_pad.loc[running_id] = [
            input_dir / "image" / f"{i}.png",
            f"{running_id}.png",
        ]
        running_id += 1

    video_frames_cropped_pad.to_csv(
        output_dir / "video_frame_list_crop_pad.csv",
        index=False,
        na_rep="Unknown",
    )


def data_generator_fb(input_dir, output_dir):
    # video = input_dir / "NSy12_001.mp4"
    # Setting the paths to the folder according to generated Dataset

    # Seed, otherwise there always different IDs
    np.random.seed(42)
    # PNG name

    running_id = 0
    running_id_image = 0
    running_id_mask = 0

    video_frame_list = pd.DataFrame(columns=["Video", "Frame", "Type", "Image"])
    video_frame_list_renamed = pd.DataFrame(columns=["Video", "Frame", "Type", "Image"])

    input_videos = os.listdir(input_dir)
    video_names = np.unique([os.path.splitext(i)[0] for i in input_videos])

    for video in video_names:
        # Iterate over video and mask
        try:
            video_images = io.mimread(input_dir / f"{video}.mp4", memtest=False)
            file = fl.load(input_dir / f"{video}.mask")
            mask = file["mask"]
        except (FileNotFoundError, IOError):
            # print("Mask file not available!")
            continue

        if len(video_images) != len(mask):
            print("NOT EQUAL:", video, len(video_images), len(mask), "NOT EQUAL:")
            continue

        shape_img = (video_images[0].shape)[0:2]
        chars_to_rmv = "[(, )]"
        folder_name = re.sub(chars_to_rmv, "", str(shape_img))
        image_dir = output_dir / folder_name / "raw_image"
        mask_dir = output_dir / folder_name / "raw_mask"

        image_dir.mkdir(exist_ok=True)
        mask_dir.mkdir(exist_ok=True)

        for i, mask in enumerate(mask):
            io.imsave(
                (
                    output_dir
                    / f"{folder_name}"
                    / "raw_mask"
                    / f"{running_id_mask}.png"
                ).as_posix(),
                cv2.convertScaleAbs(np.uint8(mask.transpose()), alpha=(255.0)),
            )

            video_frame_list_renamed.loc[running_id] = [
                video,
                i,
                "mask",
                running_id_mask,
            ]
            running_id_mask += 1
            running_id += 1

        for i, image in enumerate(video_images):
            io.imsave(
                (
                    output_dir
                    / f"{folder_name}"
                    / "raw_image"
                    / f"{running_id_image}.png"
                ).as_posix(),
                image[..., 0],
            )

            video_frame_list_renamed.loc[running_id] = [
                video,
                i,
                "image",
                running_id_image,
            ]
            running_id_image += 1
            running_id += 1

    video_frame_list_renamed.to_csv(
        output_dir / "video_frame_list.csv",
        index=False,
        na_rep="Unknown",
    )


def data_augmentation_alb(output_dir, bs, size_img, padded):
    seed = 42
    number_of_images = len(os.listdir(output_dir / "image"))
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(exist_ok=True)

    # List for appending the masks and images
    input_mask = []
    input_image = []
    ind_mask = []
    ind_img = []

    transform = A.Compose(
        [A.HorizontalFlip(p=0.5), A.RandomBrightnessContrast(p=0.2), A.Rotate(limit=10)]
    )

    # Loop over all images in order to keep mask and image same
    mask_ds = image_dataset_from_directory(
        output_dir / "mask",
        labels=None,
        batch_size=1,
        color_mode="grayscale",
        shuffle=False,
        image_size=size_img[0:2],
        seed=42,
    )
    image_ds = image_dataset_from_directory(
        output_dir / "image",
        labels=None,
        batch_size=1,
        color_mode="grayscale",
        shuffle=False,
        image_size=size_img[0:2],
        seed=42,
    )

    for i in mask_ds:
        input_mask.append(np.array(i[0] / 255))
    for i in image_ds:
        input_image.append(np.array(i[0] / 255))

    mask_input = np.array(input_mask)
    image_input = np.array(input_image)

    indices = np.arange(len(image_input))
    rand = indices
    np.random.shuffle(rand)

    # Shuffle images and masks in same order
    mask_input = mask_input[rand]
    image_input = image_input[rand]

    # Defining split
    split = int(0.2 * number_of_images)

    # Splitting in train and validation set according to split number
    test_image = image_input[0:split]
    test_mask = mask_input[0:split]
    val_image = image_input[split : 2 * split]
    val_mask = mask_input[split : 2 * split]
    train_image = image_input[2 * split :]
    train_mask = mask_input[2 * split :]
    np.savetxt(
        csv_dir / f"test_image{padded}.csv", ind_img[0:split], delimiter=",", fmt="%s"
    )
    np.savetxt(
        csv_dir / f"test_mask{padded}.csv", ind_mask[0:split], delimiter=",", fmt="%s"
    )
    np.savetxt(
        csv_dir / f"val_image{padded}.csv",
        ind_img[split : 2 * split],
        delimiter=",",
        fmt="%s",
    )
    np.savetxt(
        csv_dir / f"val_mask{padded}.csv",
        ind_mask[split : 2 * split],
        delimiter=",",
        fmt="%s",
    )
    np.savetxt(
        csv_dir / f"train_image{padded}.csv",
        ind_img[2 * split :],
        delimiter=",",
        fmt="%s",
    )
    np.savetxt(
        csv_dir / f"train_mask{padded}.csv",
        ind_mask[2 * split :],
        delimiter=",",
        fmt="%s",
    )

    x = []
    x1 = []
    for i in range(len(train_image)):
        transformed = transform(image=train_image[i], mask=train_mask[i])
        x.append(np.array(transformed["image"]))
        x1.append(np.array(transformed["mask"]))

    train_i = np.array(x)
    train_m = np.array(x1)

    return train_i, train_m, val_image, val_mask, test_image, test_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--data-dir", required=True, help="Path to data directory"
    )
    default_out = pathlib.Path(SCRIPT_DIR / "output")
    parser.add_argument(
        "-o", "--out-dir", default=default_out, help="Path to output dir"
    )
    parser.add_argument(
        "-t",
        "--target",
        default="cropped",
        const="cropped",
        nargs="?",
        choices=["cropped", "resized"],
        help="Target type of data.",
    )
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit("data_dir is not a valid path!")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generating train data
    if args.target == "resized":
        data_generator(data_dir, out_dir)
    elif args.target == "cropped":
        data_generator_cropped(data_dir, out_dir, (512, 512))