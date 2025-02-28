import os
import argparse
import sys
import pathlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio as io
import json
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPool2D,
    UpSampling2D,
    Concatenate,
    BatchNormalization,
    Activation,
)
from tensorflow.keras.optimizers import (
    Adam,
    Nadam,
    RMSprop,
    SGD,
    Adadelta,
    Adagrad,
    Adamax,
)
from tensorflow.keras.metrics import (
    Accuracy,
    Recall,
    Precision,
    MeanSquaredError,
    MeanIoU,
)
from lr_finder import LRFind
from data_generation import data_augmentation_alb
from tensorflow.keras.models import load_model

SCRIPT_DIR = pathlib.Path(os.path.realpath(os.path.dirname(__file__)))


def conv_layer(x, filters, activation, name, batch_norm=True):
    x = Conv2D(filters, 3, padding="same", name=name)(x)
    if batch_norm:
        # x = BatchNormalization(name=name + "_BN")(x)
        x = tfa.layers.InstanceNormalization(name=name + "_IN")(x)
        # x = tfa.layers.FilterResponseNormalization(name=name + "_FRN")(x)
    x = Activation(activation, name=name + "_activation")(x)
    return x


def unet(filters, layers, activation, n_classes, size_img):
    skip_con = []
    model_input = Input(size_img)
    x = model_input
    # Downsampling
    for i in range(layers):
        x = conv_layer(x, filters * 2 ** i, activation, f"enc_layer{i}_conv1")
        x = conv_layer(x, filters * 2 ** i, activation, f"enc_layer{i}_conv2")
        # Saving last convolution for skip connection
        skip_con.append(x)
        x = MaxPool2D()(x)

    x = conv_layer(x, filters * 2 ** (i + 1), activation, "latent_conv")

    # Upsampling
    for i in range(layers):
        x = UpSampling2D()(x)
        # Connecting the saved skip connection, accessing the array as LIFO
        x = Concatenate(name=f"skip_{i}")([x, skip_con.pop()])
        x = conv_layer(
            x, filters * 2 ** (layers - i - 1), activation, f"dec_layer{layers-i}_conv1"
        )
        x = conv_layer(
            x, filters * 2 ** (layers - i - 1), activation, f"dec_layer{layers-i}_conv2"
        )

    # Point convolution with the number of classes and kernel_size
    output_layer = Conv2D(
        n_classes, kernel_size=1, activation="sigmoid", padding="same", name="final"
    )(x)

    model = Model(model_input, output_layer)
    return model


# Definition of Dice Coefficient and two versions of dice_loss definitions --> first choice
def dice_coefficient(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return numerator / (denominator + tf.keras.backend.epsilon())


def dice_coef_loss(y_true, y_pred):
    # return tf.keras.losses.binary_crossentropy(y_true, y_pred) - tf.math.log(dice_coefficient(y_true, y_pred) + tf.keras.backend.epsilon())
    return 1.0 - dice_coefficient(y_true, y_pred) + tf.keras.backend.epsilon()


def IoU(y_pred, y_true):
    I = tf.reduce_sum(y_pred * y_true, axis=(1, 2))
    U = tf.reduce_sum(y_pred + y_true, axis=(1, 2)) - I
    return tf.reduce_mean(I / U)


def lr_finder(model, train_image, train_mask):
    EPOCHS = 1
    BATCH_SIZE = 64
    lr_finder_steps = 400
    lr_find = LRFind(1e-6, 1e1, lr_finder_steps)
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_image, train_mask))
        .repeat()
        .shuffle(2048)
        .batch(BATCH_SIZE)
    )
    model.fit(
        train_dataset,
        steps_per_epoch=lr_finder_steps,
        epochs=EPOCHS,
        callbacks=[lr_find],
    )

    plt.plot(lr_find.lrs, lr_find.losses)
    print(lr_find.lrs, type(lr_find.lrs))
    np.savetxt("plt_lr_lrs.csv", lr_find.lrs, delimiter=",", fmt="%10.5f")
    np.savetxt("plt_lr_losses.csv", lr_find.losses, delimiter=",", fmt="%10.5f")
    plt.xscale("log")
    plt.show()


def main(
    out_dir: pathlib.Path,
    bs,
    train_image,
    train_mask,
    val_image,
    val_mask,
    test_image,
    test_mask,
    result,
    opt,
    name_opt,
    padded,
    size_img,
    filter_size,
):

    filter_base = filter_size
    layer = 4
    activation = "relu"
    n_classes = 1
    epochs = 1
    base_learning_rate = 10e-3
    result["epochs"] = epochs
    step_size = len(train_image) // bs
    val_step_size = len(val_image) // bs
    test_step_size = len(test_image) // bs

    model = unet(filter_base, layer, activation, n_classes, size_img)

    # # Code for CLR
    # model_clr = unet(filter_base, layer, activation, n_classes, size_img)
    # model.compile(loss=dice_coef_loss, optimizer=SGD(momentum=0.9), metrics=["accuracy"])
    #
    # # Detect upper lower bound for CLR
    # lr_finder(model, train_image, train_mask)
    #
    # # From lr-finder visually detect INIT_LR and MAX_LR
    # INIT_LR = 1e-4
    # MAX_LR = 1e-1
    #
    # # Define CLR
    # clr = tfa.optimizers.Triangular2CyclicalLearningRate(
    #    initial_learning_rate=INIT_LR,
    #    maximal_learning_rate=MAX_LR,
    #    step_size= 4 * step_size,
    # )
    #
    # # Visualization of CLR profile
    # step = np.arange(0, epochs * step_size)
    # lr = clr(step)
    # plt.plot(step, lr)
    # plt.xlabel("Steps")
    # plt.ylabel("Learning Rate")
    # plt.savefig(file_dir / f"{name_opt}_{filter_size}_{epochs}_clr.png")
    # result["learning_rate"] = "cyclic"

    result["learning_rate"] = base_learning_rate
    result["filter_size"] = filter_base
    result["loss"] = "dice_coef_loss"

    defined_metrics = [
        dice_coefficient,
        "accuracy",
        "mean_squared_error",
        IoU,
    ]

    # # CLR model
    # model_clr.compile(
    #     optimizer=opt(learning_rate=clr),
    #     loss=dice_coef_loss,
    #     metrics=defined_metrics,
    # )

    model.compile(
        optimizer=opt(learning_rate=base_learning_rate),
        loss=dice_coef_loss,
        metrics=defined_metrics,
    )

    callbacks = [
        ModelCheckpoint(
            str(out_dir / f"model_{name_opt}{padded}_{filter_size}.h5"),
            save_best_only=True,
        ),
        CSVLogger(
            str(out_dir / f"model_{name_opt}{padded}_{filter_size}.log"),
            separator=";",
            append=False,
        ),
    ]

    train_gen = (
        tf.data.Dataset.from_tensor_slices((train_image, train_mask))
        .shuffle(10000, reshuffle_each_iteration=True)
        .batch(bs, drop_remainder=False)
        .repeat(epochs)
    )
    val_gen = (
        tf.data.Dataset.from_tensor_slices((val_image, val_mask))
        .shuffle(10000, reshuffle_each_iteration=True)
        .repeat(epochs)
        .batch(bs, drop_remainder=False)
    )
    test_gen = (
        tf.data.Dataset.from_tensor_slices((test_image, test_mask))
        .shuffle(10000, reshuffle_each_iteration=True)
        .repeat(epochs)
        .batch(bs, drop_remainder=False)
    )

    # # CLR model train
    # history_model_clr = model_clr.fit(
    #     train_gen,
    #     batch_size=bs,
    #     validation_data=val_gen,
    #     validation_steps=val_step_size,
    #     steps_per_epoch=step_size,
    #     epochs=epochs,
    #     callbacks=callbacks,
    # )
    # out_file = f"{name_opt}_{filter_size}_{epochs}_clr.json"
    # json.dump(history_model_clr.history, str(out_dir / out_file))
    #
    # loss_clr, dc_clr, acc_clr, mse_clr, iou_clr = model_clr.evaluate(
    #     test_gen, steps=test_step_size, batch_size=bs
    # )
    # # Saving parameter
    # result["val_loss_clr"] = np.min(history_model_clr.history["val_loss"])
    # result["val_dc_clr"] = np.max(history_model_clr.history["val_dice_coefficient"])
    # result["val_iou_clr"] = np.max(history_model_clr.history["val_IoU"])
    # result["val_mse_clr"] = np.min(history_model_clr.history["val_mean_squared_error"])
    # result["loss_clr"] = np.min(history_model_clr.history["loss"])
    # result["dc_clr"] = np.max(history_model_clr.history["dice_coefficient"])
    # result["iou_clr"] = np.max(history_model_clr.history["IoU"])
    # result["mse_clr"] = np.min(history_model_clr.history["mean_squared_error"])
    # result["test_loss_clr"] = loss_clr
    # result["test_dc_clr"] = dc_clr
    # result["test_acc_clr"] = acc_clr
    # result["test_mse_clr"] = mse_clr
    # result["test_iou_clr"] = iou_clr

    history_model = model.fit(
        train_gen,
        batch_size=bs,
        validation_data=val_gen,
        validation_steps=val_step_size,
        steps_per_epoch=step_size,
        epochs=epochs,
        callbacks=callbacks,
    )
    out_file = f"model_{name_opt}_{filter_size}_{epochs}_normal.json"
    with open(out_dir / out_file, "w", encoding="utf-8") as f:
        json.dump(history_model.history, f)

    loss, dc, acc, mse, iou = model.evaluate(
        test_gen, steps=test_step_size, batch_size=bs
    )

    result["loss"] = np.min(history_model.history["loss"])
    result["dc"] = np.max(history_model.history["dice_coefficient"])
    result["iou"] = np.max(history_model.history["IoU"])
    result["mse"] = np.min(history_model.history["mean_squared_error"])
    result["val_loss"] = np.min(history_model.history["val_loss"])
    result["val_dc"] = np.max(history_model.history["val_dice_coefficient"])
    result["val_iou"] = np.max(history_model.history["val_IoU"])
    result["val_mse"] = np.min(history_model.history["val_mean_squared_error"])
    result["test_loss"] = loss
    result["test_dc"] = dc
    result["test_acc"] = acc
    result["test_mse"] = mse
    result["test_iou"] = iou
    return result


def countNonZero(arr):
    return (arr == 1).sum()


def predict(out_dir, test_image, test_mask, padded, model_path):
    custom_objects = {
        "dice_coef_loss": dice_coef_loss,
        "dice_coefficient": dice_coefficient,
        "IoU": IoU,
    }
    gt_area = []
    pred_area = []
    intersection = []
    union = []
    iou = []

    if padded == "_pad":
        model = load_model(model_path, custom_objects)

        result = model.predict(test_image, test_mask)
        for i, val in enumerate(result):
            io.imsave(out_dir / f"{i}.png", val.squeeze().astype("uint8"))

            figure, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(val.squeeze(), cmap="gray")
            ax1.imshow(
                test_mask[i][0],
                cmap=matplotlib.colors.ListedColormap(["none", "green"]),
                alpha=0.4,
            )
            ax1.set_title("Prediction")
            ax2.imshow(test_image[i][0], cmap="gray")
            ax2.set_title("Original")
            plt.tight_layout()
            plt.savefig(out_dir / f"prediction_{i}.png")
    else:
        model = load_model(model_path, custom_objects)
        figure, (ax1, ax2) = plt.subplots(1, 2)
        result = model.predict(test_image, test_mask)
        for i, val in enumerate(result):
            io.imsave(out_dir / f"{i}.png", (val * 255).squeeze().astype("uint8"))

            # Measurements
            test_m = np.where(test_mask[i][0].squeeze() > 0.1, 1, 0)
            pred_mask = np.where((val * 255).squeeze().astype("uint8") > 0.1, 1, 0)
            gt_area.append(countNonZero(test_m))
            pred_area.append(countNonZero(pred_mask))
            intersection.append(countNonZero(pred_mask * test_m))
            union.append(countNonZero((np.logical_or(pred_mask, test_m)) * 1))
            if i < 2:
                np.savetxt(out_dir / "test_m.csv", test_m, fmt="%d", delimiter=",")
                np.savetxt(
                    out_dir / "pred_mask.csv", pred_mask, fmt="%d", delimiter=","
                )
                print(np.min(pred_mask), np.max(pred_mask), countNonZero(pred_mask))
                print(np.min(test_m), np.max(test_m), countNonZero(test_m))
                print("inter: ", countNonZero(pred_mask * test_m))
                print("union: ", countNonZero(np.logical_or(pred_mask, test_m) * 1))
        iou = np.array(intersection) / np.array(union)


def predict_fb(model_path, test_image, out_dir: pathlib.Path):
    custom_objects = {
        "dice_coef_loss": dice_coef_loss,
        "dice_coefficient": dice_coefficient,
        "IoU": IoU,
    }

    model = load_model(
        model_path,
        custom_objects,
    )

    idx_pred = 0
    for i in range(len(test_image)):
        pred_mask = model.predict(test_image[i])
        if np.max(pred_mask <= 1):
            io.imsave(
                out_dir / f"{idx_pred}.png",
                (pred_mask.squeeze() * 255).astype(np.uint8),
            )
        else:
            io.imsave(
                out_dir / f"{idx_pred}.png",
                (pred_mask.squeeze()).astype(np.uint8),
            )
        idx_pred += 1


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
    args = parser.parse_args()

    data_dir = pathlib.Path(args.data_dir)
    if not data_dir.is_dir():
        sys.exit("data_dir is not a valid path!")
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    size_img = (224, 224, 1)
    size_img_cropped = (256, 256, 1)
    filter = [4]  # [4, 8, 16, 32, 48, 64]
    final_json = {}
    batch_size = 32
    batch_size_crop = 16

    # # Using the raw resized to 224x224 data
    # (
    #     train_image,
    #     train_mask,
    #     val_image,
    #     val_mask,
    #     test_image,
    #     test_mask,
    # ) = data_augmentation_alb(out_data, batch_size, size_img, "")
    #
    # for i in filter:
    #     result = {"cropped":False}
    #     main(
    #         out_dir,
    #         batch_size,
    #         train_image,
    #         train_mask,
    #         val_image,
    #         val_mask,
    #         test_image,
    #         test_mask,
    #         result,
    #         Adam,
    #         "Adam",
    #         "",
    #         size_img,
    #         i,
    #     )
    #     final_json[f"Adam_{i}"] = result
    #     with open(out_dir / "result.json", "w", encoding="utf-8") as f:
    #         json.dump(final_json, f, indent=4)

    # # Using the cropped resized to 256x256 data
    (
        train_image_crop,
        train_mask_crop,
        val_image_crop,
        val_mask_crop,
        test_image_crop,
        test_mask_crop,
    ) = data_augmentation_alb(data_dir, batch_size_crop, size_img_cropped, "_padded")

    for i in filter:
        result = {"cropped": True}
        main(
            out_dir,
            batch_size_crop,
            train_image_crop,
            train_mask_crop,
            val_image_crop,
            val_mask_crop,
            test_image_crop,
            test_mask_crop,
            result,
            Adam,
            "Adam",
            "_cropped",
            size_img_cropped,
            i,
        )
        final_json[f"Adam_{i}"] = result
        with open(out_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(final_json, f, indent=4)