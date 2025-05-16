import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback


def main():
    data_yaml_path = "coco8.yaml"
    model = YOLO('yolov8n.pt', task="detect")
    # Add W&B callback for Ultralytics
    add_wandb_callback(model, enable_model_checkpointing=True)

    model.train(
        data=data_yaml_path,
        epochs=70,
        imgsz=1024,
        batch=32,
        name='yolo8n',
        device=0,
        workers=3,
        save=True,
        project="roi_detection",
    )

    model.val()

if __name__ == '__main__':
    main()
