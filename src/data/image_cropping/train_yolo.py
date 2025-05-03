import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback


def main():
    data_yaml_path = "coco8.yaml"
    model = YOLO('yolov8n.pt')
    # Add W&B callback for Ultralytics
    add_wandb_callback(model, enable_model_checkpointing=True)

    model.train(
        data=data_yaml_path,
        epochs=100,
        imgsz=512,
        batch=6,
        name='yolo8n',
        device=0,
        workers=3,
        save=True,
        project="roi_detection",
    )

    model.val()

if __name__ == '__main__':
    main()
