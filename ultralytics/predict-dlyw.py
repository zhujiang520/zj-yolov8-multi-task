import sys
sys.path.insert(0, "/data/zhujiang/code/YOLOv8-multi-task-main/ultralytics")

from ultralytics import YOLO


number = 2 #input how many tasks in your work
model = YOLO('/data/zhujiang/code/YOLOv8-multi-task-main/runs/multi/yolopm5-dlyw-20260330-640/weights/best.pt')  # Validate the model
model.predict(
    source='/data/zhujiang/code/YOLOv8-multi-task-main/dataset/dlyw/有障碍物的图片',
    data='/data/zhujiang/code/YOLOv8-multi-task-main/ultralytics/datasets/bdd-multi-dlyw.yaml',
    imgsz=(384, 672),
    device=[3],
    name='v4_daytime',
    save=True,
    conf=0.25,
    iou=0.45,
    show_labels=False,
)
