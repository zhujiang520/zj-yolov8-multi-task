import sys
sys.path.insert(0, "/data/zhujiang/code/YOLOv8-multi-task-main/ultralytics")

from ultralytics import YOLO


number = 3 #input how many tasks in your work
model = YOLO('/data/zhujiang/code/YOLOv8-multi-task-main/ultralytics/runs/multi/yolopm/weights/best.pt')  # Validate the model
model.predict(source='/data/zhujiang/code/YOLOv8-multi-task-main/dataset/bdd100k/images/10k/test/testA', imgsz=(384,672), device=[3],name='v4_daytime', save=True, conf=0.25, iou=0.45, show_labels=False)
