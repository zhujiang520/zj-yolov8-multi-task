import sys
sys.path.insert(0, "/data/zhujiang/code/YOLOv8-multi-task-main/ultralytics")
# 现在就可以导入Yolo类了
from ultralytics import YOLO

# Load a model
model = YOLO('/data/zhujiang/code/YOLOv8-multi-task-main/ultralytics/models/v8/yolov8-bdd-v4-one-dropout-individual-n.yaml', task='multi')  # build a new model from YAML
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
# model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights

# Train the model
model.train(data='/data/zhujiang/code/YOLOv8-multi-task-main/ultralytics/datasets/bdd-multi-dlyw.yaml', batch=12, epochs=300, imgsz=(640,640), device=[0,1,2,3], name='yolopm', val=True, task='multi',classes=[0,1],combine_class=[0],single_cls=True)
