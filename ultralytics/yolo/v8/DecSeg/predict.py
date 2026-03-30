# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.yolo.engine.predictor_multi import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops, yaml_load
from ultralytics.yolo.utils.checks import check_file


class MultiPredictor(BasePredictor):
    """Predictor that only keeps the first ``len(labels_list)`` heads (same order as training)."""

    def _n_multi_tasks_for_predict(self):
        """How many heads to run through postprocess/visualize; extra heads are untrained noise."""
        d = self.args.data
        if not d:
            return None
        if isinstance(d, dict) and 'labels_list' in d:
            return len(d['labels_list'])
        try:
            data = yaml_load(check_file(d))
            if isinstance(data, dict) and 'labels_list' in data:
                return len(data['labels_list'])
        except Exception:
            pass
        return None

    def postprocess_det(self, preds, img, orig_imgs):
        """Postprocesses predictions and returns a list of Results objects."""
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def postprocess_seg(self, preds):
        """Postprocesses YOLO predictions and returns output detections with proto."""
        preds = torch.nn.functional.interpolate(preds, size=(720, 1280), mode='bilinear', align_corners=False)
        preds = self.sigmoid(preds)
        _, preds = torch.max(preds, 1)
        return preds


def predict(cfg=DEFAULT_CFG, use_python=False):
    """Runs YOLO model inference on input image(s)."""
    model = cfg.model or 'yolov8n.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = MultiPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()
