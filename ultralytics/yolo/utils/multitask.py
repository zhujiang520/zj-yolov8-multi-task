# Ultralytics YOLO — helpers for detection + multi-head segmentation training.


def multitask_loss_bucket(label_name: str) -> str:
    """Map a ``labels_list`` entry to the loss group used in ``DetectionSegmentationTrainer.loss_names``."""
    ln = str(label_name).lower()
    if 'det' in ln:
        return 'det'
    if 'seg' in ln:
        return 'seg'
    raise ValueError(
        f"Multi-task dataset: each ``labels_list`` name must contain 'det' (detection) or 'seg' (segmentation); "
        f"got {label_name!r}. Example: 'detection-object', 'seg-road', 'seg-lane'.")


def n_multi_tasks(data) -> int:
    """Number of supervised tasks; equals length of ``labels_list`` in the data yaml."""
    return len(data['labels_list'])


def assert_model_outputs_cover_tasks(preds_list, labels_list):
    """``MultiModel`` returns one tensor/tuple per Detect/Segment head; each ``labels_list`` task must have one."""
    n_head = len(preds_list)
    n_task = len(labels_list)
    if n_head < n_task:
        raise ValueError(
            f'Multi-task: model produced {n_head} head output(s) but data defines {n_task} task(s) in '
            f'labels_list={labels_list!r}. Add a Detect/Segment head per task in the model yaml (same order), '
            f'or shorten labels_list.')
