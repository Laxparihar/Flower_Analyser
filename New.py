import numpy as np
from collections import defaultdict
from sklearn.metrics import precision_recall_curve, auc


# 1. IOU CALCULATION
def calculate_iou(box1, box2):
    """
    box format: [x_min, y_min, x_max, y_max]
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area else 0
    return iou


# 2. MATCH PREDICTIONS TO GROUND TRUTH
def match_predictions(pred_boxes, gt_boxes, iou_threshold=0.5):
    matches = []
    used_gt = set()
    
    for pred_idx, pred in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt in enumerate(gt_boxes):
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou >= iou_threshold and iou > best_iou and gt_idx not in used_gt:
                best_iou = iou
                best_gt_idx = gt_idx
        if best_gt_idx >= 0:
            matches.append((pred_idx, best_gt_idx))
            used_gt.add(best_gt_idx)
    
    tp = len(matches)
    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return matches, tp, fp, fn


# 3. COMPUTE PRECISION & RECALL PER CLASS
def evaluate_class(preds, gts, class_name, iou_threshold=0.5):
    pred_filtered = [p for p in preds if p['class'] == class_name]
    gt_filtered = [g for g in gts if g['class'] == class_name]
    
    pred_filtered = sorted(pred_filtered, key=lambda x: x['confidence'], reverse=True)
    
    tp_list = []
    conf_list = []
    matched_gt = set()
    
    for pred in pred_filtered:
        iou_max = 0
        matched = False
        for gt_idx, gt in enumerate(gt_filtered):
            if gt_idx in matched_gt:
                continue
            iou = calculate_iou(pred['bbox'], gt['bbox'])
            if iou >= iou_threshold and iou > iou_max:
                iou_max = iou
                matched_idx = gt_idx
                matched = True
        if matched:
            tp_list.append(1)
            matched_gt.add(matched_idx)
        else:
            tp_list.append(0)
        conf_list.append(pred['confidence'])
    
    tp_cumsum = np.cumsum(tp_list)
    fp_cumsum = np.cumsum([1 - x for x in tp_list])
    
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recalls = tp_cumsum / (len(gt_filtered) + 1e-6)
    
    return precisions, recalls, conf_list


# 4. AVERAGE PRECISION CALCULATION
def compute_ap(precisions, recalls):
    return auc(recalls, precisions)


# 5. MEAN AVERAGE PRECISION (mAP)
def evaluate_map(preds, gts, class_names, iou_threshold=0.5):
    ap_per_class = {}
    for cls in class_names:
        precisions, recalls, _ = evaluate_class(preds, gts, cls, iou_threshold)
        ap = compute_ap(precisions, recalls)
        ap_per_class[cls] = ap
    mAP = np.mean(list(ap_per_class.values()))
    return ap_per_class, mAP



ap_results, mean_ap = evaluate_map(predictions, ground_truths, classes, iou_threshold=0.5)
print("AP per class:", ap_results)
print("mAP:", mean_ap)
