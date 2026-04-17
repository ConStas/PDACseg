import os
import glob
import torch
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score, confusion_matrix
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    ScaleIntensityRanged, ToTensord, Lambdad
)
from monai.data import DataLoader, Dataset
from monai.networks.nets import DiNTS, TopologySearch
from monai.inferers import sliding_window_inference
import torch.nn.functional as F
import torch.amp

# ==========================================
# 1. SETUP & MAPPING (6-CLASS)
# ==========================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Initializing Leaderboard Evaluation on: {device}")

def map_pants_labels(x):
    out = torch.zeros_like(x)
    out[(x == 18) | (x == 19) | (x == 20) | (x == 21)] = 1  # Panc + PD
    out[x == 28] = 2  # Tumor
    out[x == 26] = 3  # SMA
    out[x == 27] = 4  # Vein
    out[x == 7]  = 5  # CBD
    return out

test_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityRanged(keys=["image"], a_min=-96, a_max=215, b_min=0.0, b_max=1.0, clip=True),
    Lambdad(keys=["label"], func=map_pants_labels),
    ToTensord(keys=["image", "label"])
])

# ==========================================
# 2. LOAD TEST SET & MODEL
# ==========================================
test_img_dir = '/mnt/dev1/kstasinos/panTS/PanTS/data/ImageTe' 
test_lbl_dir = '/mnt/dev1/kstasinos/panTS/PanTS/data/LabelTe'

test_files = []
case_dirs = sorted([d for d in os.listdir(test_img_dir) if os.path.isdir(os.path.join(test_img_dir, d))])

for case_id in case_dirs:
    img_paths = glob.glob(os.path.join(test_img_dir, case_id, "*.nii.gz"))
    lbl_paths = glob.glob(os.path.join(test_lbl_dir, case_id, "*.nii.gz"))
    if img_paths and lbl_paths:
        test_files.append({"image": img_paths[0], "label": lbl_paths[0]})

print(f"Found {len(test_files)} valid Image/Label pairs.")

test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

checkpoint_path = '/mnt/dev1/kstasinos/PDACseg/PDACseg.pt'
topology_space = TopologySearch(channel_mul=1.0, num_blocks=12, num_depths=4, use_downsample=True, spatial_dims=3, device=device)
model = DiNTS(dints_space=topology_space, in_channels=1, num_classes=6, use_downsample=True).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ==========================================
# 3. METRIC TRACKERS & INFERENCE LOOP
# ==========================================
metrics = {
    "dice_scores": [],
    "total_tumors_gt": 0, "tumors_detected": 0,
    "ct_gt_labels": [],       
    "ct_pred_binary": [],     
    "ct_pred_probs": []       
}

print(f"Evaluating {len(test_files)} cases for CT-Level Metrics...")

with torch.no_grad():
    for step, batch in enumerate(test_loader):
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = sliding_window_inference(inputs, (96, 96, 96), 2, model, overlap=0.5, sw_device=device, device=torch.device('cpu'))
            
        # 1. Extract Hard Masks & Probs (No harsh probability threshold)
        probs = F.softmax(logits, dim=1)[0, 2].numpy() # Class 2 is Tumor
        pred_classes = torch.argmax(logits, dim=1)[0].numpy()
        pred_tumor = (pred_classes == 2)
        pred_panc = (pred_classes == 1)
        gt_mask = (labels[0, 0].cpu() == 2).numpy()

        # Step A: The Pancreas Bounding Box
        labeled_panc, num_panc = label(pred_panc)
        if num_panc > 0:
            sizes = [np.sum(labeled_panc == i) for i in range(1, num_panc + 1)]
            largest_panc_idx = np.argmax(sizes) + 1
            true_panc = (labeled_panc == largest_panc_idx)
            
            coords = np.argwhere(true_panc)
            z_min, y_min, x_min = coords.min(axis=0)
            z_max, y_max, x_max = coords.max(axis=0)

            pad = 30 
            z_min = max(0, z_min - pad); z_max = min(pred_tumor.shape[0], z_max + pad)
            y_min = max(0, y_min - pad); y_max = min(pred_tumor.shape[1], y_max + pad)
            x_min = max(0, x_min - pad); x_max = min(pred_tumor.shape[2], x_max + pad)

            safe_zone = np.zeros_like(pred_tumor)
            safe_zone[z_min:z_max, y_min:y_max, x_min:x_max] = True
            pred_mask = np.logical_and(pred_tumor, safe_zone)
        else:
            pred_mask = np.zeros_like(pred_tumor) 

        # Step B: Moderate Size Filter (No probability requirements)
        labeled_tumor, num_tumor = label(pred_mask)
        cleaned_tumor = np.zeros_like(pred_mask)
        for i in range(1, num_tumor + 1):
            component = (labeled_tumor == i)
            # 500 voxels is the sweet spot
            if component.sum() > 500: 
                cleaned_tumor[component] = True
                
        pred_mask = cleaned_tumor
        # ==========================================

        # 2. DICE SCORE & TUMOR-WISE SENSITIVITY
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        dice = (2. * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-8)
        if gt_mask.sum() > 0: metrics["dice_scores"].append(dice)

        labeled_gt, num_tumors_gt = label(gt_mask)
        metrics["total_tumors_gt"] += num_tumors_gt
        for i in range(1, num_tumors_gt + 1):
            single_tumor_mask = (labeled_gt == i)
            if np.logical_and(pred_mask, single_tumor_mask).sum() > 0:
                metrics["tumors_detected"] += 1

        # 3. CT-LEVEL METRICS TRACKING
        has_tumor_gt = int(gt_mask.sum() > 0)
        has_tumor_pred = int(pred_mask.sum() > 0)
        
        ct_prob = float(probs[pred_mask].max()) if has_tumor_pred else 0.0

        metrics["ct_gt_labels"].append(has_tumor_gt)
        metrics["ct_pred_binary"].append(has_tumor_pred)
        metrics["ct_pred_probs"].append(ct_prob)

        print(f"   [{step+1}/{len(test_files)}] | CT GT: {has_tumor_gt} | CT Pred: {has_tumor_pred} | Dice: {dice:.4f}")

y_true = np.array(metrics["ct_gt_labels"])
y_pred = np.array(metrics["ct_pred_binary"])
y_prob = np.array(metrics["ct_pred_probs"])

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

patient_sens = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
patient_spec = tn / (tn + fp) if (tn + fp) > 0 else float('nan')
tumor_sens = metrics["tumors_detected"] / max(metrics["total_tumors_gt"], 1)

if len(np.unique(y_true)) > 1:
    patient_auc = roc_auc_score(y_true, y_prob)
else:
    patient_auc = float('nan')

final_dsc = np.mean(metrics["dice_scores"])

print(f"   Patient-wise Sensitivity: {patient_sens:.4f}")
print(f"   Tumor-wise Sensitivity:   {tumor_sens:.4f}")
print(f"   Specificity:              {patient_spec:.4f}")  
print(f"   AUC:                      {patient_auc:.4f}")
print(f"   Dice Score (DSC):         {final_dsc:.4f}")
