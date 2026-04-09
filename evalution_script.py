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

# Map PanTS labels to our 6-Class System
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
test_lbl_dir = '/mnt/dev1/kstasinos/panTS/PanTS/data/LabelTe' # Double-check this path!

test_files = []
# Get all the patient folder names (e.g., 'PanTS_00009001')
case_dirs = sorted([d for d in os.listdir(test_img_dir) if os.path.isdir(os.path.join(test_img_dir, d))])


for case_id in case_dirs:
    # Use glob to find the .nii.gz file INSIDE the patient's folder
    img_paths = glob.glob(os.path.join(test_img_dir, case_id, "*.nii.gz"))
    lbl_paths = glob.glob(os.path.join(test_lbl_dir, case_id, "*.nii.gz"))
    
    # Only add to our list if BOTH the image and label exist
    if img_paths and lbl_paths:
        test_files.append({"image": img_paths[0], "label": lbl_paths[0]})

print(f"Found {len(test_files)} valid Image/Label pairs.")

test_ds = Dataset(data=test_files, transform=test_transforms)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

# Load Model
checkpoint_path = '/mnt/dev1/kstasinos/PDACseg/PDACseg.pt'
topology_space = TopologySearch(channel_mul=1.0, num_blocks=12, num_depths=4, use_downsample=True, spatial_dims=3, device=device)
model = DiNTS(dints_space=topology_space, in_channels=1, num_classes=6, use_downsample=True).to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

# ==========================================
# 3. METRIC TRACKERS
# ==========================================
metrics = {
    "patients_w_tumor": 0, "patients_detected": 0,
    "total_tumors_gt": 0, "tumors_detected": 0,
    "dice_scores": [], "auc_scores": [], "specificity_scores": []
}

print(f" Evaluating {len(test_files)} cases for Leaderboard Metrics...")

with torch.no_grad():
    for step, batch in enumerate(test_loader):
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        
        with torch.amp.autocast('cuda', dtype=torch.float16):
            logits = sliding_window_inference(inputs, (96, 96, 96), 2, model, overlap=0.5, sw_device=device, device=torch.device('cpu'))
            
        # 1. Extract Tumor Probabilities & Hard Mask
        probs = F.softmax(logits, dim=1)[0, 2].numpy() # Class 2 is Tumor
        pred_mask = (torch.argmax(logits, dim=1)[0] == 2).numpy()
        gt_mask = (labels[0, 0].cpu() == 2).numpy()

        labeled_pred, num_features = label(pred_mask)
        cleaned_pred = np.zeros_like(pred_mask)
        
        for i in range(1, num_features + 1):
            component = (labeled_pred == i)
            # Only keep the tumor blob if it is larger than 200 voxels (filters out noise speckles)
            if component.sum() > 200: 
                cleaned_pred[component] = True
                
        # Overwrite the noisy prediction with our clean one before calculating metrics
        pred_mask = cleaned_pred 

        # 2. DICE SCORE (Tumor)
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        dice = (2. * intersection) / (pred_mask.sum() + gt_mask.sum() + 1e-8)
        if gt_mask.sum() > 0: metrics["dice_scores"].append(dice)

        # 3. PATIENT-WISE SENSITIVITY (P-Sen)
        if gt_mask.sum() > 0:
            metrics["patients_w_tumor"] += 1
            if pred_mask.sum() > 0:
                metrics["patients_detected"] += 1

        # 4. TUMOR-WISE SENSITIVITY (T-Sen)
        labeled_gt, num_tumors = label(gt_mask)
        metrics["total_tumors_gt"] += num_tumors
        
        for i in range(1, num_tumors + 1):
            single_tumor_mask = (labeled_gt == i)
            if np.logical_and(pred_mask, single_tumor_mask).sum() > 0:
                metrics["tumors_detected"] += 1

        # 5. SPECIFICITY & AUC (Subsampled for Memory Safety)
        gt_flat = gt_mask[::10, ::10, ::10].flatten()
        prob_flat = probs[::10, ::10, ::10].flatten()
        pred_flat = pred_mask[::10, ::10, ::10].flatten()

        if len(np.unique(gt_flat)) > 1:
            metrics["auc_scores"].append(roc_auc_score(gt_flat, prob_flat))
            tn, fp, fn, tp = confusion_matrix(gt_flat, pred_flat).ravel()
            metrics["specificity_scores"].append(tn / (tn + fp + 1e-8))

        print(f"   [{step+1}/{len(test_files)}] | Dice: {dice:.4f} | Tumors Found: {num_tumors}")


final_p_sen = metrics["patients_detected"] / max(metrics["patients_w_tumor"], 1)
final_t_sen = metrics["tumors_detected"] / max(metrics["total_tumors_gt"], 1)
final_spe = np.mean(metrics["specificity_scores"])
final_auc = np.mean(metrics["auc_scores"])
final_dsc = np.median(metrics["dice_scores"])


print(f"   Patient-wise Sensitivity (P-Sen): {final_p_sen:.4f}")
print(f"   Tumor-wise Sensitivity (T-Sen):   {final_t_sen:.4f}")
print(f"   Specificity (Spe):                {final_spe:.4f}")
print(f"   AUC:                              {final_auc:.4f}")
print(f"   Dice Score (DSC):                 {final_dsc:.4f}")
