import os
import glob
import torch
import argparse
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, 
    ScaleIntensityRanged, ToTensord, SaveImaged, AsDiscrete
)
from monai.data import DataLoader, Dataset, decollate_batch
from monai.networks.nets import DiNTS, TopologySearch
from monai.inferers import sliding_window_inference
import torch.amp

def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Initializing PanTS Evaluation on: {device}")

    # 1. SETUP TRANSFORMS
    test_transforms = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=-96, a_max=215, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=["image"])
    ])

    # 2. LOAD DATA
    test_images = sorted(glob.glob(os.path.join(args.input_dir, "*.nii.gz")))
    test_files = [{"image": img} for img in test_images]
    
    if len(test_files) == 0:
        raise ValueError(f"No .nii.gz files found in {args.input_dir}")
        
    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    # 3. LOAD MODEL
    print(f"Loading Model Checkpoint: {args.checkpoint}")
    topology_space = TopologySearch(channel_mul=1.0, num_blocks=12, num_depths=4, use_downsample=True, spatial_dims=3, device=device)
    model = DiNTS(dints_space=topology_space, in_channels=1, num_classes=6, use_downsample=True).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # 4. SETUP SAVER
    # This automatically matches the output NIfTI to the input NIfTI's affine matrix
    saver = SaveImaged(
        output_dir=args.output_dir, 
        output_postfix="pred", 
        output_ext=".nii.gz", 
        resample=False, 
        separate_folder=False
    )
    post_pred = AsDiscrete(argmax=True)

    # 5. INFERENCE LOOP
    print(f"⚙️ Processing {len(test_images)} cases...")
    os.makedirs(args.output_dir, exist_ok=True)

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            inputs = test_data["image"].to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.float16):
                outputs = sliding_window_inference(
                    inputs=inputs, roi_size=(96, 96, 96), sw_batch_size=2, 
                    predictor=model, overlap=0.5, 
                    sw_device=device, device=torch.device('cpu')
                )
            
            # Extract the argmax class (0-6)
            outputs = [post_pred(i) for i in decollate_batch(outputs)]
            
            # Save to disk
            for batch_idx, output_tensor in enumerate(outputs):
                # Attach the original image's metadata to the prediction so it saves correctly
                test_data["pred"] = output_tensor
                saver(test_data["pred"], test_data["image_meta_dict"][batch_idx])
                
            print(f"Saved mask {i+1}/{len(test_images)}")

    print(f"Evaluation complete! Masks saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PanTS Challenge Inference Script")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing test .nii.gz images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predicted masks")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .pt model weights")
    args = parser.parse_args()
    main(args)