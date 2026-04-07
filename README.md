# PDACseg

# PanTS Challenge Submission 

**Author:** Konstantinos Stasinos MD, PhD, Christos Athanasiou MD, PhD

**Model Architecture:** DiNTS (num_classes=6)

This repository contains the inference script and model weights for evaluating our 3D segmentation model on pancreatic cancer datasets. The model was trained on a PanTS dataset utilizing a 6-class anatomical mapping system.

## Environment Requirements
* `torch` >= 2.0.0
* `monai` >= 1.3.0
* `nibabel`

#### Class Output Mapping
The model outputs a single `.nii.gz` mask per input image, with a 6-class mapping system:
* `0`: Background
* `1`: Pancreas (Note: Includes the Pancreatic Duct due to PanTS label 21 integration)
* `2`: Pancreatic Tumor
* `3`: Superior Mesenteric Artery (SMA)
* `4`: Vein (SMV / Portal Vein)
* `5`: Biliary Tract / CBD

## Usage Instructions
The `evaluation_script.py` script requires three arguments: the input directory containing raw CT images, the output directory for the prediction masks, and the path to the model weights.

```bash
python evaluation_script.py \
  --input_dir /path/to/test/images \
  --output_dir /path/to/save/predictions \
  --checkpoint /path/to/PDACSeg_model.pt
