# PDACseg

# PanTS Challenge Submission 

**Author:** Konstantinos Stasinos MD, PhD, Christos Athanasiou MD, PhD

**Model Architecture:** DiNTS (num_classes=6)

This repository contains the inference script and model weights for evaluating our 3D segmentation model on pancreatic cancer datasets. The model was trained on a PanTS dataset utilizing a 6-class anatomical mapping system.

## Environment Requirements
* `torch` >= 2.0.0
* `monai` >= 1.3.0
* `nibabel`

## Usage Instructions
The `evaluation_script.py` script requires three arguments: the input directory containing raw CT images, the output directory for the prediction masks, and the path to the model weights.

```bash
python evaluation_script.py \
  --input_dir /path/to/test/images \
  --output_dir /path/to/save/predictions \
  --checkpoint /path/to/PDACSeg_model.pt