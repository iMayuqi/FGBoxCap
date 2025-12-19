# FGBoxCap
Official implementation of FGBoxCap and FG-GDino for fine-grained open-vocabulary object detection

# FGBoxCap Inference with GroundingDINO

This repository contains the inference code for the FGBoxCap dataset using the GroundingDINO model.

# Dataset Information
- The FGBoxCap test set provides multi-dimensional evaluation subsets, enabling comprehensive assessment of a model’s ability to understand fine-grained information.
- FGBoxCap will be available in the near future.

## Script Overview

- `groundingdino_inference.py`  
  Provides the inference pipeline to perform object detection on the FGBoxCap test set using GroundingDINO.  
  The script loads the pre-trained model, processes test images from the FGBoxCap dataset, and outputs detection results with fine-grained captions.

## Usage

1. Prepare the FGBoxCap test dataset and ensure the paths are correctly set in the script or config files.
2. Run the inference script:

## Usage Example

```bash
python groundingdino_inference.py \
  -c /path/to/config.py \
  -p /path/to/checkpoint.pth \
  --imgs_path /path/to/images/ \
  --map_out /path/to/output/ \
  --n_hardnegatives 5\
  --genfolder /path/to/annotation_folder/ 
