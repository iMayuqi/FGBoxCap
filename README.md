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

# Dataset Statistics
| #Categories |
| color,  physical characteristics, action,  resolution,  material,  status,  shape,  style type,  text format,  texture,  human race,  size,  pattern,  degree of old and new,  integrity,  order,  dimensionality,  facial expression,  length,  color quantity,  brightness,  gender,  time,  transparency,  age maturity,  dryness,  clarity,  brand name,  cooked,  hardness,  cleanliness,  location,  temperature,  scent,  pitch,  direction,  rarity,  position,  weight,  material, style type,  percentage,  disease,  food,  quantity,  texture, pattern,  grade,  object type,  hardware,  relationship,  weather condition,  taste,  brand,  alignment,  chemical compound,  topography,  material, pattern,  size, length,  product name,  punctuation,  speed,  sound,  orientation,  emotional state,  date,  number,  location/position,  material, shape,  flavor,  color, degree of old and new,  degree of old and new, dryness,  texture, material,  functionality,  material, integrity,  color quantity, color,  math operator,  smell,  era,  blood type,  chemical property,  diet type,  energy source,  material property,  color, size,  degree of old and new, cooked,  product model,  angle,  weather,  degree of old and new, material,  chemical composition,  status (brand/model),  degree of old and new, texture,  texture, color,  season,  density,  music note,  car model,  medical condition,  symbol,  time of day,  film title,  integrity (damaged),  size, color,  color quantity, material,  product brand,  lighting,  pattern, material,  color material,  composition,  language,  frequency |

| Split        | #Images | #Instances | #Categories | Notes |
|--------------|--------:|-----------:|------------:|-------|
| Train (seen) | 3,200,000 | 5,800,000 | 1,200 | Includes fine-grained class labels |
| Train (unseen) | 1,500,000 | 2,700,000 | 900 | Hard negatives included |
| Test         | 800,000 | 1,450,000 | 1,500 | Zero-shot evaluation supported |
