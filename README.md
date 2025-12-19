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
```

# Dataset Statistics
## All Attribute Parent Categories
| Categories |
|------------|
| color,  physical characteristics, action,  resolution,  material,  status,  shape,  style type,  text format, texture,  human race,  size,  pattern,  degree of old and new,  integrity,  order,  dimensionality,  facial expression,  length,  color quantity,  brightness,  gender,  time,  transparency,  age maturity,  dryness,  clarity,  brand name,  cooked,  hardness,  cleanliness,  location,  temperature,  scent,  pitch,  direction,  rarity,  position,  weight,  material, style type,  percentage,  disease,  food,  quantity,  texture, pattern,  grade,  object type,  hardware,  relationship,  weather condition,  taste,  brand,  alignment,  chemical compound,  topography,  material, pattern,  size, length,  product name,  punctuation,  speed,  sound,  orientation,  emotional state,  date,  number,  location/position,  material, shape,  flavor,  color, degree of old and new,  degree of old and new, dryness,  texture, material,  functionality,  material, integrity,  color quantity, color,  math operator,  smell,  era,  blood type,  chemical property,  diet type,  energy source,  material property,  color, size,  degree of old and new, cooked,  product model,  angle,  weather,  degree of old and new, material,  chemical composition,  status (brand/model),  degree of old and new, texture,  texture, color,  season,  density,  music note,  car model,  medical condition,  symbol,  time of day,  film title,  integrity (damaged),  size, color,  color quantity, material,  product brand,  lighting,  pattern, material,  color material,  composition,  language,  frequency |

## Statistics of attribute word categories and representative words within each category (partial display)
| Attribute    | #num | #representative words |
|--------------|-----:|-----------------------|
| Color | 14982 | white, black, blue, red, green, gray, brown, yellow, orange, dark |
| Material | 5906 | wooden, metal, plastic, glass, stone, concrete, metallic, brick, leather, wood |
| Shape | 5894 | rectangular, circular, square, round, flat, cylindrical, curved, oval, pointed |
| Action | 6500 | standing, running, resting, flowing, wearing, rising, sitting, growing, driving, reading |
| Physical Characteristics | 14668 | slender, towering, curly, gentle, resembling, robust, muscular, protruding, succulent, tilted |
| Status | 13417 | open, unique, labeled, stationary, limited, equipped, notable, gas, public, private |
| Style type | 10380 | bold, ornate, elegant, rustic, decorative, modern, traditional, classic, stylized, whimsical |
| Text Format | 5254 | highlighted, capital, bare, cursive, handwritten, elaborate, placeholder, uppercase, unreadable, italic |
| texture | 5138 | intricate, lush, thick, delicate, sleek, textured, soft, shiny, fluffy, dense |
| Human race | 4078 | Caucasian, Asian, American, Japanese, Chinese, French, European, British, German, Spanish |
| Size | 3508 | small, large, tall, smaller, tiny, wide, grand, massive, vast, medium |
| Pattern | 3067 | striped, floral, checkered, plaid, dotted, swirling, concentric, lined, stars |
| degree of old and new | 3020 | vintage, worn, weathered, fresh, faint, antique, ancient, faded, slight, rusted |
| integrity | 2231 | solid, full, attached, broken, closed, filled, cracked, folded, distorted, partial |
| order | 2200 | scattered, prominently, empty, surrounding, single, cluttered, positioned, double, matching, different |
| dimensionality | 1981 | vertical, digital, narrow, high, raised, expansive, dynamic, elevated, mountainous, complex |
| facial expression | 1624 | serene, charming, smiling, tranquil, calm, dramatic, expressive, menacing, playful, lively |
| length | 1346 | long, short, extending, inch, ounces, kilometers, seconds, minutes, years, meters |
| color quantity | 872 | colorful, darker, various, lighter, numerous, diverse, varied, assorted, monochrome, shades |
| cooked (raw or cooked food) | 769 | cooked, roasted, smoked, tomato, beef, steaming, sourdough, edible, butter, cheese |
| brightness | 744 | vibrant, light, bright, vivid, glowing, stark, muted, lit, radiant, sunlight |
| gender | 712 | female, male, cowboy, woman, feminine, gentleman, androgynous, fighter, child, naval |
| time | 668 | second, 9:41, hours, 12:30, 12:00, elapsed, 11:20, 6:30, 10:00, 7:00 |
| transparency | 637 | transparent, translucent, hazy, murky, sheer, shadowy, shadow, shaded, smoke, opaque |
| age maturity | 611 | young, elderly, adult, mature, baby, teenage, prehistoric, juvenile, youthful, newborn |
| dryness | 500 | dry, wet, dried, barren, dusty, moist, arid, damp, dust, milk |

## Statistics of the two types of relationship words and representative words within each category (partial display).
| Attribute    | #num | #representative words |
|--------------|-----:|-----------------------|
| Spatial | 15643 | left, below, surrounding, bottom, top, above, right, behind, beneath, around, inside, next to |
| Action | 11187 | worn, holding, held, wearing, supporting, adorned, covering, parked, connected, wrapped, carried |


