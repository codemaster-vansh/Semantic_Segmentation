import os
import json
from PIL import Image
import numpy as np

mask_dir = r''  # PUT THE FILE DIRECTORY WHERE THE IMAGE MASKS ARE THERE
color_to_class = {}
color_to_class_json = {}
next_class_id = 0
NUM_MASKS =   # CHANGE WITH NUMBER OF CLASSES. 

mask_files = sorted(os.listdir(mask_dir))

for i, filename in enumerate(mask_files):
    if next_class_id == NUM_MASKS:
        break
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(mask_dir, filename)
        img = Image.open(path).convert("RGB")
        mask = np.array(img)

        # Get unique colors in the mask
        unique_colors = np.unique(mask.reshape(-1, 3), axis=0)

        print(f"\n{filename}: Found {len(unique_colors)} unique color(s)")
        for color in unique_colors:
            color_tuple = tuple(color.tolist())  # convert from np.array to tuple
            if color_tuple not in color_to_class:
                color_to_class[color_tuple] = next_class_id
                print(f" \u2795 Added color {color_tuple} as class {next_class_id}")
                next_class_id += 1
            else:
                print(f" \u2705 Color {color_tuple} already mapped to class {color_to_class[color_tuple]}")

for keys, values in color_to_class.items():
    color_to_class_json[values] = keys

with open("color_mapping.json","w") as file:
    json.dump(color_to_class_json,file,indent=2)
print("\nFinal color-to-class mapping stored in 'color_mapping.json'")
