import json
import os

# Your data dictionaries
ML = {
    "mel_10_000.tif": [106, 78, 423],
    "mel_10_001.tif": [159, 183, 375],
    "mel_10_002.tif": [141, 66, 119],
    "mel_10_003.tif": [164, 147, 87],
    "mel_10_004.tif": [147, 128, 429],
    "mel_10_005.tif": [118, 121, 138],
}

PP = {
    "fluepup_1_000.tif": [141, 159, 307],
    "fluepup_1_001.tif": [112, 165, 321],
    "fluepup_1_002.tif": [113, 163, 196],
    "fluepup_1_003.tif": [154, 142, 205],
    "fluepup_1_004.tif": [138, 94, 319],
    "fluepup_1_005.tif": [144, 152, 191],
}

SL = {
    "soldat_10_000.tif": [90, 104, 394],
    "soldat_10_001.tif": [119, 115, 129],
    "soldat_10_002.tif": [135, 106, 132],
    "soldat_10_003.tif": [89, 79, 154],
    "soldat_10_004.tif": [95, 158, 152],
    "soldat_10_005.tif": [146, 167, 122],
}

WO = {
    "bank_1_000.tif": [173, 123, 310],
    "bank_1_001.tif": [166, 82, 292],
    "bank_1_002.tif": [164, 77, 332],
    "bank_1_003.tif": [120, 155, 325],
    "bank_1_004.tif": [122, 194, 275],
    "bank_1_005.tif": [130, 164, 188],
}

AC = {
    "bcrick_1_000.tif": [193, 139, 355],
    "bcrick_1_001.tif": [106, 184, 171],
    "bcrick_1_002.tif": [108, 64, 327],
    "bcrick_1_003.tif": [180, 161, 332],
    "bcrick_1_004.tif": [142, 78, 343],
    "bcrick_1_005.tif": [117, 149, 352],
}

BC = {
    "sfaar_1_000.tif": [200, 140, 210],
    "sfaar_1_001.tif": [79, 158, 340],
    "sfaar_1_002.tif": [81, 82, 312],
    "sfaar_1_003.tif": [60, 119, 292],
    "sfaar_1_004.tif": [166, 194, 306],
    "sfaar_1_005.tif": [192, 108, 312],
}

BF = {
    "spy_1_000.tif": [196, 113, 242],
    "spy_1_001.tif": [174, 144, 209],
    "spy_1_002.tif": [116, 116, 224],
    "spy_1_003.tif": [78, 109, 202],
    "spy_1_004.tif": [192, 131, 231],
    "spy_1_005.tif": [76, 118, 196],
}

BL = {
    "boffel_1_000.tif": [157, 113, 173],
    "boffel_1_001.tif": [126, 137, 168],
    "boffel_1_002.tif": [106, 111, 192],
    "boffel_1_003.tif": [169, 142, 205],
    "boffel_1_004.tif": [142, 170, 186],
    "boffel_1_005.tif": [130, 110, 372],
}

BP = {
    "guld_1_000.tif": [145,107,320],
    "guld_1_001.tif": [145,129,213],
    "guld_1_002.tif": [153,163,317],
    "guld_1_003.tif": [106,135,200],
    "guld_1_004.tif": [140,97,327],
}

CF = {
    "krol_1_000.tif": [139,114,211],
    "krol_1_001.tif": [160,115,295],
    "krol_1_002.tif": [139,97,215],
    "krol_1_003.tif": [111,129,109],
    "krol_1_004.tif": [164,129,307],
}

GH = {
    "gras_1_000.tif": [156,187,346],
    "gras_1_001.tif": [178,150,200],
    "gras_1_002.tif": [84,165,261],
    "gras_1_003.tif": [86,106,161],
    "gras_1_004.tif": [181,146,191],
}

MA = {
    "maddi_1_000.tif": [129,93,192],
    "maddi_1_001.tif": [117,146,189],
    "maddi_1_002.tif": [134,99,183],
    "maddi_1_003.tif": [113,123,190],
    "maddi_1_004.tif": [146,151,323],
}

# Combine them
combined_data = {
    "ML": ML,
    "PP": PP,
    "SL": SL,
    "WO": WO,
    "AC": AC,
    "BC": BC,
    "BF": BF,
    "BL": BL,
    "BP": BP,
    "CF": CF,
    "GH": GH,
    "MA": MA,
}

# Create a new local folder
folder_path = "annotations_output"
os.makedirs(folder_path, exist_ok=True)

# Save file inside the new folder
file_name = "image_annotations.json"
full_path = os.path.join(folder_path, file_name)

with open(full_path, "w") as f:
    json.dump(combined_data, f, indent=4)

print(f"Successfully saved to: {full_path}")