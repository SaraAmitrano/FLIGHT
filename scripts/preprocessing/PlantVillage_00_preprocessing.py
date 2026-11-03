# %%
import os
import re
import numpy as np
import tqdm as tqdm
import pickle as pkl
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

dataset = "PlantVillage"

label_to_use = "disease"

root_path = Path().resolve()
if root_path.name == "scripts_dev":
    root_path = root_path.parent

datasets_path = root_path / "datasets"

dataset_path = (
    datasets_path / dataset / "Plant_leave_diseases_dataset_without_augmentation"
)

# %% Grouping images by disease

# diseases = [
#     "healthy",
#     "rust",
#     "scab",
#     "Early_blight",
#     "Black_rot",
#     "Yellow_Leaf_Curl",
#     "Septoria_leaf_spot",
#     "Target_Spot",
#     "Late_blight",
#     "Bacterial_spot",
#     "Leaf_scorch",
#     "Citrus_greening",
#     "Isariopsis_Leaf_Spot",
#     "Gray_leaf_spot",
#     "Northern_Leaf_Blight",
#     "Powdery_mildew",
#     "spider_mite",
#     "Black_Measles",
#     "Leaf_Mold",
#     "mosaic_virus",
# ]


diseases = [
    "healthy",
    "rust",
    # "scab",
    "Early_blight",
    "Black_rot",
    # "Yellow_Leaf_Curl",
    # "Septoria_leaf_spot",
    # "Target_Spot",
    "Late_blight",
    "Bacterial_spot",
    # "Leaf_scorch",
    "Citrus_greening",
    # "Isariopsis_Leaf_Spot",
    # "Gray_leaf_spot",
    # "Northern_Leaf_Blight",
    "Powdery_mildew",
    # "spider_mite",
    # "Black_Measles",
    # "Leaf_Mold",
    # "mosaic_virus",
]

dict_disease = {disease: [] for disease in diseases}

relative_path = (
    "../datasets/PlantVillage/Plant_leave_diseases_dataset_without_augmentation"
)

for f in os.listdir(dataset_path):
    if f == "Background_without_leaves":
        continue
    clean_name = re.sub(r"[^\w\s]", "", f)
    parts = set(clean_name.replace(" ", "_").replace("___", "_").split("_"))
    matched_disease = None

    # match con più parole
    for disease in diseases:
        disease_parts = set(disease.split("_"))
        if disease_parts.issubset(parts):
            matched_disease = disease
            break

    # match con una parola
    if not matched_disease:
        for disease in diseases:
            if disease in parts:
                matched_disease = disease
                break
    
    if disease not in diseases:
        print(f"Skipping {f}")
        continue

    if matched_disease:
        image_folder = relative_path + "/" + f
        dict_disease[matched_disease] += [
            str(image_folder + "/" + img)
            for img in os.listdir(image_folder)
            if img.endswith((".JPG", ".jpg", ".jpeg", ".png"))
        ]

total = 0
for disease, files in dict_disease.items():
    print(f"{disease}: {len(files)} images")
    total += len(files)
# assert total == sum(
#     [
#         len(os.listdir(dataset_path / f))
#         for f in os.listdir(dataset_path)
#         if f != "Background_without_leaves"
#     ]
# ), f"{total} != {sum([len(os.listdir(dataset_path / f)) for f in os.listdir(dataset_path) if f != 'Background_without_leaves'])}"

print(f"Total: {total} images")
print("number of classes:", len(dict_disease))

plt.bar(dict_disease.keys(), [len(dict_disease[key]) for key in dict_disease])
plt.xticks(rotation=90)
plt.show()

# downsampling the healthy class
healthy = dict_disease["healthy"]
np.random.shuffle(healthy)
# dict_disease["healthy"] = healthy[:len(dict_disease["healthy"])//3]

plt.bar(dict_disease.keys(), [len(dict_disease[key]) for key in dict_disease])
plt.xticks(rotation=90)
plt.show()

save_path = "../scripts_dev/plantvillage"
if not os.path.exists(save_path):
    os.makedirs(save_path)

with open(save_path + f"/dict_{label_to_use}.pkl", "wb") as f:
    pkl.dump(dict_disease, f)

# %% Preprocessing images

dict_path = f"../scripts_dev/plantvillage/dict_{label_to_use}.pkl"
prepro_path = "../datasets/PlantVillage/Preprocessed"

dict_prepro_path = "../scripts_dev/plantvillage"
dict_prepro = {}
if not os.path.exists(prepro_path):
    os.makedirs(prepro_path)

with open(dict_path, "rb") as f:
    dict_disease = pkl.load(f)

# check how many images are in the dict
# tot = 0
# for disease, files in dict_disease.items():
#     print(f"{disease}: {len(files)} images")
#     tot += len(files)
# print(f"Total: {tot} images") # 54305 images

for disease, files in dict_disease.items():
    print(f"Preprocessing and saving {disease} images")
    disease_path = os.path.join(prepro_path, disease)
    os.makedirs(disease_path, exist_ok=True)

    for file in tqdm.tqdm(files):
        img = Image.open(file)
        assert img.size[0] == img.size[1], f"Image {file} is not square"
        assert img.size[0] == 256, f"Image {file} is not 256x256"

        # resize image to 3x224x224
        img = img.resize((224, 224))
        img = np.array(img)
        # Normalize image between 0 and 255 and save it
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = Image.fromarray((img * 255).astype(np.uint8))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(os.path.join(disease_path, file.split('/')[-2].split('__')[0] + '_' + os.path.basename(file).split(".")[0] + f".JPG"))
        assert np.min(img) == 0, f"Min value: {np.min(img)}"
        assert np.max(img) == 255, f"Max value: {np.max(img)}"
        img.close()

    dict_prepro[disease] = [
        os.path.join(disease_path, os.path.basename(file)) for file in files
    ]

assert len(dict_prepro) == len(
    dict_disease
), f"Number of labels: {len(dict_prepro)} != {len(dict_disease)}"

assert all(
    [len(dict_prepro[key]) == len(dict_disease[key]) for key in dict_prepro]
), "Number of images is different"

total = [len(dict_prepro[key]) for key in dict_prepro]
print(f"Total: {sum(total)} images") # 54305 images

with open(dict_prepro_path + f"/dict_{label_to_use}_prepro.pkl", "wb") as f:
    pkl.dump(dict_prepro, f)
# %%
