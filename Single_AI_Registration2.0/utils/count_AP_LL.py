import os
from glob import glob
import shutil

image_path = "/workspace/data/obb_detection_data/data/images"
label_path = "/workspace/data/obb_detection_data/data/labelTxt"
lateral_image_path = "/workspace/data/obb_detection_data/lateral_data/images"
lateral_label_path = "/workspace/data/obb_detection_data/lateral_data/labelTxt"
anterior_image_path = "/workspace/data/obb_detection_data/anterior_data/images"
anterior_label_path = "/workspace/data/obb_detection_data/anterior_data/labelTxt"

if not os.path.exists(lateral_label_path):
    os.makedirs(lateral_label_path)
if not os.path.exists(anterior_label_path):
    os.makedirs(anterior_label_path)

name = set()
image_files = sorted(glob(image_path + "/*.png"))
label_files = sorted(glob(label_path + "/*.txt"))
for image_file, label_file in zip(image_files, label_files):
    filename = os.path.basename(image_file)
    file_ext = filename.split(".")[0][-2:]
    fileid = filename.split(".")[0]
    assert fileid == os.path.basename(label_file).split(".")[0], "image mismatch label!"
    name.add(file_ext)

    if "L" in file_ext:
        # shutil.copy(image_file, os.path.join(lateral_image_path, filename))
        shutil.copy(label_file, os.path.join(lateral_label_path, fileid + ".txt"))
    if "A" in file_ext:
        # shutil.copy(image_file, os.path.join(anterior_image_path, filename))
        shutil.copy(label_file, os.path.join(anterior_label_path, fileid + ".txt"))

print(name)