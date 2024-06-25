import pandas as pd
import numpy as np
import os
from glob import glob
import json

v_dict = {
    1: 'C1', 2: 'C2', 3: 'C3', 4: 'C4', 5: 'C5', 6: 'C6', 7: 'C7',
    8: 'T1', 9: 'T2', 10: 'T3', 11: 'T4', 12: 'T5', 13: 'T6', 14: 'T7',
    15: 'T8', 16: 'T9', 17: 'T10', 18: 'T11', 19: 'T12', 20: 'L1',
    21: 'L2', 22: 'L3', 23: 'L4', 24: 'L5', 25: 'L6', 26: 'Sacrum',
    27: 'Cocc', 28: 'T13'
}
map_dict = {v: k for k, v in v_dict.items()}
print(map_dict)


root_path = "/workspace/data/20221125_No5_Xray_json"
save_path = "/workspace/data/xRay_data"

json_files = sorted(glob(root_path + "/*_json.json"))
json_files.append("/workspace/data/20221125_No5_Xray_json/via_project_template_new 0425_json (1).json")
json_files.append("/workspace/data/20221125_No5_Xray_json/via_project_template_new 0506_json (1).json")

for json_file in json_files:
    print(json_file)

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for key, values in data.items():
        saved_json = []
        filename = values['filename']
        if values['regions']:
            for region in values['regions']:
                points_x = list(region['shape_attributes']['all_points_x'])
                points_y = list(region['shape_attributes']['all_points_y'])
                verte = region['region_attributes']['verte']
                center_x = np.mean(points_x)
                center_y = np.mean(points_y)

                json_data = {
                    'point1_x': points_x[0],
                    'point1_y': points_y[0],
                    'point2_x': points_x[1],
                    'point2_y': points_y[1],
                    'point3_x': points_x[2],
                    'point3_y': points_y[2],
                    'point4_x': points_x[3],
                    'point4_y': points_y[3],
                    'center_x': center_x,
                    'center_y': center_y,
                    'verte': map_dict[verte]
                }

                print(json_data)
                saved_json.append(json_data)

        # 保存为json文件
        if saved_json:
            with open(os.path.join(save_path, f'{values["filename"].split(".")[0]}.json'), 'w') as f:
                json.dump(saved_json, f)

