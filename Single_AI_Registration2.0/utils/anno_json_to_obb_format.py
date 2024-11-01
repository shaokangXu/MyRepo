import codecs
from glob import glob
import os
import json

classes = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7',
            'T1', 'T2', 'T3', 'T4', 'T5', 'T6',
           'T7', 'T8', 'T9', 'T10', 'T11', 'T12',
           'L1', 'L2', 'L3', 'L4', 'L5', 'L6']


if __name__ == '__main__':

    # data_root = "/workspace/data/xRay_data"
    data_root = "/workspace/data/c-arm_x_ray/json_data"
    # out_dir = "/workspace/data/obb_detection_data/data/labelTxt"
    out_dir = "/workspace/data/obb_detection_data/c-arm_data/labelTxt"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    json_files = sorted(glob(data_root + "/*.json"))
    for json_path in json_files:
        sub_file_name = os.path.basename(json_path).split(".")[0] + ".txt"
        print(sub_file_name)
        output_dir = os.path.join(out_dir, sub_file_name)
        with open(json_path, 'r') as load_f:
            load_dict = json.load(load_f)
            nums_verts = len(load_dict)
            with codecs.open(output_dir, 'w', 'utf-8') as f_out:
                for x in range(nums_verts):
                    # if get < 0 , must be set is 0
                    if load_dict[x]['point1_x'] < 0:
                        load_dict[x]['point1_x'] = 0
                    if load_dict[x]['point1_y'] < 0:
                        load_dict[x]['point1_y'] = 0
                    if load_dict[x]['point2_x'] < 0:
                        load_dict[x]['point2_x'] = 0
                    if load_dict[x]['point2_y'] < 0:
                        load_dict[x]['point2_y'] = 0
                    if load_dict[x]['point3_x'] < 0:
                        load_dict[x]['point3_x'] = 0
                    if load_dict[x]['point3_y'] < 0:
                        load_dict[x]['point3_y'] = 0
                    if load_dict[x]['point4_x'] < 0:
                        load_dict[x]['point4_x'] = 0
                    if load_dict[x]['point4_y'] < 0:
                        load_dict[x]['point4_y'] = 0

                    out_poly = [load_dict[x]['point1_x'], load_dict[x]['point1_y'],
                                load_dict[x]['point2_x'], load_dict[x]['point2_y'],
                                load_dict[x]['point3_x'], load_dict[x]['point3_y'],
                                load_dict[x]['point4_x'], load_dict[x]['point4_y']
                                ]

                    outline = ' '.join(list(map(str, out_poly)))
                    # outline = outline + ' ' + classes[load_dict[x]['verte'] - 1] + ' ' + '0'
                    outline = outline + ' ' + 'vertebrae' + ' ' + '0'
                    print(outline)
                    f_out.write(outline + '\n')


