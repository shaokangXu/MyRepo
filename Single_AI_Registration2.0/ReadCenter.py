'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-12-05 10:05:16
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-12-05 11:43:55
FilePath: /xushaokang/Single_AI_Registration/Center.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import os
from io import StringIO



dict={20:"centerL1",
      21:"centerL2",
      22:"centerL3",
      23:"centerL4",
      24:"centerL5",
      25:"centerL6",
      26:"centerSa"}
# read txt file and extra center coord
txt_path="/home/ps/xushaokang/Single_Dataset/Vertebral_data/"


def ReadCenter(CT_Name,Verbral):   
    with open(txt_path+CT_Name+"_0000.txt", 'r') as f:
        data = f.read()
    split_data = data.split('\n\n')

    label_data = split_data[7]
    # 提取数据表格
    label_table = pd.read_csv(StringIO(label_data), sep='\t', header=1)
    # 剔除表头中的空格
    label_table.rename(columns=lambda x: x.strip(), inplace=True)
    # 剔除unnamed列
    unnamed_cols = [col for col in label_table.columns if 'Unnamed:' in col]
    label_table = label_table.drop(unnamed_cols, axis=1)
    filtered_df = label_table[label_table['Name'].str.contains('center')]

    info_list=[]
    # 提取信息并存入列表
    for index, row in filtered_df.iterrows():
        if (row['Name'].strip()==dict[Verbral]):
            info_list.append(row['X1'])
            info_list.append(row['Y1'])
            info_list.append(row['Z1'])
            return info_list


    
    




    


