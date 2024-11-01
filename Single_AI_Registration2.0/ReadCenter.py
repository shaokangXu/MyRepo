'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-12-05 10:05:16
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-09-23 17:30:55
FilePath: /xushaokang/Single_AI_Registration/Center.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd
import os
from io import StringIO
from collections import OrderedDict


Verdict={18:"centerT11",
      19:"centerT12",
      20:"centerL1",
      21:"centerL2",
      22:"centerL3",
      23:"centerL4",
      24:"centerL5",
      25:"centerL6",
      26:"centerSa"}
# read txt file and extra center coord
#txt_path="/home/ps/xushaokang/Single_Dataset/Vertebral_data/"


def ReadCenter(ctd_path,flagZ):   
    with open(ctd_path, 'r') as f:
        data = f.read()
    split_data = data.split('\n\n')

    if(len(split_data)<9):
        label_data = split_data[3]
    else:
        label_data = split_data[7]
    # 提取数据表格
    label_table = pd.read_csv(StringIO(label_data), sep='\t', header=1)
    # 剔除表头中的空格
    label_table.rename(columns=lambda x: x.strip(), inplace=True)
    # 剔除unnamed列
    unnamed_cols = [col for col in label_table.columns if 'Unnamed:' in col]
    label_table = label_table.drop(unnamed_cols, axis=1)
    filtered_df = label_table[label_table['Name'].str.contains('center')]


    #VertebralNameList=[]
    VertebralNameDict={}
    for index, row in filtered_df.iterrows():
        for i in range(len(row['Name'].strip().replace(" ", "")) - 1, -1, -1):
            if row['Name'].strip().replace(" ", "")[i] == 'r':
                # 遇到'r'或者空格时，返回该索引之后的部分
                key=row['Name'].strip().replace(" ", "")[i + 1:]
                #VertebralNameList.append (row['Name'].strip().replace(" ", "")[i + 1:])
                VertebralNameDict[key]=[row['X1'],row['Y1'],row['Z1']]
                break
    """ for key, value in VertebralNameDict.items():
        print(f"{key}: {value}") """
    #去点Sa
    if "Sa" in VertebralNameDict:
        del VertebralNameDict['Sa']
    # 根据Z点排序
    VertebralNameDictSort= sorted(VertebralNameDict.items(), key=lambda item: item[1][2], reverse=True)
    #保留投影中心Z轴附近的椎体信息
    result = {}
    for i, (key, value) in enumerate(VertebralNameDictSort):
        if value[2] <= flagZ:
            # 找到目标值位置时，确定保留上下范围
            start_index = max(i - 2, 0)  # 确保不会超出上界
            end_index = min(i + 2, len(VertebralNameDictSort))  # 确保不会超出下界
            
            for j in range(start_index, end_index):
                result[VertebralNameDictSort[j][0]] = VertebralNameDictSort[j][1]
            break
        if (i == len(VertebralNameDictSort) - 1) and value[2] > flagZ:
            # 找到目标值位置时，确定保留上下范围
            start_index = max(i - 1, 0)  # 确保不会超出上界
            end_index = len(VertebralNameDictSort)  # 确保不会超出下界
            for j in range(start_index, end_index):
                result[VertebralNameDictSort[j][0]] = VertebralNameDictSort[j][1]
            break

    result=dict(result)     
    return result



    """  info_list=[]
    # 提取信息并存入列表
    for index, row in filtered_df.iterrows():
        if (row['Name'].strip().replace(" ", "")==Verdict[Verbral]):          
            info_list.append(row['X1'])
            info_list.append(row['Y1'])
            info_list.append(row['Z1'])
            return info_list,VertebralNameList """


    
    




    


