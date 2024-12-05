'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-11-17 13:20:08
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-12-05 10:41:55
FilePath: /xushaokang/Single_AI_Registration/SaveResultData.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import pandas as pd 
def save_result(transform_parameters_delta,delta_Pose,i,time):

    #tensor to numpy
    error=abs(transform_parameters_delta-delta_Pose)[0].to("cpu").tolist()

    #++++++++++++++++++++++++++++++++++++++

    #------保存配准误差数据------
    # 读取现有的Excel文件
    file_path = 'Output/regis_data.xlsx'
    existing_data = pd.read_excel(file_path)

    Iter_data = {"CT": str(i)}
    all_data={}
    all_data.update(**Iter_data)        
    error_data = {
        "rx": (error[0]),
        "ry": (error[1]),
        "rz": (error[2]),
        "tx": (error[3]),
        "ty": (error[4]),
        "tz": (error[5]),
        "Time": (time),
        "R_error":((error[0]+error[1]+error[2])/3),
        "T_error":((error[3]+error[4]+error[5])/3),
        }
    all_data.update(**error_data)
    df_to_append = pd.DataFrame([all_data])

    #print(all_data)
    # 将新数据添加到现有数据后面
    updated_data = pd.concat([existing_data,df_to_append], ignore_index=True)
    updated_data.to_excel(file_path, index=False)
