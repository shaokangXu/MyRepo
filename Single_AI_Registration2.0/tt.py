'''
Author: qiuyi.ye qiuyi.ye@maestrosurgical.com
Date: 2024-09-23 16:24:26
LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
LastEditTime: 2024-09-23 17:19:53
FilePath: /xushaokang/Single_AI_Registration2.0_BN5/tt.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''



import random
from ReadCenter import ReadCenter
data='../Datasets/ctd.txt'
with open(data, 'r') as file:
            CT_num = len(file.readlines())
with open(data, 'r') as file:
# 使用 len() 函数获取CT数量
    lines = file.readlines()      

for j,line in enumerate(lines):
    print(line.strip())
    VertebralNameDict=ReadCenter(line.strip(),-750.3500138819218)  #获取CT中所有椎体质心
    print(VertebralNameDict)
    exit()