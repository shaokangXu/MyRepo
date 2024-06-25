<!--
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2023-12-05 15:46:49
 * @LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @LastEditTime: 2023-12-14 11:42:17
 * @FilePath: /xushaokang/Single_AI_Registration/readme.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# Title
单椎体AI配准算法模型

### 单椎体数据集放置路径(均在A100中)

~~~
/home/ps/xushaokang/Single_Dataset/CT   #源CT数据
/home/ps/xushaokang/Single_Dataset/CT_seg    #对应上述CT的标注分割数据
/home/ps/xushaokang/Single_Dataset/Single    #对应上述CT的各个单椎体CT数据
/home/ps/xushaokang/Single_Dataset/Vertebral_data   #对应上述CT的各个单椎体CT所标注的质心等相关数据
/home/ps/xushaokang/Single_Dataset/ct_Single_train.txt   #对应上述CT的各个用于训练的单椎体CT路径文件
/home/ps/xushaokang/Single_Dataset/ct_Single_test.txt   #对应上述CT的各个用于测试的单椎体CT路径文件
~~~


### 训练
运行train.py文件，即可开始训练模型。训练的相关参数可在文件中自行调整(如学习率、batch_size等)
模型文件保存至/checkpoint_model/Pose_Net_Single.pth
~~~
python train.py
~~~

### 推理
运行test.py文件，即可进行模型推理。推理的相关参数可在文件中自行调整
~~~
python test.py
~~~
