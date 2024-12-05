<!--
 * @Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
 * @Date: 2023-12-05 15:46:49
 * @LastEditors: qiuyi.ye qiuyi.ye@maestrosurgical.com
 * @LastEditTime: 2024-12-03 15:08:29
 * @FilePath: /xushaokang/Single_AI_Registration/readme.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
# Title
单椎体AI配准算法模型

### Environment
- Linux (Recommend),
- Torch   1.13.1+cu116 
(Follow the instructions in [pytorch.org](http://pytorch.org) for your current setup)
- Cuda   11.6


~~~
git clone http://172.16.90.181/DefaultCollection/MaestroAlgo/_git/MaestroAlgoImgRegisDL
cd MaestroAlgoImgRegisDL/Single_AI_Registration2.0
pip install -r requirements.txt     # 运行之前需要先安装requirements.txt中的依赖包
（注，也可在可在registration_env环境下直接执行代码）
~~~

### 相关模型权重文件拿取地址
~~~
地址：/home/ps/xushaokang/Datasets/checkpoint_model  // A100机器中
- AP_X-ray_netG_A2B.pth 、LAT_X-ray_netG_A2B.pth    // 风格迁移模型
- Pose_Net_Single_m1_big.pth  // 模型1权重文件
- Pose_Net_Single_m2_big.pth  // 模型2权重文件
~~~

### 单椎体数据集放置路径(均在A100中)
~~~
/home/ps/xushaokang/Datasets/CT-Lumbar  #源CT数据(训练用的)
/home/ps/xushaokang/Datasets/CT-test    #源CT数据(测试用的)
/home/ps/xushaokang/Datasets/ctd-test   #对应上述测试CT的各个单椎体CT中心点文件
/home/ps/xushaokang/Datasets/ctd    #对应上述训练CT的各个单椎体CT中心点文件

/home/ps/xushaokang/Datasets/ctd-test.txt   #对应上述CT的各个单椎体CT中心点文件路径
/home/ps/xushaokang/Datasets/ctd-train.txt  #对应上述CT的各个单椎体CT中心点文件路径

~~~


### 投影DRR部分
- 目前所用的DRR投影算法工具均在./DRR_modules/文件夹里，包括执行代码的.py文件和编译好的.so文件（window和Linux两个）。
- 目前的DRR投影里面的一些计算过程已跟原始的差别较大，建议就用这版的投影。
- 如到新环境，需重新编译SiddonGpuPy.cpython-38-x86_64-linux-gnu.so时，具体可参考./MaestroAlgoImgRegisDL/DRR_Proj_Pytho/readme.md。

### 椎体检测与命名
- 目前所用的椎体检测与命名工具均来自于（MaestroAlgoXrayImageDetection）中，主要用于./datasets中；
- 主要调用执行的代码在./detect.py以及./location.py文件中，
- 涉及到一些库以及环境问题具体可参考./MaestroAlgoXrayImageDetection/两个算法中的readme.md。


### 算法参数

- 算法设计的超参数基本均在./Parameters.py文件中，包括训练参数和推理参数。
- 请在训练或推理前，根据需要自行调整参数（参数作用请参考各自help）。
  
### 训练
运行train.py文件，即可开始训练模型。
模型文件保存至/checkpoint_model/Pose_Net_Single.pth
~~~
python train.py
~~~

### 推理
运行test.py文件，即可进行模型推理。推理的相关参数可在文件中自行调整
~~~
python test.py
~~~
