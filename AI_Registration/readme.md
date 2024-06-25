# Title
整椎体AI配准训练模型

### CT数据集放置路径(均在A100中)

~~~
/home/ps/xushaokang/Datasets/CT-train   #训练集   200
/home/ps/xushaokang/Datasets/CT-test    #测试集   40

~~~


### 训练
运行train.py文件，即可开始训练模型。训练的相关参数可在文件中自行调整(如)
~~~
python train.py
~~~

### 推理
运行test.py文件，即可进行模型推理。推理的相关参数可在文件中自行调整
~~~
python test.py
~~~
