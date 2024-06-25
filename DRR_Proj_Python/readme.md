# Scope
通过CUDA源码编译DRR投影的Python库文件。

# Build
## Windows
打开`DrrPythonLib/SRC/SiddonGpu`项目下的`sln`项目，通过Visual Studio编译静态链接库文件，生成文件在`CUDA_DigitallyReconstructedRadiographs-master\SiddonPythonModule\lib`目录中，编译完成后会自动将头文件拷贝到`CUDA_DigitallyReconstructedRadiographs-master\SiddonPythonModule\include`目录中，完成后可检查以下两个文件是否更新确认：
~~~
CUDA_DigitallyReconstructedRadiographs-master\SiddonPythonModule\include\siddon_class.cuh
CUDA_DigitallyReconstructedRadiographs-master\SiddonPythonModule\lib\SiddonGpu.lib
~~~

在`CUDA_DigitallyReconstructedRadiographs-master\SiddonPythonModule`目录下编译python库文件：
~~~
cd CUDA_DigitallyReconstructedRadiographs-master\SiddonPythonModule
python setup.py build_ext --inplace
~~~

将`CUDA_DigitallyReconstructedRadiographs-master\SiddonPythonModule`目录下生成的`*.pyd`文件拷贝到`modules`文件夹下。

## Linux
~~~
./update.sh
~~~