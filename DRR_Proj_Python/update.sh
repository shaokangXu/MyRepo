cd CUDA_DigitallyReconstructedRadiographs-master/SiddonClassLib/
mkdir -p build
cd build
cmake ../src
make
cd ..
mkdir -p ../SiddonPythonModule/lib
cp build/libSiddonGpu.a ../SiddonPythonModule/lib/libSiddonGpu.a -f
cp src/SiddonLib/siddon_class.cuh ../SiddonPythonModule/include/siddon_class.cuh -f
cd ../SiddonPythonModule
python setup.py build_ext --inplace
echo $PWD
ls | grep *gnu.so | xargs -i cp -rf ./{} ../../modules/{}