# trt_api_infer
### This repo is using tensorrt api infer yolov3 model，now can support int8 fp16 and fp32 precision to infer.
------
### Test Environment:
- Ubuntu18.04LTS
- TensorRT7.0
- Cuda 10.0
------
### Start Using:
&ensp;&ensp;Get the [pre-prepared model](https://pan.baidu.com/s/1XNRdyEPsD8_MA5wlRV1rNQ)，the key is 9i1r, and place it in model file.

------
&ensp;&ensp;Building this repo:
```
cd build
cmake ..
make
```
------
&ensp;&ensp;Start run:
```
cd ..
cd ./infer
sudo ./project -getengine -int8
sudo ./projet -inference ../test_data
```
------
&ensp;&ensp;Using -int8,-fp16,-fp32 to select the appropriate inference precision,and if chosed int8,Prepare appropriate calibration data and place it in calid_data file.Additional functionality can be extended by referring to the TensorRT plugin implementation.



