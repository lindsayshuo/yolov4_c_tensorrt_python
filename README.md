# yolov4_c_tensorrt_python
convert yolov4 to TensorRT engine，use C++  infer code to make Python  api
具体的配置信息可以参考我的另外一个博客
	
	https://blog.csdn.net/weixin_43269994/article/details/117814951?spm=1001.2014.3001.5501
github链接后续放出

# YOLOv4

配置CMakeLists.txt,以下为需要检查并修改的地方

```bash
#cuda
include_directories(/usr/local/cuda-11.1/include)
link_directories(/usr/local/cuda-11.1/lib64)

#tensorrt
include_directories(/home/username/TensorRT-7.2.2.3/include)
link_directories(/home/username/TensorRT-7.2.2.3/lib)

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu ${PROJECT_SOURCE_DIR}/mish.cu)

cuda_add_library(yolov4_trt SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu ${PROJECT_SOURCE_DIR}/mish.cu ${PROJECT_SOURCE_DIR}/yolov4_lib.cpp)
```

封装

1. yolov4_lib.h
```c++
#include "cuda_runtime_api.h"
extern "C"
{
//int add(int a, int b);

void * yolov4_trt_create(const char * engine_name);
const char * yolov4_trt_detect(void *h, cv::Mat &img, float threshold);
void yolov4_trt_destroy(void *h);
}
```

2. yolov4_lib.cpp
* **yolov4_trt_create**()
  
    ```c++
    void * yolov4_trt_create(const char * engine_name)
    {   //std::cout<<'111111111111111111'<<std::endl;
        cudaSetDevice(DEVICE);
        char *trtModelStream{nullptr};
        size_t size{0};
        Yolov4TRTContext* trt_ctx = NULL;
    
        trt_ctx = new Yolov4TRTContext();
        std::ifstream file("./yolov4.engine", std::ios::binary);
        printf("yolov4_trt_create ... %s\n",engine_name);
    
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            std::cout<<size<<std::endl;
            file.read(trtModelStream, size);
            file.close();
        }else
            return NULL;
    
        trt_ctx->data = new float[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
        trt_ctx->prob = new float[BATCH_SIZE * OUTPUT_SIZE];
        trt_ctx->runtime = createInferRuntime(gLogger);
        assert(trt_ctx->runtime != nullptr);
        printf("yolov4_trt_create  cuda engine... \n");
    
        trt_ctx->engine = trt_ctx->runtime->deserializeCudaEngine(trtModelStream, size);
        assert(trt_ctx->engine != nullptr);
    
        trt_ctx->exe_context = trt_ctx->engine->createExecutionContext();
        delete[] trtModelStream;
    
        assert(trt_ctx->engine->getNbBindings() == 2);
        trt_ctx->inputIndex = trt_ctx->engine->getBindingIndex(INPUT_BLOB_NAME);
        trt_ctx->outputIndex = trt_ctx->en574gine->getBindingIndex(OUTPUT_BLOB_NAME);
        assert(trt_ctx->inputIndex == 0);
        assert(trt_ctx->outputIndex == 1);
    
        printf("yolov4_trt_create  buffer ... \n");
        CUDA_CHECK(cudaMalloc(&trt_ctx->buffers[trt_ctx->inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&trt_ctx->buffers[trt_ctx->outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    
        printf("yolov4_trt_create  stream ... \n");
        CUDA_CHECK(cudaStreamCreate(&trt_ctx->cuda_stream));
    
        printf("yolov4_trt_create  done ... \n");
    
        return (void *)trt_ctx;
    }
    ```
    
* **yolov4_trt_detect**()

    ```c++
    const char * yolov4_trt_detect(void *h, cv::Mat &img, float threshold){
        int times;
        int str_len;
        int num_det;
        int delay_preprocess;
        Yolov4TRTContext *trt_ctx;
        trt_ctx = (Yolov4TRTContext *)h;
        
        trt_ctx->result_json_str[0] = 0;
        if (img.empty()) return trt_ctx->result_json_str;
    
        auto start0 = std::chrono::system_clock::now();
        cv::Mat pr_img=preprocess_img(img);
        
        int i=0;
        for(int f=0;f<INPUT_H;++f){
           uchar* uc_pixel = pr_img.data + f * pr_img.step;
           for (int c=0;c<INPUT_W;++c){
           trt_ctx->data[i]=(float)uc_pixel[2]/255.0;
           trt_ctx->data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
           trt_ctx->data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
           uc_pixel += 3;
           ++i;
           }
        }
        auto end0 = std::chrono::system_clock::now();
         
        auto start = std::chrono::system_clock::now();
        doInference(*trt_ctx->exe_context, trt_ctx->data, trt_ctx->prob, BATCH_SIZE);
        auto end = std::chrono::system_clock::now();
        times=std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        std::cout <<times << "ms" << std::endl;
        
        
        std::vector<std::vector<Yolo::Detection>> batch_res(1);
        auto& res = batch_res[0];
        nms(res, &trt_ctx->prob[0]);
        num_det=(int)res.size();
        sprintf(trt_ctx->result_json_str,"{" "delay_preprocess:%d,""times:%d,"  "num_det:%d," "object:""[",delay_preprocess,times,num_det);
        //sprintf(trt_ctx->result_json_str,"object:");
        str_len = strlen(trt_ctx->result_json_str);
        i=0;
        for(i = 0 ; i < res.size(); i++){
            
            cv::Rect r = get_rect(img, res[i].bbox);
            	    
            //cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            //cv::putText(img, std::to_string((int)res[i].class_id), cv::Point(r.x, r.y - 1),
    //cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            //sprintf(trt_ctx->result_json_str,"object_class_id:%d", (int)res[i].class_id);
            int class_id;
            int x1, y1, x2, y2;
            x1 = r.x;
            y1 = r.y;
            x2 = r.x + r.width;
            y2 = r.y + r.height;
            //sprintf(trt_ctx->result_json_str + str_len, "object:(%d,%d,%d,%d)", x1, y1, x2, y2);
            //str_len = strlen(trt_ctx->result_json_str);
            class_id=(int)res[i].class_id;
           
             if (0 == i){
                sprintf(trt_ctx->result_json_str + str_len, "(%d,%d,%d,%d,%d)", class_id, x1, y1, x2, y2);
            }else {
                sprintf(trt_ctx->result_json_str + str_len, ",(%d,%d,%d,%d,%d)", class_id, x1, y1, x2, y2);
            }
            str_len = strlen(trt_ctx->result_json_str);
    
            if (str_len >= 16300)
                break;
         }
         sprintf(trt_ctx->result_json_str + str_len, "]}");
        //sprintf(trt_ctx->result_json_str,"num_det:%d", (int) res.size());
    
         return trt_ctx->result_json_str;
    
    
    }
    ```

    

* **yolov4_trt_destroy**()

    ```c++
    void  yolov4_trt_destroy(void *h){
    
        Yolov4TRTContext *trt_ctx;
        trt_ctx = (Yolov4TRTContext *)h;
        cudaStreamDestroy(trt_ctx->cuda_stream);
        CUDA_CHECK(cudaFree(trt_ctx->buffers[trt_ctx->inputIndex]));
        CUDA_CHECK(cudaFree(trt_ctx->buffers[trt_ctx->outputIndex]));
       
        
        trt_ctx->exe_context->destroy();
        trt_ctx->engine->destroy();
        trt_ctx->runtime->destroy();
    
    }
    ```

生成引擎命令:

```
mkdir build
cd build
cmake ..
make
sudo ./yolov4 -s #sudo ./yolov4 -s ../yolov4.wts ../yolov4.engine s
```

//**yolov4_trt_py_module.cpp**

```c++
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <Python.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "./yolov4_lib.h"
#include "pyboostcvconverter/pyboostcvconverter.hpp"
#include <boost/python.hpp>
using namespace cv;
using namespace boost::python;

static PyObject * mpyCreate(PyObject *self,  PyObject *args)
{
    char *engine_path = NULL;
    void *trt_engine = NULL;
    
    if (!PyArg_ParseTuple(args, "s", &engine_path)){
        return  Py_BuildValue("K", (unsigned long long)trt_engine);
    }

    trt_engine = yolov4_trt_create(engine_path);
    printf("create yolov4-trt , instance = %p\n", trt_engine);
    return Py_BuildValue("K", (unsigned long long)trt_engine);
}

static PyObject *mpyDetect(PyObject *self, PyObject *args)
{
    void *trt_engine = NULL;
    PyObject *ndArray = NULL;
    float conf_thresh = 0.45;
    const char *ret = NULL;
    unsigned long long v; 
    
    if (!PyArg_ParseTuple(args, "KOf", &v, &ndArray, &conf_thresh))
        return Py_BuildValue("s", "");
    Mat mat = pbcvt::fromNDArrayToMat(ndArray);
    trt_engine = (void *)v;
    //ret = yolov4_trt_detect(trt_engine, mat, conf_thresh);
    ret = yolov4_trt_detect(trt_engine,mat, conf_thresh);
    return Py_BuildValue("s", ret);
}

static PyObject * mPyDestroy(PyObject *self, PyObject *args)
{
    void *engine = NULL;
    unsigned long long v; 
    if (!PyArg_ParseTuple(args, "K", &v))
        return Py_BuildValue("");     
    printf("destroy engine , engine = %lu\n", v);
    engine = (void *)v;  
    yolov4_trt_destroy(engine);
    return Py_BuildValue("");
}

static PyMethodDef TRTYolov4MeThods[] = {
    {"create", mpyCreate, METH_VARARGS, "Create the engine."},
    {"detect", mpyDetect, METH_VARARGS, "use the engine to detect image"},    
    {"destroy", mPyDestroy, METH_VARARGS, "destroy the engine"},        
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef TRTYolov4Module = {
    PyModuleDef_HEAD_INIT,
    "TRTYolov4",     /* name of module */
    "",          /* module documentation, may be NULL */
    -1,          /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    TRTYolov4MeThods
};

PyMODINIT_FUNC PyInit_TRTYolov4(void) {
    printf("init module ... \n");
    return PyModule_Create(&TRTYolov4Module);
}

```

编写setup.py

```python
from setuptools import setup, Extension, find_packages
import distutils.command.clean
from torch.utils.cpp_extension import BuildExtension
import numpy as np
setup(
    name='TRTYolov4',
    version='1.0',
    author="lindsay",
    author_email="lindsayshuo@foxmail.com",
    url="lindsayshuo@foxmail.com",
    description='Python Package with Hello World C++ Extension',

    # Package info
    packages=find_packages(exclude=('a',)),
    zip_safe=False,

    ext_modules=[
        Extension(
            'TRTYolov4',
            sources=['pyboostcvconverter/pyboost_cv4_converter.cpp', 'yolov4_trt_py_module.cpp'],
            include_dirs=['/home/lindsay/anaconda3/lib/python3.8/site-packages/numpy/core/include/numpy',
                          '/usr/local/cuda-11.1/include/',
                          '/usr/local/include/',
                          '/home/lindsay/TensorRT-7.2.2.3/include',
                          '../include',
                          '/home/lindsay/Downloads/tensorrtx-master/yolov4/'
                         ],
            libraries=['gstvideo-1.0', 'yolov4_trt',  'opencv_features2d', 'opencv_flann',  'opencv_imgcodecs', 'opencv_imgproc', 'opencv_core', 'opencv_highgui', 'opencv_videoio',   "boost_python3"],
            library_dirs=[ './build','/home/lindsay/anaconda3/lib', '/usr/local/lib', '/home/lindsay/Downloads/tensorrtx-master/yolov4/build'],
            py_limited_api=True)
    ],
    include_dirs=[np.get_include()]
  
) 
```

测试//a.py

```python
import cv2
import TRTYolov4 as t

engine = t.create('./build/yolov4.engine')
img = cv2.imread('./samples/lius.jpg')
result = t.detect(engine, img, 0.45)
t.destroy(engine)
print(result)
```



