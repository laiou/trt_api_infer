#include "yololayer.h"
#include "utils.h"
#include <assert.h>

using namespace Yolo;

namespace nvinfer1
{
    YoloLayerPlugin::YoloLayerPlugin()
    {
        mClassCount = CLASS_NUM;
        mYoloKernel.clear();
        mYoloKernel.push_back(yolo1);
        mYoloKernel.push_back(yolo2);
        mYoloKernel.push_back(yolo3);

        mKernelCount = mYoloKernel.size();
    }
    
    YoloLayerPlugin::~YoloLayerPlugin()
    {
    }

    // create the plugin at runtime from a byte stream
    YoloLayerPlugin::YoloLayerPlugin(const void* data, size_t length)
    {
        using namespace Tn;
        const char *d = reinterpret_cast<const char *>(data), *a = d;
        read(d, mClassCount);
        read(d, mThreadCount);
        read(d, mKernelCount);
        mYoloKernel.resize(mKernelCount);
        auto kernelSize = mKernelCount*sizeof(YoloKernel);
        memcpy(mYoloKernel.data(),d,kernelSize);
        d += kernelSize;

        assert(d == a + length);
    }

    void YoloLayerPlugin::serialize(void* buffer) const
    {
        using namespace Tn;
        char* d = static_cast<char*>(buffer), *a = d;
        write(d, mClassCount);
        write(d, mThreadCount);
        write(d, mKernelCount);
        auto kernelSize = mKernelCount*sizeof(YoloKernel);
        memcpy(d,mYoloKernel.data(),kernelSize);
        d += kernelSize;

        assert(d == a + getSerializationSize());
    }
    
    size_t YoloLayerPlugin::getSerializationSize() const
    {  
        return sizeof(mClassCount) + sizeof(mThreadCount) + sizeof(mKernelCount)  + sizeof(Yolo::YoloKernel) * mYoloKernel.size();
    }

    int YoloLayerPlugin::initialize()
    { 
        return 0;
    }
    
    Dims YoloLayerPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
    {
        //output the result to channel
        int totalsize = MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);

        return Dims3(totalsize + 1, 1, 1);
    }

    // Set plugin namespace
    void YoloLayerPlugin::setPluginNamespace(const char* pluginNamespace)
    {
        mPluginNamespace = pluginNamespace;
    }

    const char* YoloLayerPlugin::getPluginNamespace() const
    {
        return mPluginNamespace;
    }

    // Return the DataType of the plugin output at the requested index
    DataType YoloLayerPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
    {
        return DataType::kFLOAT;
    }

    // Return true if output tensor is broadcast across a batch.
    bool YoloLayerPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
    {
        return false;
    }

    // Return true if plugin can use input that is broadcast across batch without replication.
    bool YoloLayerPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
    {
        return false;
    }

    void YoloLayerPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput)
    {
    }

    // Attach the plugin object to an execution context and grant the plugin the access to some context resource.
    void YoloLayerPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
    {
    }

    // Detach the plugin object from its execution context.
    void YoloLayerPlugin::detachFromContext() {}

    const char* YoloLayerPlugin::getPluginType() const
    {
        return "YoloLayer_TRT";
    }

    const char* YoloLayerPlugin::getPluginVersion() const
    {
        return "1";
    }

    void YoloLayerPlugin::destroy()
    {
        delete this;
    }

    // Clone the plugin
    IPluginV2IOExt* YoloLayerPlugin::clone() const
    {
        YoloLayerPlugin *p = new YoloLayerPlugin();
        p->setPluginNamespace(mPluginNamespace);
        return p;
    }

    __device__ float Logist(float data){ return 1.0f / (1.0f + expf(-data)); };
    //执行相应的操作
    __global__ void CalDetection(const float *input, float *output,int noElements, 
            int yoloWidth,int yoloHeight,const float anchors[CHECK_COUNT*2],int classes,int outputElem) {
        //计算线程id，这里每一个线程处理一个grid上的预测（整个batch的数据一共是有yolo.width*yolo.height*batchSize个grid的）
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= noElements) return;
        //统计当前yolo层图像划分的grid个数
        int total_grid = yoloWidth * yoloHeight;
        int bnIdx = idx / total_grid;
        //获取当前的线程偏移量，也就是当前在某张图片的哪一个grid上
        idx = idx - total_grid*bnIdx;
        int info_len_i = 5 + classes;
        //定位当前输入的位置，也就是一个batch中的第几张图数据的起始位置
        const float* curInput = input + bnIdx * (info_len_i * total_grid * CHECK_COUNT);
        //遍历不同的anchor上的数据
        for (int k = 0; k < 3; ++k) {
            int class_id = 0;
            float max_cls_prob = 0.0;
            //然后遍历每一个预测值
            for (int i = 5; i < info_len_i; ++i) {
            //通过logist进行计算类别置信度
                float p = Logist(curInput[idx + k * info_len_i * total_grid + i * total_grid]);
                //更新max_cla_prob
                if (p > max_cls_prob) {
                    max_cls_prob = p;
                    //之所以要减去5的原因前面一开始i是从5开始的，这里需要计算对应的class_id
                    class_id = i - 5;
                }
            }
            //然后在对相应的box输出进行logist转换，得到box的阈值
            float box_prob = Logist(curInput[idx + k * info_len_i * total_grid + 4 * total_grid]);
            //如果类别置信度或者box置信度小于阈值，就可以放弃了
            if (max_cls_prob < IGNORE_THRESH || box_prob < IGNORE_THRESH) continue;
            //定位对应的保存输出的位置
            float *res_count = output + bnIdx*outputElem;
            //然后将相应的输出值抽取到output中
            int count = (int)atomicAdd(res_count, 1);
            if (count >= MAX_OUTPUT_BBOX_COUNT) return;
            char* data = (char * )res_count + sizeof(float) + count*sizeof(Detection);
            Detection* det =  (Detection*)(data);

            int row = idx / yoloWidth;
            int col = idx % yoloWidth;

            //Location
            det->bbox[0] = (col + Logist(curInput[idx + k * info_len_i * total_grid + 0 * total_grid])) * INPUT_W / yoloWidth;
            det->bbox[1] = (row + Logist(curInput[idx + k * info_len_i * total_grid + 1 * total_grid])) * INPUT_H / yoloHeight;
            det->bbox[2] = expf(curInput[idx + k * info_len_i * total_grid + 2 * total_grid]) * anchors[2*k];
            det->bbox[3] = expf(curInput[idx + k * info_len_i * total_grid + 3 * total_grid]) * anchors[2*k + 1];
            det->det_confidence = box_prob;
            det->class_id = class_id;
            det->class_confidence = max_cls_prob;
        }
    }

    void YoloLayerPlugin::forwardGpu(const float *const * inputs, float* output, cudaStream_t stream, int batchSize) {

        void* devAnchor;
        size_t AnchorLen = sizeof(float)* CHECK_COUNT*2;
        //分配相应的内存
        CUDA_CHECK(cudaMalloc(&devAnchor,AnchorLen));
        //输出元素的数量
        int outputElem = 1 + MAX_OUTPUT_BBOX_COUNT * sizeof(Detection) / sizeof(float);

        //循环给每一个batch的数据的输出分配内存
        for(int idx = 0 ; idx < batchSize; ++idx) {
            CUDA_CHECK(cudaMemset(output + idx*outputElem, 0, sizeof(float)));
        }
        int numElem = 0;
        //循环处理每一个yolo层
        for (unsigned int i = 0;i< mYoloKernel.size();++i)
        {
            const auto& yolo = mYoloKernel[i];
            //计算不同层输入的数据量，实际上也就是一个batchsize上全部的grid数量
            numElem = yolo.width*yolo.height*batchSize;
            //判断输入数据是否超出默认的线程数量
            if (numElem < mThreadCount)
                mThreadCount = numElem;
            //将相应的数据从主机内存复制到设备内存，也就是将当前的yolo层的anchor数据从yolo.anchors传输到devAnchor
            CUDA_CHECK(cudaMemcpy(devAnchor, yolo.anchors, AnchorLen, cudaMemcpyHostToDevice));
            //进行具体操作的核函数，整个yolo层完成的就是提取每一个grid下面不同尺度的anchor上的预测框，判断是否超过阈值，然后将相应满足
            //条件的框计算对应的框的置信度和类别置信度，然后将结果存入output中
            CalDetection<<< (yolo.width*yolo.height*batchSize + mThreadCount - 1) / mThreadCount, mThreadCount>>>
                (inputs[i],output, numElem, yolo.width, yolo.height, (float *)devAnchor, mClassCount ,outputElem);
        }

        CUDA_CHECK(cudaFree(devAnchor));
    }

    //这里看一下具体的实现yolo层的操作
    int YoloLayerPlugin::enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream)
    {
        //assert(batchSize == 1);
        //GPU
        //CUDA_CHECK(cudaStreamSynchronize(stream));
        //具体的yolo层的gpu前向传播
        forwardGpu((const float *const *)inputs, (float*)outputs[0], stream, batchSize);

        return 0;
    }

    PluginFieldCollection YoloPluginCreator::mFC{};
    std::vector<PluginField> YoloPluginCreator::mPluginAttributes;

    YoloPluginCreator::YoloPluginCreator()
    {
        mPluginAttributes.clear();

        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* YoloPluginCreator::getPluginName() const
    {
            return "YoloLayer_TRT";
    }

    const char* YoloPluginCreator::getPluginVersion() const
    {
            return "1";
    }

    const PluginFieldCollection* YoloPluginCreator::getFieldNames()
    {
            return &mFC;
    }

    IPluginV2IOExt* YoloPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
    {
        YoloLayerPlugin* obj = new YoloLayerPlugin();
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

    IPluginV2IOExt* YoloPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
    {
        // This object will be deleted when the network is destroyed, which will
        // call MishPlugin::destroy()
        YoloLayerPlugin* obj = new YoloLayerPlugin(serialData, serialLength);
        obj->setPluginNamespace(mNamespace.c_str());
        return obj;
    }

}
