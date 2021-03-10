#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "utils.h"
#include "logging.h"
#include "yololayer.h"
#include "calibrator.h"

#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.4
#define BBOX_CONF_THRESH 0.5

using namespace nvinfer1;

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
//这里的Yolo::Detection实际上就是检测输出的值，也就是置信度加上4个坐标。。。
static const int DETECTION_SIZE = sizeof(Yolo::Detection) / sizeof(float);
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

cv::Rect get_rect(cv::Mat& img, float bbox[4]) {
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2]/2.f;
        r = bbox[0] + bbox[2]/2.f;
        t = bbox[1] - bbox[3]/2.f - (INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3]/2.f - (INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2]/2.f - (INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2]/2.f - (INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3]/2.f;
        b = bbox[1] + bbox[3]/2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r-l, b-t);
}
//计算两个框的iou
float iou(float lbox[4], float rbox[4]) {
    float interBox[] = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
}

bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) {
    return a.det_confidence > b.det_confidence;
}
//具体的nms实现
void nms(std::vector<Yolo::Detection>& res, float *output, float nms_thresh = NMS_THRESH) {

    std::map<float, std::vector<Yolo::Detection>> m;
    //遍历全部的框
    for (int i = 0; i < output[0] && i < 1000; i++) {
        //预测的输出包括4个坐标，类别id，和类别置信度以及框的置信度
        if (output[1 + 7 * i + 4] <= BBOX_CONF_THRESH) continue;
        Yolo::Detection det;
        //将相应的output中的值复制到det
        memcpy(&det, &output[1 + 7 * i], 7 * sizeof(float));
        //判断map中是不是不存在det.class_id，如果不存在的话就利用emplace来将相应的det.class_id扩充到容器里
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo::Detection>());
        //通过push_back将det元素添加到 m[det.class_id]
        m[det.class_id].push_back(det);
    }
    //到这里实际上就将全部框根据预测类别分别统计到了一起
    //然后就是遍历m中的全部数据，也就是一个类别一个类别的来遍历
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        //到了某一个类别的位置，通过dets提取相应的预测值，毕竟这里的it相当于是一个键值对
        auto& dets = it->second;
        //对相应的dets中的预测框根据cmp进行排序，默认是升序，这里通过cmp来排序，实际上也就是根据相应的框的置信度来判断
        std::sort(dets.begin(), dets.end(), cmp);
        //循环遍历相应的dets中的每个数据
        for (size_t m = 0; m < dets.size(); ++m) {
            //对于dets中的每一个框
            auto& item = dets[m];
            //将相应的框的坐标添加到res中，实际上就是先将排序后的第一个框添加到res里面
            res.push_back(item);
            //然后循环遍历第m个框后面的框
            for (size_t n = m + 1; n < dets.size(); ++n) {
                //计算当前第m个框和后面每一个框的iou，判断是否超过阈值
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    //如果超过的话，将对应的第n个框删掉
                    dets.erase(dets.begin()+n);
                    //然后更新n的值
                    --n;
                }
            }
        }
    }
}

// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
//从相应文件中加载权重
std::map<std::string, Weights> loadWeights(const std::string file) {
    
    std::cout << "Loading weights: " << file << std::endl;
    //创建存储权重的std::map实例
    std::map<std::string, Weights> weightMap;

    // Open weights file
    //std::ifstream是c++中用于操作文件的输入流类，以输入的方式打开文件
    //这里的input相当于一个句柄
    std::ifstream input(file);
    //判断文件是否打开
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    //实际上就是提取文件中第一个int值，记录了整个权重文件有多少行
    int32_t count;
    //从文件中输入一个值到count
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    //通过count进行while循环来读取对应的权重
    while (count--)
    {
        //声明weights的类型，weights应该是trt中定义好的数据格式
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        //读取相应的某一行权重的名称
        std::string name;
        //从文件中接着输入一个字符串到name
        //然后通过std::dec来用十进制的基数格式读取size
        //具体参考https://blog.csdn.net/colddie/article/details/20667603?fps=1&locationNum=1
        input >> name >> std::dec >> size;
        //当前设定weights的数据类型，实际上所有的权值都是float
        wt.type = DataType::kFLOAT;

        // Load blob
        //加载具体的权重
        //根据size和数据类型分配内存，利用reinterpret_cast进行类型转换
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        //根据size循环读取相应的权重
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            //这里通过std::hex来使用16进制的基数格式读取相应的权重，输入到val数组
            input >> std::hex >> val[x];
        }
        //给wt.value赋值
        wt.values = val;
        //给wt.count赋值
        wt.count = size;
        //把读取的权重和对应的参数名称添加到weightmap中
        weightMap[name] = wt;
    }

    return weightMap;
}
//向network中添加一个bn层
IScaleLayer* addBatchNorm2d(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
    float *gamma = (float*)weightMap[lname + ".weight"].values;
    float *beta = (float*)weightMap[lname + ".bias"].values;
    float *mean = (float*)weightMap[lname + ".running_mean"].values;
    float *var = (float*)weightMap[lname + ".running_var"].values;
    int len = weightMap[lname + ".running_var"].count;

    float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        scval[i] = gamma[i] / sqrt(var[i] + eps);
    }
    Weights scale{DataType::kFLOAT, scval, len};
    
    float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
    }
    Weights shift{DataType::kFLOAT, shval, len};

    float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
    for (int i = 0; i < len; i++) {
        pval[i] = 1.0;
    }
    Weights power{DataType::kFLOAT, pval, len};

    weightMap[lname + ".scale"] = scale;
    weightMap[lname + ".shift"] = shift;
    weightMap[lname + ".power"] = power;
    IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, shift, scale, power);
    assert(scale_1);
    return scale_1;
}

ILayer* convBnLeaky(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, int linx) {
    //声明一个新的weights变量
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    //向tensorrt网络中添加一个卷积层
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
    assert(conv1);
    //设置相应的stride和padding值
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});
    //添加一个BN层
    IScaleLayer* bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-5);
    //然后就是激活了。利用auto关键字
    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    //设置leaky relu的超参数
    lr->setAlpha(0.1);

    return lr;
}

// Creat the engine using only the API and not any parser.
//使用相应的tensorrt的api创建网络模型，填充权重，搭建模型
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt) {
    //创建network实例
    INetworkDefinition* network = builder->createNetworkV2(0U);

    // Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
    //向network中添加一个input结点，设置名称，数据类型和维度，这里的维度实际上也就是（3，608，608）
    ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{3, INPUT_H, INPUT_W});
    assert(data);
    //加载相应的权重值，这里的std::map类似于python中的字典
    //loadweights参考本文件下的实现
    std::map<std::string, Weights> weightMap = loadWeights("../yolov3.wts");
    //声明一个新的weights变量
    Weights emptywts{DataType::kFLOAT, nullptr, 0};

    // Yeah I am stupid, I just want to expand the complete arch of darknet..
    //接着就是根据上面的输入结点来实现的yolov3的网络结构了
    //convBnLeaky参考本文件的实现，也就是卷积，BN,和激活层整合在一起了
    auto lr0 = convBnLeaky(network, weightMap, *data, 32, 3, 1, 1, 0);
    auto lr1 = convBnLeaky(network, weightMap, *lr0->getOutput(0), 64, 3, 2, 1, 1);
    auto lr2 = convBnLeaky(network, weightMap, *lr1->getOutput(0), 32, 1, 1, 0, 2);
    auto lr3 = convBnLeaky(network, weightMap, *lr2->getOutput(0), 64, 3, 1, 1, 3);
    //这里的addElementWise实现的是一个元素级别的相加操作，相当于resnet残差连接操作，把两个层次的数据相加，具体可以参考tensorrt的文档
    auto ew4 = network->addElementWise(*lr3->getOutput(0), *lr1->getOutput(0), ElementWiseOperation::kSUM);
    auto lr5 = convBnLeaky(network, weightMap, *ew4->getOutput(0), 128, 3, 2, 1, 5);
    auto lr6 = convBnLeaky(network, weightMap, *lr5->getOutput(0), 64, 1, 1, 0, 6);
    auto lr7 = convBnLeaky(network, weightMap, *lr6->getOutput(0), 128, 3, 1, 1, 7);
    auto ew8 = network->addElementWise(*lr7->getOutput(0), *lr5->getOutput(0), ElementWiseOperation::kSUM);
    auto lr9 = convBnLeaky(network, weightMap, *ew8->getOutput(0), 64, 1, 1, 0, 9);
    auto lr10 = convBnLeaky(network, weightMap, *lr9->getOutput(0), 128, 3, 1, 1, 10);
    auto ew11 = network->addElementWise(*lr10->getOutput(0), *ew8->getOutput(0), ElementWiseOperation::kSUM);
    auto lr12 = convBnLeaky(network, weightMap, *ew11->getOutput(0), 256, 3, 2, 1, 12);
    auto lr13 = convBnLeaky(network, weightMap, *lr12->getOutput(0), 128, 1, 1, 0, 13);
    auto lr14 = convBnLeaky(network, weightMap, *lr13->getOutput(0), 256, 3, 1, 1, 14);
    auto ew15 = network->addElementWise(*lr14->getOutput(0), *lr12->getOutput(0), ElementWiseOperation::kSUM);
    auto lr16 = convBnLeaky(network, weightMap, *ew15->getOutput(0), 128, 1, 1, 0, 16);
    auto lr17 = convBnLeaky(network, weightMap, *lr16->getOutput(0), 256, 3, 1, 1, 17);
    auto ew18 = network->addElementWise(*lr17->getOutput(0), *ew15->getOutput(0), ElementWiseOperation::kSUM);
    auto lr19 = convBnLeaky(network, weightMap, *ew18->getOutput(0), 128, 1, 1, 0, 19);
    auto lr20 = convBnLeaky(network, weightMap, *lr19->getOutput(0), 256, 3, 1, 1, 20);
    auto ew21 = network->addElementWise(*lr20->getOutput(0), *ew18->getOutput(0), ElementWiseOperation::kSUM);
    auto lr22 = convBnLeaky(network, weightMap, *ew21->getOutput(0), 128, 1, 1, 0, 22);
    auto lr23 = convBnLeaky(network, weightMap, *lr22->getOutput(0), 256, 3, 1, 1, 23);
    auto ew24 = network->addElementWise(*lr23->getOutput(0), *ew21->getOutput(0), ElementWiseOperation::kSUM);
    auto lr25 = convBnLeaky(network, weightMap, *ew24->getOutput(0), 128, 1, 1, 0, 25);
    auto lr26 = convBnLeaky(network, weightMap, *lr25->getOutput(0), 256, 3, 1, 1, 26);
    auto ew27 = network->addElementWise(*lr26->getOutput(0), *ew24->getOutput(0), ElementWiseOperation::kSUM);
    auto lr28 = convBnLeaky(network, weightMap, *ew27->getOutput(0), 128, 1, 1, 0, 28);
    auto lr29 = convBnLeaky(network, weightMap, *lr28->getOutput(0), 256, 3, 1, 1, 29);
    auto ew30 = network->addElementWise(*lr29->getOutput(0), *ew27->getOutput(0), ElementWiseOperation::kSUM);
    auto lr31 = convBnLeaky(network, weightMap, *ew30->getOutput(0), 128, 1, 1, 0, 31);
    auto lr32 = convBnLeaky(network, weightMap, *lr31->getOutput(0), 256, 3, 1, 1, 32);
    auto ew33 = network->addElementWise(*lr32->getOutput(0), *ew30->getOutput(0), ElementWiseOperation::kSUM);
    auto lr34 = convBnLeaky(network, weightMap, *ew33->getOutput(0), 128, 1, 1, 0, 34);
    auto lr35 = convBnLeaky(network, weightMap, *lr34->getOutput(0), 256, 3, 1, 1, 35);
    auto ew36 = network->addElementWise(*lr35->getOutput(0), *ew33->getOutput(0), ElementWiseOperation::kSUM);
    auto lr37 = convBnLeaky(network, weightMap, *ew36->getOutput(0), 512, 3, 2, 1, 37);
    auto lr38 = convBnLeaky(network, weightMap, *lr37->getOutput(0), 256, 1, 1, 0, 38);
    auto lr39 = convBnLeaky(network, weightMap, *lr38->getOutput(0), 512, 3, 1, 1, 39);
    auto ew40 = network->addElementWise(*lr39->getOutput(0), *lr37->getOutput(0), ElementWiseOperation::kSUM);
    auto lr41 = convBnLeaky(network, weightMap, *ew40->getOutput(0), 256, 1, 1, 0, 41);
    auto lr42 = convBnLeaky(network, weightMap, *lr41->getOutput(0), 512, 3, 1, 1, 42);
    auto ew43 = network->addElementWise(*lr42->getOutput(0), *ew40->getOutput(0), ElementWiseOperation::kSUM);
    auto lr44 = convBnLeaky(network, weightMap, *ew43->getOutput(0), 256, 1, 1, 0, 44);
    auto lr45 = convBnLeaky(network, weightMap, *lr44->getOutput(0), 512, 3, 1, 1, 45);
    auto ew46 = network->addElementWise(*lr45->getOutput(0), *ew43->getOutput(0), ElementWiseOperation::kSUM);
    auto lr47 = convBnLeaky(network, weightMap, *ew46->getOutput(0), 256, 1, 1, 0, 47);
    auto lr48 = convBnLeaky(network, weightMap, *lr47->getOutput(0), 512, 3, 1, 1, 48);
    auto ew49 = network->addElementWise(*lr48->getOutput(0), *ew46->getOutput(0), ElementWiseOperation::kSUM);
    auto lr50 = convBnLeaky(network, weightMap, *ew49->getOutput(0), 256, 1, 1, 0, 50);
    auto lr51 = convBnLeaky(network, weightMap, *lr50->getOutput(0), 512, 3, 1, 1, 51);
    auto ew52 = network->addElementWise(*lr51->getOutput(0), *ew49->getOutput(0), ElementWiseOperation::kSUM);
    auto lr53 = convBnLeaky(network, weightMap, *ew52->getOutput(0), 256, 1, 1, 0, 53);
    auto lr54 = convBnLeaky(network, weightMap, *lr53->getOutput(0), 512, 3, 1, 1, 54);
    auto ew55 = network->addElementWise(*lr54->getOutput(0), *ew52->getOutput(0), ElementWiseOperation::kSUM);
    auto lr56 = convBnLeaky(network, weightMap, *ew55->getOutput(0), 256, 1, 1, 0, 56);
    auto lr57 = convBnLeaky(network, weightMap, *lr56->getOutput(0), 512, 3, 1, 1, 57);
    auto ew58 = network->addElementWise(*lr57->getOutput(0), *ew55->getOutput(0), ElementWiseOperation::kSUM);
    auto lr59 = convBnLeaky(network, weightMap, *ew58->getOutput(0), 256, 1, 1, 0, 59);
    auto lr60 = convBnLeaky(network, weightMap, *lr59->getOutput(0), 512, 3, 1, 1, 60);
    auto ew61 = network->addElementWise(*lr60->getOutput(0), *ew58->getOutput(0), ElementWiseOperation::kSUM);
    auto lr62 = convBnLeaky(network, weightMap, *ew61->getOutput(0), 1024, 3, 2, 1, 62);
    auto lr63 = convBnLeaky(network, weightMap, *lr62->getOutput(0), 512, 1, 1, 0, 63);
    auto lr64 = convBnLeaky(network, weightMap, *lr63->getOutput(0), 1024, 3, 1, 1, 64);
    auto ew65 = network->addElementWise(*lr64->getOutput(0), *lr62->getOutput(0), ElementWiseOperation::kSUM);
    auto lr66 = convBnLeaky(network, weightMap, *ew65->getOutput(0), 512, 1, 1, 0, 66);
    auto lr67 = convBnLeaky(network, weightMap, *lr66->getOutput(0), 1024, 3, 1, 1, 67);
    auto ew68 = network->addElementWise(*lr67->getOutput(0), *ew65->getOutput(0), ElementWiseOperation::kSUM);
    auto lr69 = convBnLeaky(network, weightMap, *ew68->getOutput(0), 512, 1, 1, 0, 69);
    auto lr70 = convBnLeaky(network, weightMap, *lr69->getOutput(0), 1024, 3, 1, 1, 70);
    auto ew71 = network->addElementWise(*lr70->getOutput(0), *ew68->getOutput(0), ElementWiseOperation::kSUM);
    auto lr72 = convBnLeaky(network, weightMap, *ew71->getOutput(0), 512, 1, 1, 0, 72);
    auto lr73 = convBnLeaky(network, weightMap, *lr72->getOutput(0), 1024, 3, 1, 1, 73);
    auto ew74 = network->addElementWise(*lr73->getOutput(0), *ew71->getOutput(0), ElementWiseOperation::kSUM);
    auto lr75 = convBnLeaky(network, weightMap, *ew74->getOutput(0), 512, 1, 1, 0, 75);
    auto lr76 = convBnLeaky(network, weightMap, *lr75->getOutput(0), 1024, 3, 1, 1, 76);
    auto lr77 = convBnLeaky(network, weightMap, *lr76->getOutput(0), 512, 1, 1, 0, 77);
    auto lr78 = convBnLeaky(network, weightMap, *lr77->getOutput(0), 1024, 3, 1, 1, 78);
    auto lr79 = convBnLeaky(network, weightMap, *lr78->getOutput(0), 512, 1, 1, 0, 79);
    //实际上从这里就涉及到yolov3中的yolo层相关内容了
    //从这里开始就是yolov3中多尺度检测的第一个尺度的分支
    auto lr80 = convBnLeaky(network, weightMap, *lr79->getOutput(0), 1024, 3, 1, 1, 80);
    //这一层输出的维度是检测的类别数量+5（4个坐标一个置信度），乘上3的原因是启用了3中不同尺寸的anchor
    //然后这一层的输出将会传入yolo层进行分析，得到最终的输出结果
    IConvolutionLayer* conv81 = network->addConvolutionNd(*lr80->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.81.Conv2d.weight"], weightMap["module_list.81.Conv2d.bias"]);
    assert(conv81);
    // 82 is yolo
    //接下来沿着主分支继续
    auto l83 = lr79;
    auto lr84 = convBnLeaky(network, weightMap, *l83->getOutput(0), 256, 1, 1, 0, 84);

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts85{DataType::kFLOAT, deval, 256 * 2 * 2};
    //这里实际上是一个upsample层，默认权重都是1.0
    IDeconvolutionLayer* deconv85 = network->addDeconvolutionNd(*lr84->getOutput(0), 256, DimsHW{2, 2}, deconvwts85, emptywts);
    assert(deconv85);
    //设置反卷积的超参数
    deconv85->setStrideNd(DimsHW{2, 2});
    //这里的group实际上就是根据通道进行分组，实现分组卷积
    deconv85->setNbGroups(256);
    //将这个85层的权重添加到weightMap中
    weightMap["deconv85"] = deconvwts85;

    ITensor* inputTensors[] = {deconv85->getOutput(0), ew61->getOutput(0)};
    auto cat86 = network->addConcatenation(inputTensors, 2);
    auto lr87 = convBnLeaky(network, weightMap, *cat86->getOutput(0), 256, 1, 1, 0, 87);
    auto lr88 = convBnLeaky(network, weightMap, *lr87->getOutput(0), 512, 3, 1, 1, 88);
    auto lr89 = convBnLeaky(network, weightMap, *lr88->getOutput(0), 256, 1, 1, 0, 89);
    auto lr90 = convBnLeaky(network, weightMap, *lr89->getOutput(0), 512, 3, 1, 1, 90);
    auto lr91 = convBnLeaky(network, weightMap, *lr90->getOutput(0), 256, 1, 1, 0, 91);
    auto lr92 = convBnLeaky(network, weightMap, *lr91->getOutput(0), 512, 3, 1, 1, 92);
    IConvolutionLayer* conv93 = network->addConvolutionNd(*lr92->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.93.Conv2d.weight"], weightMap["module_list.93.Conv2d.bias"]);
    assert(conv93);
    // 94 is yolo
    auto l95 = lr91;
    auto lr96 = convBnLeaky(network, weightMap, *l95->getOutput(0), 128, 1, 1, 0, 96);
    Weights deconvwts97{DataType::kFLOAT, deval, 128 * 2 * 2};
    IDeconvolutionLayer* deconv97 = network->addDeconvolutionNd(*lr96->getOutput(0), 128, DimsHW{2, 2}, deconvwts97, emptywts);
    assert(deconv97);
    deconv97->setStrideNd(DimsHW{2, 2});
    deconv97->setNbGroups(128);
    ITensor* inputTensors1[] = {deconv97->getOutput(0), ew36->getOutput(0)};
    auto cat98 = network->addConcatenation(inputTensors1, 2);
    auto lr99 = convBnLeaky(network, weightMap, *cat98->getOutput(0), 128, 1, 1, 0, 99);
    auto lr100 = convBnLeaky(network, weightMap, *lr99->getOutput(0), 256, 3, 1, 1, 100);
    auto lr101 = convBnLeaky(network, weightMap, *lr100->getOutput(0), 128, 1, 1, 0, 101);
    auto lr102 = convBnLeaky(network, weightMap, *lr101->getOutput(0), 256, 3, 1, 1, 102);
    auto lr103 = convBnLeaky(network, weightMap, *lr102->getOutput(0), 128, 1, 1, 0, 103);
    auto lr104 = convBnLeaky(network, weightMap, *lr103->getOutput(0), 256, 3, 1, 1, 104);
    IConvolutionLayer* conv105 = network->addConvolutionNd(*lr104->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.105.Conv2d.weight"], weightMap["module_list.105.Conv2d.bias"]);
    assert(conv105);

    //这里用来获取相应的yolo层自定义插件的creator
    //yolo层通过自定义插件的方式集成到tensorrt中
    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    //获取需要传递给creator的字段列表
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    //通过creator创建具体的插件实例
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    //这里获取相应yolo插件的输入数据
    ITensor* inputTensors_yolo[] = {conv81->getOutput(0), conv93->getOutput(0), conv105->getOutput(0)};
    //然后将yolo层通过插件添加到network中，addPluginV2实际上就是用来添加一个插件层 
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);
    //将yolo层的输出标记成网络的输出
    yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*yolo->getOutput(0));

    // Build engine
    //设置引擎的超参数，然后根据不同的配置构建引擎
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    //Int8EntropyCalibrator2参考calibrator.cpp中的实现，实现的是int8的校准器，重写相关的虚函数
    //根据需要重写getbatch以及读取和保存校准文件的辅助函数
    //具体参考https://github.com/NVIDIA/TensorRT/tree/master/samples/opensource/sampleINT8#batch-files-for-calibration
    Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);
#endif

    std::cout << "Building engine, please wait for a while..." << std::endl;
    //根据配置构建引擎
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine successfully!" << std::endl;

    // Don't need the network any more
    //引擎创建完，释放network
    network->destroy();

    // Release host memory
    //释放主机内存，也就是存储weightMap的内存
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}
//利用tensorrt的api创建yolov3网络，构建相应的引擎
void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream) {
    // Create builder
    //创建builder和builderconfig实例
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    //创建引擎实例，也就是搭建网络，并用权重填充，具体参考本文件下的实现
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    //序列化引擎到文件
    (*modelStream) = engine->serialize();

    // Close everything down
    //销毁相应的实例
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize) {
    //从上下文获取引擎
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    //判断绑定是否正确
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    //获取输入输出的索引
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // Create GPU buffers on device
    //分配相应的内存缓冲区，这里分配的是设备内存
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    //创建cuda stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    //将数据从主机内存传输到设备上
    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    //enqueue执行具体的推理操作，对批处理数据执行异步推理
    context.enqueue(batchSize, buffers, stream, nullptr);
    //将结果从设备内存传输到主机内存
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    //线程同步操作
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    //销毁cuda stream和释放相应的内存
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv) {
    //设置设备编号
    cudaSetDevice(DEVICE);
    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    //解析命令行参数
    //如果是-s，则创建推理引擎，并将其序列化到文件
    if (argc == 2 && std::string(argv[1]) == "-s") {
        //声明内存，IHostMemory的具体实现参考https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/FoundationalTypes/HostMemory.html
        //用于处理用户可访问的库分配的内存
        IHostMemory* modelStream{nullptr};
        //具体实现参考本文件下的实现，也就是利用tensorrt的api实现yolov3，构建相应的推理引擎
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);
        //接下来就是把序列化的引擎保存到文件里面
        //std::ofstream以输出的方式打开文件，std::ios::binary设置以二进制的方式写入
        std::ofstream p("yolov3.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        //将序列化的引擎数据写入相应文件中
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        //销毁modelstream
        modelStream->destroy();
        return 0;
    } else if (argc == 3 && std::string(argv[1]) == "-d") {
        //如果参数是-d的话，以二进制读取引擎文件
        std::ifstream file("yolov3.engine", std::ios::binary);
        //file.good()表示文件正常，不是坏的，没有读写错误，也没有结束
        if (file.good()) {
            //seekg是对输入流定位，第一个参数是偏移量，第二个参数是基地址
            //这里定位到输入流的末尾
            file.seekg(0, file.end);
            //tellg返回当前定位的指针位置，也代表着输入流的大小
            //取到输入流的大小
            size = file.tellg();
            //然后将定位调整到输入流的开始
            file.seekg(0, file.beg);
            //创建一个新的trtmodelstream
            trtModelStream = new char[size];
            assert(trtModelStream);
            //将相应的输入流读取到trtmodelstream中
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./yolov3 -s  // serialize model to plan file" << std::endl;
        std::cerr << "./yolov3 -d ../samples  // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    //声明一个变量用来存储测试文件路径
    std::vector<std::string> file_names;
    //read_files_in_dir参考utils.h下的实现
    if (read_files_in_dir(argv[2], file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    // prepare input data ---------------------------
    //准备输入数据
    static float data[3 * INPUT_H * INPUT_W];
    //for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //    data[i] = 1.0;
    //prob用来保存推理的输出
    static float prob[OUTPUT_SIZE];
    //创建一个runtime实例
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    //通过读取的文件流反序列化引擎
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    //创建一个推理上下文实例
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    //销毁trtmodelstream
    delete[] trtModelStream;

    int fcount = 0;
    //循环读取测试数据
    for (auto f: file_names) {
        fcount++;
        std::cout << fcount << "  " << f << std::endl;
        //利用opencv读取图片
        cv::Mat img = cv::imread(std::string(argv[2]) + "/" + f);
        if (img.empty()) continue;
        //process_img参考utils.h下的实现，对图片进行预处理
        cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);
        //将图片的具体数据传递到data中
        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
            data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
            data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
        }

        // Run inference
        //进行推理并统计推理时间
        auto start = std::chrono::system_clock::now();
        //doinference参考本文件的实现
        doInference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        //输出推理时间
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<Yolo::Detection> res;
        //对输出进行nms，具体参考本文件的实现
        nms(res, prob);
        //遍历nms之后剩下的框
        for (size_t j = 0; j < res.size(); j++) {
            //get_rect实际上就是坐标框的转换
            cv::Rect r = get_rect(img, res[j].bbox);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
        cv::imwrite("_" + f, img);
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
//还剩下本文件下的一些数据处理，和utils.h以及yolo.cu里面的内容没看了。Int8校准的过程也可以再补补。
//后面看看数据处理的东西，写一下整个推理的文档，以及int8校准需要重写的内容，yolo插件基本上还是一样的套路，注意一下具体的实现流程就行
//以及相应的重写函数等等，明天写文档吧