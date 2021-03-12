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
#include "yoloplugin.h"
#include "calibrator.h"

using namespace nvinfer1;

IScaleLayer* BN_layer(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, std::string lname, float eps) {
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

ILayer* BaseBlock(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input,  int outch, int ksize, int s, int p, int linx) {
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{ksize, ksize}, weightMap["module_list." + std::to_string(linx) + ".Conv2d.weight"], emptywts);
    assert(conv1);
    conv1->setStrideNd(DimsHW{s, s});
    conv1->setPaddingNd(DimsHW{p, p});

    IScaleLayer* bn1 = BN_layer(network, weightMap, *conv1->getOutput(0), "module_list." + std::to_string(linx) + ".BatchNorm2d", 1e-5);

    auto lr = network->addActivation(*bn1->getOutput(0), ActivationType::kLEAKY_RELU);
    lr->setAlpha(0.1);

    return lr;
}

void get_model(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, Weights& emptywts,char* INPUT_NODE,char* OUTPUT_NODE,DataType dt,int INPUT_H,int INPUT_W){
    ITensor* data = network->addInput(INPUT_NODE, dt, Dims3{3, INPUT_H, INPUT_W});
    auto lr0 = BaseBlock(network, weightMap, *data, 32, 3, 1, 1, 0);
    auto lr1 = BaseBlock(network, weightMap, *lr0->getOutput(0), 64, 3, 2, 1, 1);
    auto lr2 = BaseBlock(network, weightMap, *lr1->getOutput(0), 32, 1, 1, 0, 2);
    auto lr3 = BaseBlock(network, weightMap, *lr2->getOutput(0), 64, 3, 1, 1, 3);
    auto ew4 = network->addElementWise(*lr3->getOutput(0), *lr1->getOutput(0), ElementWiseOperation::kSUM);
    auto lr5 = BaseBlock(network, weightMap, *ew4->getOutput(0), 128, 3, 2, 1, 5);
    auto lr6 = BaseBlock(network, weightMap, *lr5->getOutput(0), 64, 1, 1, 0, 6);
    auto lr7 = BaseBlock(network, weightMap, *lr6->getOutput(0), 128, 3, 1, 1, 7);
    auto ew8 = network->addElementWise(*lr7->getOutput(0), *lr5->getOutput(0), ElementWiseOperation::kSUM);
    auto lr9 = BaseBlock(network, weightMap, *ew8->getOutput(0), 64, 1, 1, 0, 9);
    auto lr10 = BaseBlock(network, weightMap, *lr9->getOutput(0), 128, 3, 1, 1, 10);
    auto ew11 = network->addElementWise(*lr10->getOutput(0), *ew8->getOutput(0), ElementWiseOperation::kSUM);
    auto lr12 = BaseBlock(network, weightMap, *ew11->getOutput(0), 256, 3, 2, 1, 12);
    auto lr13 = BaseBlock(network, weightMap, *lr12->getOutput(0), 128, 1, 1, 0, 13);
    auto lr14 = BaseBlock(network, weightMap, *lr13->getOutput(0), 256, 3, 1, 1, 14);
    auto ew15 = network->addElementWise(*lr14->getOutput(0), *lr12->getOutput(0), ElementWiseOperation::kSUM);
    auto lr16 = BaseBlock(network, weightMap, *ew15->getOutput(0), 128, 1, 1, 0, 16);
    auto lr17 = BaseBlock(network, weightMap, *lr16->getOutput(0), 256, 3, 1, 1, 17);
    auto ew18 = network->addElementWise(*lr17->getOutput(0), *ew15->getOutput(0), ElementWiseOperation::kSUM);
    auto lr19 = BaseBlock(network, weightMap, *ew18->getOutput(0), 128, 1, 1, 0, 19);
    auto lr20 = BaseBlock(network, weightMap, *lr19->getOutput(0), 256, 3, 1, 1, 20);
    auto ew21 = network->addElementWise(*lr20->getOutput(0), *ew18->getOutput(0), ElementWiseOperation::kSUM);
    auto lr22 = BaseBlock(network, weightMap, *ew21->getOutput(0), 128, 1, 1, 0, 22);
    auto lr23 = BaseBlock(network, weightMap, *lr22->getOutput(0), 256, 3, 1, 1, 23);
    auto ew24 = network->addElementWise(*lr23->getOutput(0), *ew21->getOutput(0), ElementWiseOperation::kSUM);
    auto lr25 = BaseBlock(network, weightMap, *ew24->getOutput(0), 128, 1, 1, 0, 25);
    auto lr26 = BaseBlock(network, weightMap, *lr25->getOutput(0), 256, 3, 1, 1, 26);
    auto ew27 = network->addElementWise(*lr26->getOutput(0), *ew24->getOutput(0), ElementWiseOperation::kSUM);
    auto lr28 = BaseBlock(network, weightMap, *ew27->getOutput(0), 128, 1, 1, 0, 28);
    auto lr29 = BaseBlock(network, weightMap, *lr28->getOutput(0), 256, 3, 1, 1, 29);
    auto ew30 = network->addElementWise(*lr29->getOutput(0), *ew27->getOutput(0), ElementWiseOperation::kSUM);
    auto lr31 = BaseBlock(network, weightMap, *ew30->getOutput(0), 128, 1, 1, 0, 31);
    auto lr32 = BaseBlock(network, weightMap, *lr31->getOutput(0), 256, 3, 1, 1, 32);
    auto ew33 = network->addElementWise(*lr32->getOutput(0), *ew30->getOutput(0), ElementWiseOperation::kSUM);
    auto lr34 = BaseBlock(network, weightMap, *ew33->getOutput(0), 128, 1, 1, 0, 34);
    auto lr35 = BaseBlock(network, weightMap, *lr34->getOutput(0), 256, 3, 1, 1, 35);
    auto ew36 = network->addElementWise(*lr35->getOutput(0), *ew33->getOutput(0), ElementWiseOperation::kSUM);
    auto lr37 = BaseBlock(network, weightMap, *ew36->getOutput(0), 512, 3, 2, 1, 37);
    auto lr38 = BaseBlock(network, weightMap, *lr37->getOutput(0), 256, 1, 1, 0, 38);
    auto lr39 = BaseBlock(network, weightMap, *lr38->getOutput(0), 512, 3, 1, 1, 39);
    auto ew40 = network->addElementWise(*lr39->getOutput(0), *lr37->getOutput(0), ElementWiseOperation::kSUM);
    auto lr41 = BaseBlock(network, weightMap, *ew40->getOutput(0), 256, 1, 1, 0, 41);
    auto lr42 = BaseBlock(network, weightMap, *lr41->getOutput(0), 512, 3, 1, 1, 42);
    auto ew43 = network->addElementWise(*lr42->getOutput(0), *ew40->getOutput(0), ElementWiseOperation::kSUM);
    auto lr44 = BaseBlock(network, weightMap, *ew43->getOutput(0), 256, 1, 1, 0, 44);
    auto lr45 = BaseBlock(network, weightMap, *lr44->getOutput(0), 512, 3, 1, 1, 45);
    auto ew46 = network->addElementWise(*lr45->getOutput(0), *ew43->getOutput(0), ElementWiseOperation::kSUM);
    auto lr47 = BaseBlock(network, weightMap, *ew46->getOutput(0), 256, 1, 1, 0, 47);
    auto lr48 = BaseBlock(network, weightMap, *lr47->getOutput(0), 512, 3, 1, 1, 48);
    auto ew49 = network->addElementWise(*lr48->getOutput(0), *ew46->getOutput(0), ElementWiseOperation::kSUM);
    auto lr50 = BaseBlock(network, weightMap, *ew49->getOutput(0), 256, 1, 1, 0, 50);
    auto lr51 = BaseBlock(network, weightMap, *lr50->getOutput(0), 512, 3, 1, 1, 51);
    auto ew52 = network->addElementWise(*lr51->getOutput(0), *ew49->getOutput(0), ElementWiseOperation::kSUM);
    auto lr53 = BaseBlock(network, weightMap, *ew52->getOutput(0), 256, 1, 1, 0, 53);
    auto lr54 = BaseBlock(network, weightMap, *lr53->getOutput(0), 512, 3, 1, 1, 54);
    auto ew55 = network->addElementWise(*lr54->getOutput(0), *ew52->getOutput(0), ElementWiseOperation::kSUM);
    auto lr56 = BaseBlock(network, weightMap, *ew55->getOutput(0), 256, 1, 1, 0, 56);
    auto lr57 = BaseBlock(network, weightMap, *lr56->getOutput(0), 512, 3, 1, 1, 57);
    auto ew58 = network->addElementWise(*lr57->getOutput(0), *ew55->getOutput(0), ElementWiseOperation::kSUM);
    auto lr59 = BaseBlock(network, weightMap, *ew58->getOutput(0), 256, 1, 1, 0, 59);
    auto lr60 = BaseBlock(network, weightMap, *lr59->getOutput(0), 512, 3, 1, 1, 60);
    auto ew61 = network->addElementWise(*lr60->getOutput(0), *ew58->getOutput(0), ElementWiseOperation::kSUM);
    auto lr62 = BaseBlock(network, weightMap, *ew61->getOutput(0), 1024, 3, 2, 1, 62);
    auto lr63 = BaseBlock(network, weightMap, *lr62->getOutput(0), 512, 1, 1, 0, 63);
    auto lr64 = BaseBlock(network, weightMap, *lr63->getOutput(0), 1024, 3, 1, 1, 64);
    auto ew65 = network->addElementWise(*lr64->getOutput(0), *lr62->getOutput(0), ElementWiseOperation::kSUM);
    auto lr66 = BaseBlock(network, weightMap, *ew65->getOutput(0), 512, 1, 1, 0, 66);
    auto lr67 = BaseBlock(network, weightMap, *lr66->getOutput(0), 1024, 3, 1, 1, 67);
    auto ew68 = network->addElementWise(*lr67->getOutput(0), *ew65->getOutput(0), ElementWiseOperation::kSUM);
    auto lr69 = BaseBlock(network, weightMap, *ew68->getOutput(0), 512, 1, 1, 0, 69);
    auto lr70 = BaseBlock(network, weightMap, *lr69->getOutput(0), 1024, 3, 1, 1, 70);
    auto ew71 = network->addElementWise(*lr70->getOutput(0), *ew68->getOutput(0), ElementWiseOperation::kSUM);
    auto lr72 = BaseBlock(network, weightMap, *ew71->getOutput(0), 512, 1, 1, 0, 72);
    auto lr73 = BaseBlock(network, weightMap, *lr72->getOutput(0), 1024, 3, 1, 1, 73);
    auto ew74 = network->addElementWise(*lr73->getOutput(0), *ew71->getOutput(0), ElementWiseOperation::kSUM);
    auto lr75 = BaseBlock(network, weightMap, *ew74->getOutput(0), 512, 1, 1, 0, 75);
    auto lr76 = BaseBlock(network, weightMap, *lr75->getOutput(0), 1024, 3, 1, 1, 76);
    auto lr77 = BaseBlock(network, weightMap, *lr76->getOutput(0), 512, 1, 1, 0, 77);
    auto lr78 = BaseBlock(network, weightMap, *lr77->getOutput(0), 1024, 3, 1, 1, 78);
    auto lr79 = BaseBlock(network, weightMap, *lr78->getOutput(0), 512, 1, 1, 0, 79);
    auto lr80 = BaseBlock(network, weightMap, *lr79->getOutput(0), 1024, 3, 1, 1, 80);
    IConvolutionLayer* conv81 = network->addConvolutionNd(*lr80->getOutput(0), 3 * (Yolo_plugin::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.81.Conv2d.weight"], weightMap["module_list.81.Conv2d.bias"]);
    assert(conv81);
    // 82 is yolo
    auto l83 = lr79;
    auto lr84 = BaseBlock(network, weightMap, *l83->getOutput(0), 256, 1, 1, 0, 84);

    float *deval = reinterpret_cast<float*>(malloc(sizeof(float) * 256 * 2 * 2));
    for (int i = 0; i < 256 * 2 * 2; i++) {
        deval[i] = 1.0;
    }
    Weights deconvwts85{DataType::kFLOAT, deval, 256 * 2 * 2};
    IDeconvolutionLayer* deconv85 = network->addDeconvolutionNd(*lr84->getOutput(0), 256, DimsHW{2, 2}, deconvwts85, emptywts);
    assert(deconv85);
    deconv85->setStrideNd(DimsHW{2, 2});
    deconv85->setNbGroups(256);
    weightMap["deconv85"] = deconvwts85;

    ITensor* inputTensors[] = {deconv85->getOutput(0), ew61->getOutput(0)};
    auto cat86 = network->addConcatenation(inputTensors, 2);
    auto lr87 = BaseBlock(network, weightMap, *cat86->getOutput(0), 256, 1, 1, 0, 87);
    auto lr88 = BaseBlock(network, weightMap, *lr87->getOutput(0), 512, 3, 1, 1, 88);
    auto lr89 = BaseBlock(network, weightMap, *lr88->getOutput(0), 256, 1, 1, 0, 89);
    auto lr90 = BaseBlock(network, weightMap, *lr89->getOutput(0), 512, 3, 1, 1, 90);
    auto lr91 = BaseBlock(network, weightMap, *lr90->getOutput(0), 256, 1, 1, 0, 91);
    auto lr92 = BaseBlock(network, weightMap, *lr91->getOutput(0), 512, 3, 1, 1, 92);
    IConvolutionLayer* conv93 = network->addConvolutionNd(*lr92->getOutput(0), 3 * (Yolo_plugin::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.93.Conv2d.weight"], weightMap["module_list.93.Conv2d.bias"]);
    assert(conv93);
    // 94 is yolo
    auto l95 = lr91;
    auto lr96 = BaseBlock(network, weightMap, *l95->getOutput(0), 128, 1, 1, 0, 96);
    Weights deconvwts97{DataType::kFLOAT, deval, 128 * 2 * 2};
    IDeconvolutionLayer* deconv97 = network->addDeconvolutionNd(*lr96->getOutput(0), 128, DimsHW{2, 2}, deconvwts97, emptywts);
    assert(deconv97);
    deconv97->setStrideNd(DimsHW{2, 2});
    deconv97->setNbGroups(128);
    ITensor* inputTensors1[] = {deconv97->getOutput(0), ew36->getOutput(0)};
    auto cat98 = network->addConcatenation(inputTensors1, 2);
    auto lr99 = BaseBlock(network, weightMap, *cat98->getOutput(0), 128, 1, 1, 0, 99);
    auto lr100 = BaseBlock(network, weightMap, *lr99->getOutput(0), 256, 3, 1, 1, 100);
    auto lr101 = BaseBlock(network, weightMap, *lr100->getOutput(0), 128, 1, 1, 0, 101);
    auto lr102 = BaseBlock(network, weightMap, *lr101->getOutput(0), 256, 3, 1, 1, 102);
    auto lr103 = BaseBlock(network, weightMap, *lr102->getOutput(0), 128, 1, 1, 0, 103);
    auto lr104 = BaseBlock(network, weightMap, *lr103->getOutput(0), 256, 3, 1, 1, 104);
    IConvolutionLayer* conv105 = network->addConvolutionNd(*lr104->getOutput(0), 3 * (Yolo_plugin::CLASS_NUM + 5), DimsHW{1, 1}, weightMap["module_list.105.Conv2d.weight"], weightMap["module_list.105.Conv2d.bias"]);
    assert(conv105);

    auto creator = getPluginRegistry()->getPluginCreator("YoloLayer_TRT", "1");
    const PluginFieldCollection* pluginData = creator->getFieldNames();
    IPluginV2 *pluginObj = creator->createPlugin("yololayer", pluginData);
    ITensor* inputTensors_yolo[] = {conv81->getOutput(0), conv93->getOutput(0), conv105->getOutput(0)};
    auto yolo = network->addPluginV2(inputTensors_yolo, 3, *pluginObj);

    yolo->getOutput(0)->setName(OUTPUT_NODE);
    network->markOutput(*yolo->getOutput(0));

}
