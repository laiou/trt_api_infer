#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include "../include/NvInfer.h"
#include "../include/cuda_runtime_api.h"
#include "../include/utils.h"
#include "../include/logging.h"
#include "../include/yoloplugin.h"
#include "../include/calibrator.h"
#include "../include/model.h"

#define CUDA_DEVICE 0  
#define NMS_THRESH 0.4
#define BBOX_CONF_THRESH 0.5

using namespace nvinfer1;

static const int INPUT_H = Yolo_plugin::INPUT_H;
static const int INPUT_W = Yolo_plugin::INPUT_W;
static const int DETECTION_SIZE = sizeof(Yolo_plugin::Detection) / sizeof(float);
static const int OUTPUT_SIZE = Yolo_plugin::MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;    
char* INPUT_NODE = "input";
char* OUTPUT_NODE = "output";
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

bool cmp(const Yolo_plugin::Detection& a, const Yolo_plugin::Detection& b) {
    return a.det_confidence > b.det_confidence;
}

void nms(std::vector<Yolo_plugin::Detection>& res, float *output, float nms_thresh = NMS_THRESH) {
    std::map<float, std::vector<Yolo_plugin::Detection>> m;
    for (int i = 0; i < output[0] && i < 1000; i++) {
        if (output[1 + 7 * i + 4] <= BBOX_CONF_THRESH) continue;
        Yolo_plugin::Detection det;
        memcpy(&det, &output[1 + 7 * i], 7 * sizeof(float));
        if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Yolo_plugin::Detection>());
        m[det.class_id].push_back(det);
    }
    for (auto it = m.begin(); it != m.end(); it++) {
        //std::cout << it->second[0].class_id << " --- " << std::endl;
        auto& dets = it->second;
        std::sort(dets.begin(), dets.end(), cmp);
        for (size_t m = 0; m < dets.size(); ++m) {
            auto& item = dets[m];
            res.push_back(item);
            for (size_t n = m + 1; n < dets.size(); ++n) {
                if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }
}

// [tesor name] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file) {
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

// Creat infer engine
ICudaEngine* get_Engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config,int type, DataType dt) {
    INetworkDefinition* network = builder->createNetworkV2(0U);
    std::map<std::string, Weights> weightMap = loadWeights("../model/yolov3.txt");
    Weights emptywts{DataType::kFLOAT, nullptr, 0};
    get_model(network,weightMap,emptywts,INPUT_NODE,OUTPUT_NODE,dt,INPUT_H,INPUT_W);
    
    // set engine config
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 * (1 << 20)); 

    if(type == 0){
        std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
        assert(builder->platformHasFastInt8());
        config->setFlag(BuilderFlag::kINT8);
	std::cout<<"using int8 create engine."<< std::endl;
        Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "../calib_data/", "calibration.table.trt", INPUT_NODE);
        config->setInt8Calibrator(calibrator);
    }else if(type == 1){
      config->setFlag(BuilderFlag::kFP16); 
      std::cout<<"using fp16 create engine."<< std::endl;
    }else if(type == 2){
      std::cout<<"using fp32 create engine."<< std::endl;	
    }

    std::cout << "Building engine..." << std::endl;
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    std::cout << "Build engine end." << std::endl;

    // destory network
    network->destroy();

    // free host memory
    for (auto& mem : weightMap)
    {
        free((void*) (mem.second.values));
    }

    return engine;
}

void Model_engine(unsigned int maxBatchSize, IHostMemory** modelStream,int type) {
    // Create builder and build config
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // get tensorrt engine
    ICudaEngine* engine = get_Engine(maxBatchSize, builder, config, type,DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize emgine
    (*modelStream) = engine->serialize();
    engine->destroy();
    builder->destroy();
}

void Inference(IExecutionContext& context, float* input, float* output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    //check engine input and output
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    //useing getBindingIndex to   get intput and output index 
    const int inputIndex = engine.getBindingIndex(INPUT_NODE);
    const int outputIndex = engine.getBindingIndex(OUTPUT_NODE);

    //device memory 
    CUDA_CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create cuda  stream
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    //copy input data to device
    CUDA_CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    //copy result to host
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CUDA_CHECK(cudaFree(buffers[inputIndex]));
    CUDA_CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv) {
    cudaSetDevice(CUDA_DEVICE);
    // create a model and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};
    int type;
    if (argc == 3 && std::string(argv[1]) == "-getengine") {
        if(std::string(argv[2]) == "-int8"){
		type = 0;
	}else if(std::string(argv[2]) == "-fp16"){
		type = 1;
	}else if(std::string(argv[2]) == "-fp32"){
		type = 2;
	}else{
		std::cerr << "infer type set error,use'-int8','-fp16','-fp32' to set type." << std::endl;
	}
   
        assert(type == 0 || type ==1 || type == 2);
        IHostMemory* modelStream{nullptr};
        Model_engine(1, &modelStream,type);
        assert(modelStream != nullptr);
        std::ofstream p("infer.engine", std::ios::binary);
        if (!p) {
            std::cerr << "could not open engine output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 0;
    } else if (argc == 3 && std::string(argv[1]) == "-inference") {
        std::ifstream file("infer.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        std::cerr << "check input parameter." << std::endl;
        std::cerr << "use './project -getengine -int8 or -fp16 or -fp32' to build tensorrt engine file." << std::endl;
        std::cerr << "usd './project -inference ../test_data' to test model." << std::endl;
        return -1;
    }

    std::vector<std::string> file_names;
    if (read_files_in_dir(argv[2], file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }

    
    static float data[3 * INPUT_H * INPUT_W];
    static float prob[OUTPUT_SIZE];
    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;

    int fcount = 0;
    for (auto f: file_names) {
        fcount++;
        std::cout << fcount << "  " << f << std::endl;
        cv::Mat img = cv::imread(std::string(argv[2]) + "/" + f);
        if (img.empty()) continue;
        cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);
        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
            data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
            data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
        }

        //inference
        auto start = std::chrono::system_clock::now();
        Inference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
        std::vector<Yolo_plugin::Detection> res;
        nms(res, prob);
        for (size_t j = 0; j < res.size(); j++) {
            cv::Rect r = get_rect(img, res[j].bbox);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
        cv::imwrite("result_" + f, img);
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 0;
}
