#ifndef _RTM_DET_ONNX_RUNTIME_H_
#define _RTM_DET_ONNX_RUNTIME_H_

#include <string>

#include "rtmpose_utils.h"
#include "onnxruntime_cpu_model_base.h"

class RTMDetOnnxruntime : public OnnxruntimeCPUModelBase
{
public:
	RTMDetOnnxruntime();
	virtual~RTMDetOnnxruntime();

public:
	DetectBox Inference(const cv::Mat& input_mat);

};

#endif // !_RTM_DET_ONNX_RUNTIME_H_
