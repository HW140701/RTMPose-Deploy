#ifndef _RTM_POSE_ONNXRUNTIME_H_
#define _RTM_POSE_ONNXRUNTIME_H_

#include <string>

#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"
#include "opencv2/opencv.hpp"

#include "rtmdet_onnxruntime.h"
#include "rtmpose_utils.h"
#include "onnxruntime_cpu_model_base.h"

class RTMPoseOnnxruntime : public OnnxruntimeCPUModelBase
{
public:
	RTMPoseOnnxruntime();
	virtual~RTMPoseOnnxruntime();

public:
	std::vector<PosePoint> Inference(const cv::Mat& input_mat, const DetectBox& box);

private:
	std::pair<cv::Mat, cv::Mat> CropImageByDetectBox(const cv::Mat& input_image, const DetectBox& box);

};

#endif // !_RTM_POSE_ONNXRUNTIME_H_
