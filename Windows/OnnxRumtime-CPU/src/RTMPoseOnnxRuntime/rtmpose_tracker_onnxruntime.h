#ifndef _RTM_POSE_TRACKER_ONNXRUNTIME_H_
#define _RTM_POSE_TRACKER_ONNXRUNTIME_H_

#include "rtmdet_onnxruntime.h"
#include "rtmpose_onnxruntime.h"

#include <vector>
#include <memory>

class RTMPoseTrackerOnnxruntime
{
public:
	RTMPoseTrackerOnnxruntime();
	virtual~RTMPoseTrackerOnnxruntime();

public:
	bool LoadModel(const std::string& det_model_path, const std::string& pose_model_path, int dectect_interval = 10);
	std::pair<DetectBox, std::vector<PosePoint>> Inference(const cv::Mat& input_mat);

private:
	std::unique_ptr<RTMDetOnnxruntime> m_ptr_rtm_det;
	std::unique_ptr<RTMPoseOnnxruntime> m_ptr_rtm_pose;
	unsigned int m_frame_num;
	DetectBox m_detect_box;
	int m_dectect_interval;
};

#endif // !_RTM_POSE_TRACKER_ONNXRUNTIME_H_
