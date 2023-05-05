#include "rtmpose_tracker_onnxruntime.h"

RTMPoseTrackerOnnxruntime::RTMPoseTrackerOnnxruntime()
	:m_ptr_rtm_det(nullptr),
	m_ptr_rtm_pose(nullptr),
	m_frame_num(0),
	m_dectect_interval(10)
{

}

RTMPoseTrackerOnnxruntime::~RTMPoseTrackerOnnxruntime()
{
}

bool RTMPoseTrackerOnnxruntime::LoadModel(const std::string& det_model_path, const std::string& pose_model_path, int dectect_interval)
{
	m_ptr_rtm_det = std::make_unique<RTMDetOnnxruntime>();
	if (m_ptr_rtm_det == nullptr)
		return false;

	if (!m_ptr_rtm_det->LoadModel("rtm_det_nano", det_model_path))
		return false;

	m_ptr_rtm_pose = std::make_unique<RTMPoseOnnxruntime>();
	if (m_ptr_rtm_pose == nullptr)
		return false;

	if (!m_ptr_rtm_pose->LoadModel("rtm_pose", pose_model_path))
		return false;

	m_dectect_interval = dectect_interval;

	return true;
}

std::pair<DetectBox, std::vector<PosePoint>> RTMPoseTrackerOnnxruntime::Inference(const cv::Mat& input_mat)
{
	std::pair<DetectBox, std::vector<PosePoint>> result;

	if (m_ptr_rtm_det == nullptr || m_ptr_rtm_pose == nullptr)
		return result;

	if (m_frame_num % m_dectect_interval == 0)
	{
		m_detect_box = m_ptr_rtm_det->Inference(input_mat);
	}

	std::vector<PosePoint> pose_result = m_ptr_rtm_pose->Inference(input_mat, m_detect_box);

	m_frame_num += 1;

	return std::make_pair(m_detect_box, pose_result);
}
