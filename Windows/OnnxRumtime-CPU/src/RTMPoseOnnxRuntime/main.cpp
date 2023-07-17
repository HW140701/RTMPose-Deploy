#include <iostream>

#include "opencv2/opencv.hpp"

#include "rtmpose_utils.h"
#include "rtmpose_onnxruntime.h"
#include "rtmdet_onnxruntime.h"
#include "rtmpose_tracker_onnxruntime.h"

std::vector<std::pair<int, int>> coco_17_joint_links = {
	{0,1},{0,2},{1,3},{2,4},{5,7},{7,9},{6,8},{8,10},{5,6},{5,11},{6,12},{11,12},{11,13},{13,15},{12,14},{14,16}
};

int main()
{
	std::string rtm_detnano_onnx_path = "";
	std::string rtm_pose_onnx_path = "";
#ifdef _DEBUG
	rtm_detnano_onnx_path = "../../resource/model/rtmpose-cpu/rtmpose-ort/rtmdet-nano/end2end.onnx";
	rtm_pose_onnx_path = "../../resource/model/rtmpose-cpu/rtmpose-ort/rtmpose-m/end2end.onnx";
#else
	rtm_detnano_onnx_path = "./resource/model/rtmpose-cpu/rtmpose-ort/rtmdet-nano/end2end.onnx";
	rtm_pose_onnx_path = "./resource/model/rtmpose-cpu/rtmpose-ort/rtmpose-m/end2end.onnx";
#endif

	RTMPoseTrackerOnnxruntime rtmpose_tracker_onnxruntime;
	bool load_model_result = rtmpose_tracker_onnxruntime.LoadModel(rtm_detnano_onnx_path, rtm_pose_onnx_path, 5);

	if (!load_model_result)
	{
		std::cout << "onnx model loaded failed!" << std::endl;
		return 0;
	}

	// 如果要检测视频
	//std::string video_path = "./test.mp4";
	//cv::VideoCapture video_reader(video_path);

	cv::VideoCapture video_reader(0);
	int frame_num = 0;
	DetectBox detect_box;
	while (video_reader.isOpened())
	{
		cv::Mat frame;
		video_reader >> frame;

		if (frame.empty())
			break;

		int frame_width = frame.cols;
		int frame_height = frame.rows;

		cv::Mat frame_resize;
		float scale = LetterBoxImage(frame, frame_resize, cv::Size(320, 320), 32, cv::Scalar(128,128,128), true);

		std::pair<DetectBox, std::vector<PosePoint>> inference_result = rtmpose_tracker_onnxruntime.Inference(frame_resize);
		DetectBox detect_box = inference_result.first;
		detect_box.left = detect_box.left * scale;
		detect_box.right = detect_box.right * scale;
		detect_box.top = detect_box.top * scale;
		detect_box.bottom = detect_box.bottom * scale;
 
		std::vector<PosePoint> pose_result = inference_result.second;

		for (int i = 0; i < pose_result.size(); ++i)
		{
			pose_result[i].x = pose_result[i].x * scale;
			pose_result[i].y = pose_result[i].y * scale;
		}

		if (detect_box.IsValid())
		{
			cv::rectangle(
				frame,
				cv::Point(detect_box.left, detect_box.top),
				cv::Point(detect_box.right, detect_box.bottom),
				cv::Scalar{ 255, 0, 0 },
				2);

			for (int i = 0; i < pose_result.size(); ++i)
			{
				cv::circle(frame, cv::Point(pose_result[i].x, pose_result[i].y), 1, cv::Scalar{ 0, 0, 255 }, 5, cv::LINE_AA);
			}

			for (int i = 0; i < coco_17_joint_links.size(); ++i)
			{
				std::pair<int, int> joint_links = coco_17_joint_links[i];
				cv::line(
					frame,
					cv::Point(pose_result[joint_links.first].x, pose_result[joint_links.first].y),
					cv::Point(pose_result[joint_links.second].x, pose_result[joint_links.second].y),
					cv::Scalar{ 0, 255, 0 },
					2,
					cv::LINE_AA);
			}
		}


		imshow("RTMPose", frame);
		if (cv::waitKey(1) >= 0)
			break;
	}

	video_reader.release();
	cv::destroyAllWindows();

	return 0;
}