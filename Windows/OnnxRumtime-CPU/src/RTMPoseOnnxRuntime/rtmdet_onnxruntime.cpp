#include "rtmdet_onnxruntime.h"

#include <iostream>
#include <thread>

#include "characterset_convert.h"



RTMDetOnnxruntime::RTMDetOnnxruntime()
{
}

RTMDetOnnxruntime::~RTMDetOnnxruntime()
{
}

DetectBox RTMDetOnnxruntime::Inference(const cv::Mat& input_mat)
{
	// Deep copy
	// 深拷贝
	cv::Mat input_mat_copy;
	input_mat.copyTo(input_mat_copy);

	// BGR to RGB
	// BGR转RGB
	cv::Mat input_mat_copy_rgb;
	cv::cvtColor(input_mat_copy, input_mat_copy_rgb, CV_BGR2RGB);

	// image data，HWC->CHW，image_data - mean / std normalize
	// 图片数据，HWC->CHW，image_data - mean / std归一化
	int image_height = input_mat_copy_rgb.rows;
	int image_width = input_mat_copy_rgb.cols;
	int image_channels = input_mat_copy_rgb.channels();

	std::vector<float> input_image_array;
	input_image_array.resize(1 * image_channels * image_height * image_width);

	float* input_image = input_image_array.data();
	for (int h = 0; h < image_height; ++h)
	{
		for (int w = 0; w < image_width; ++w)
		{
			for (int c = 0; c < image_channels; ++c)
			{
				int chw_index = c * image_height * image_width + h * image_width + w;

				float tmp = input_mat_copy_rgb.ptr<uchar>(h)[w * 3 + c];

				input_image[chw_index] = (tmp - IMAGE_MEAN[c]) / IMAGE_STD[c];
			}
		}
	}

	// inference
	// 推理
	std::vector<const char*> m_onnx_input_names{ "input" };
	std::vector<const char*> m_onnx_output_names{ "dets","labels"};
	std::array<int64_t, 4> input_shape{ 1, image_channels, image_height, image_width };

	auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
		memory_info,
		input_image_array.data(),
		input_image_array.size(),
		input_shape.data(),
		input_shape.size()
	);

	assert(input_tensor.IsTensor());

	auto output_tensors = m_session.Run(
		Ort::RunOptions{ nullptr },
		m_onnx_input_names.data(),
		&input_tensor,
		1,
		m_onnx_output_names.data(),
		m_onnx_output_names.size()
	);

	// pose process
	// 后处理
	std::vector<int64_t> det_result_dims = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
	std::vector<int64_t> label_result_dims = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

	assert(det_result_dims.size() == 3 && label_result_dims.size() == 2);

	int batch_size = det_result_dims[0] == label_result_dims[0] ? det_result_dims[0] : 0;
	int num_dets = det_result_dims[1] == label_result_dims[1] ? det_result_dims[1] : 0;
	int reshap_dims = det_result_dims[2];

	float* det_result = output_tensors[0].GetTensorMutableData<float>();
	int* label_result = output_tensors[1].GetTensorMutableData<int>();

	std::vector<DetectBox> all_box;
	for (int i = 0; i < num_dets; ++i)
	{
		int classes = label_result[i];
		if (classes != 0)
			continue;

		DetectBox temp_box;
		temp_box.left = int(det_result[i * reshap_dims]);
		temp_box.top = int(det_result[i * reshap_dims + 1]);
		temp_box.right = int(det_result[i * reshap_dims + 2]);
		temp_box.bottom = int(det_result[i * reshap_dims + 3]);
		temp_box.score = det_result[i * reshap_dims + 4];
		temp_box.label = label_result[i];

		all_box.emplace_back(temp_box);
	}

	// descending sort
	// 降序排序
	std::sort(all_box.begin(), all_box.end(), BoxCompare);

	//cv::rectangle(input_mat_copy, cv::Point{ all_box[0].left, all_box[0].top }, cv::Point{ all_box[0].right, all_box[0].bottom }, cv::Scalar{ 0, 255, 0 });

	//cv::imwrite("detect.jpg", input_mat_copy);

	DetectBox result_box;

	if (!all_box.empty())
	{
		result_box = all_box[0];
	}

	return result_box;
}

