#ifndef _RTM_POSE_UTILS_H_
#define _RTM_POSE_UTILS_H_

#include "opencv2/opencv.hpp"

const std::vector<float> IMAGE_MEAN{ 123.675, 116.28, 103.53 };
const std::vector<float> IMAGE_STD{ 58.395, 57.12, 57.375 };

struct DetectBox
{
	int left;
	int top;
	int right;
	int bottom;
	float score;
	int label;

	DetectBox()
	{
		left = -1;
		top = -1;
		right = -1;
		bottom = -1;
		score = -1.0;
		label = -1;
	}

	bool IsValid() const
	{
		return left != -1 && top != -1 && right != -1 && bottom != -1 && score != -1.0 && label != -1;
	}
};

static bool BoxCompare(
	const DetectBox& a,
	const DetectBox& b) {
	return a.score > b.score;
}

struct PosePoint
{
	int x;
	int y;
	float score;

	PosePoint()
	{
		x = 0;
		y = 0;
		score = 0.0;
	}
};

typedef PosePoint Vector2D;


static cv::Mat GetAffineTransform(float center_x, float center_y, float scale_width, float scale_height, int output_image_width, int output_image_height, bool inverse = false)
{
	// solve the affine transformation matrix
	/* 求解仿射变换矩阵 */

	// get the three points corresponding to the source picture and the target picture
	// 获取源图片与目标图片的对应的三个点
	cv::Point2f src_point_1;
	src_point_1.x = center_x;
	src_point_1.y = center_y;

	cv::Point2f src_point_2;
	src_point_2.x = center_x;
	src_point_2.y = center_y - scale_width * 0.5;

	cv::Point2f src_point_3;
	src_point_3.x = src_point_2.x - (src_point_1.y - src_point_2.y);
	src_point_3.y = src_point_2.y + (src_point_1.x - src_point_2.x);


	float alphapose_image_center_x = output_image_width / 2;
	float alphapose_image_center_y = output_image_height / 2;

	cv::Point2f dst_point_1;
	dst_point_1.x = alphapose_image_center_x;
	dst_point_1.y = alphapose_image_center_y;

	cv::Point2f dst_point_2;
	dst_point_2.x = alphapose_image_center_x;
	dst_point_2.y = alphapose_image_center_y - output_image_width * 0.5;

	cv::Point2f dst_point_3;
	dst_point_3.x = dst_point_2.x - (dst_point_1.y - dst_point_2.y);
	dst_point_3.y = dst_point_2.y + (dst_point_1.x - dst_point_2.x);


	cv::Point2f srcPoints[3];
	srcPoints[0] = src_point_1;
	srcPoints[1] = src_point_2;
	srcPoints[2] = src_point_3;

	cv::Point2f dstPoints[3];
	dstPoints[0] = dst_point_1;
	dstPoints[1] = dst_point_2;
	dstPoints[2] = dst_point_3;

	// get affine matrix
	// 获取仿射矩阵
	cv::Mat affineTransform;
	if (inverse)
	{
		affineTransform = cv::getAffineTransform(dstPoints, srcPoints);
	}
	else
	{
		affineTransform = cv::getAffineTransform(srcPoints, dstPoints);
	}

	return affineTransform;
}

static float LetterBoxImage(
	const cv::Mat& image,
	cv::Mat& out_image,
	const cv::Size& new_shape = cv::Size(640, 640),
	int stride = 32,
	const cv::Scalar& color = cv::Scalar(114, 114, 114),
	bool fixed_shape = false,
	bool scale_up = true) 
{
	cv::Size shape = image.size();
	float r = std::min((float)new_shape.height / (float)shape.height, (float)new_shape.width / (float)shape.width);

	if (!scale_up) {
		r = std::min(r, 1.0f);
	}

	int newUnpad[2]{
		(int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r) };

	cv::Mat tmp;
	if (shape.width != newUnpad[0] || shape.height != newUnpad[1]) {
		cv::resize(image, tmp, cv::Size(newUnpad[0], newUnpad[1]));
	}
	else {
		tmp = image.clone();
	}

	float dw = new_shape.width - newUnpad[0];
	float dh = new_shape.height - newUnpad[1];

	if (!fixed_shape) {
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}

	int top = int(0);
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(0));
	int right = int(std::round(dw + 0.1f));

	cv::copyMakeBorder(tmp, out_image, top, bottom, left, right, cv::BORDER_CONSTANT, color);

	return 1.0f / r;
}

#endif // !_RTM_POSE_UTILS_H_
