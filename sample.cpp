#include <iostream>
#include <cmath>
#include <string>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "json.hpp"

using namespace std;
using namespace cv;
using json = nlohmann::json;

cv::Mat project_on_ground(cv::Mat& img, cv::Mat& T_CG,
	cv::Mat& K_C, vector<double>& D_C,
	cv::Mat& K_G, int rows, int cols) {
	// 	cout<<"--------------------Init p_G and P_G------------------------"<<endl;
	cv::Mat p_G = cv::Mat::ones(3, rows * cols, CV_64FC1);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			p_G.at<double>(0, cols * i + j) = j;
			p_G.at<double>(1, cols * i + j) = i;
		}
	}

	cv::Mat P_G = cv::Mat::ones(3, rows * cols, CV_64FC1);
	P_G = K_G.inv() * p_G;
	//P_G = K_G.inv() * K_G* p_G;
	// 	cout<<"--------------------Init P_GF------------------------"<<endl;

	cv::Mat P_GC = cv::Mat::zeros(3, rows * cols, CV_64FC1);
	cv::Mat H = cv::Mat::zeros(3, 3, CV_64FC1);

	int idx[3] = { 0,1,3 };
	for (int i = 0; i < 9; i++)
		H.at<double>(i / 3, i % 3) = T_CG.at<double>(i / 3, idx[i % 3]);

	P_GC = H * P_G;

	// 	cout<<"--------------------Init P_GF1------------------------"<<endl;
	cv::Mat P_GC1 = cv::Mat::zeros(1, rows * cols, CV_64FC2);
	std::vector<cv::Mat> channels(2);
	cv::split(P_GC1, channels);
	channels[0] = P_GC(cv::Rect(0, 0, rows * cols, 1)) / P_GC(cv::Rect(0, 2, rows * cols, 1));
	channels[1] = P_GC(cv::Rect(0, 1, rows * cols, 1)) / P_GC(cv::Rect(0, 2, rows * cols, 1));
	cv::merge(channels, P_GC1);

	// 	cout<<"--------------------Init p_GF------------------------"<<endl;
	cv::Mat p_GC = cv::Mat::zeros(1, rows * cols, CV_64FC2);
	cv::fisheye::distortPoints(P_GC1, p_GC, K_C, D_C);
	cv::Mat p_GC_table = p_GC.reshape(0, rows);

	p_GC.reshape(rows, cols);

	cv::Mat p_GC_table_32F;
	p_GC_table.convertTo(p_GC_table_32F, CV_32FC2);

	cv::Mat img_GC;
	cv::remap(img, img_GC, p_GC_table_32F, cv::Mat(), cv::INTER_LINEAR); //INTER_LINEAR INTER_NEAREST

	return img_GC;
}

cv::Point2d ProjectFisheyePoint(
	const cv::Point2d& fisheye_point,
	const cv::Mat& intrinsic_matrix,
	const cv::Mat& extrinsic_matrix,
	const cv::Mat& camera_ground,
	const std::vector<double>& distortion_coefficients) {
	// 去畸变
	std::vector<cv::Point2d> fisheye_point_vec(1), undistorted_point_vec(1);
	fisheye_point_vec.at(0) = fisheye_point;
	cv::fisheye::undistortPoints(
		fisheye_point_vec, undistorted_point_vec,
		intrinsic_matrix, distortion_coefficients);
	cv::Mat point_mat = cv::Mat::ones(3, 1, CV_64FC1);
	point_mat.at<double>(0, 0) = undistorted_point_vec.at(0).x;
	point_mat.at<double>(1, 0) = undistorted_point_vec.at(0).y;

	// 获取基于外参的旋转矩阵
	cv::Mat h = cv::Mat::zeros(3, 3, CV_64FC1);
	std::vector<size_t> idxs = { 0, 1, 3 };
	for (size_t i = 0; i < 3; i++) {
		for (size_t j = 0; j < 3; j++) {
			h.at<double>(i, j) = extrinsic_matrix.at<double>(i, idxs.at(j));
		}
	}

	// 将点进行旋转
	cv::Mat tmp = h.inv() * point_mat;
	cv::Point2d world_point;
	world_point.x = tmp.at<double>(0, 0) / tmp.at<double>(2, 0);
	world_point.y = tmp.at<double>(1, 0) / tmp.at<double>(2, 0);

	cv::Point2d result;
	result.x = world_point.x * camera_ground.at<double>(0,0) + camera_ground.at<double>(0, 2);
	result.y = world_point.y * camera_ground.at<double>(1,1) + camera_ground.at<double>(1, 2);

	return result;
}

void GenerateMergeImg(
	std::vector<cv::Mat>& source_images,
	std::vector<cv::Mat>& intrinsic_matrixs,
	std::vector<cv::Mat>& extrinsic_matrixs,
	std::vector<std::vector<double>>& distortion_coefficients,
	cv::Mat& output_img,
	cv::Mat& kg,
	int img_height,
	int img_width
) {
	std::vector<cv::Mat> project_images;
	project_images.resize(4);

	int centerX = img_width / 2;
	int centerY = img_height / 2  - 30;
	int carWidth = 140;
	int carHeight = 240;
	cv::Point tl(centerX - 0.5*carWidth, centerY - 0.2*carHeight);
	cv::Point tr(centerX + 0.5*carWidth, centerY - 0.2*carHeight);
	cv::Point bl(centerX - 0.5*carWidth, centerY + 0.8*carHeight);
	cv::Point br(centerX + 0.5*carWidth, centerY + 0.8*carHeight);

	for (int i = 0; i < 4; i++) {
		project_images.at(i) = project_on_ground(
			source_images.at(i),
			extrinsic_matrixs.at(i),
			intrinsic_matrixs.at(i),
			distortion_coefficients.at(i),
			kg,
			img_height,
			img_width);
	}

	double theta = 1.0;
	output_img = cv::Mat::zeros(img_height, img_width, CV_8UC3);
	for (int i = 0; i < project_images.at(0).rows; i++) {
		for (int j = 0; j < project_images.at(0).cols; j++) {
			if ((i <= theta * (j - tl.x) + tl.y)
				&& (i <= -theta * (j - tr.x) + tr.y))
			{
				output_img.at<cv::Vec3b>(i, j) = project_images.at(0).at<cv::Vec3b>(i, j);
			}
			else if ((i > theta * (j - tl.x) + tl.y)
				&& (i < -theta * (j - bl.x) + bl.y)
				&& (j < project_images.at(0).cols / 2))
			{
				output_img.at<cv::Vec3b>(i, j) = project_images.at(2).at<cv::Vec3b>(i, j);
			}
			else if ((i > -theta * (j - tr.x) + tr.y)
				&& (i < theta * (j - br.x) + br.y)
				&& (j >= project_images.at(0).cols / 2))
			{
				output_img.at<cv::Vec3b>(i, j) = project_images.at(3).at<cv::Vec3b>(i, j);
			}
			else if ((i >= -theta * (j - bl.x) + bl.y)
				&& (i >= theta * (j - br.x) + br.y))
			{
				output_img.at<cv::Vec3b>(i, j) = project_images.at(1).at<cv::Vec3b>(i, j);
			}
		}
	}

	for (int i = tl.y; i < bl.y; i++)
		for (int j = tl.x; j < tr.x; j++)
			output_img.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);

}

void ReadSample() {
	std::vector<string> image_path = {
		"images/dataset/dotted_line/crowded/v2/img/frm_0_0201.jpg",
		"images/dataset/dotted_line/crowded/v2/img/frm_1_0201.jpg",
		"images/dataset/dotted_line/crowded/v2/img/frm_2_0201.jpg",
		"images/dataset/dotted_line/crowded/v2/img/frm_3_0201.jpg"
	};
	
	std::vector<string> lane_marking_path = {
		"images/dataset/dotted_line/crowded/v2/lane_marking/frm_0_0201.json",
		"images/dataset/dotted_line/crowded/v2/lane_marking/frm_1_0201.json",
		"images/dataset/dotted_line/crowded/v2/lane_marking/frm_2_0201.json",
		"images/dataset/dotted_line/crowded/v2/lane_marking/frm_3_0201.json"
	};
	std::vector<cv::Mat> images(4);
	std::vector<std::vector<cv::Point2d>> points(4);
	std::vector<json> points_json;
	std::ifstream fin;

	for (int i = 0; i < 4; i++) {
		images.at(i) = cv::imread(image_path.at(i));
		fin.open(lane_marking_path.at(i));
		if (fin.is_open()) {
			json label_data = json::parse(fin);
			int line_count = (label_data["label_data"]["line_count"]).get<int>();
			if (line_count > 0) {	// there may be no lane detected in the frame.
				std::vector<json> lane_json;
				lane_json = (label_data["label_data"]["point_mask"]).get<std::vector<json>>();

				for (size_t j = 0; j < lane_json.size(); j++) {
					points_json = lane_json.at(j).get<std::vector<json>>();
					for (size_t k = 0; k < points_json.size(); k++) {
						cv::Point2d tmp_point;
						tmp_point.x = points_json.at(k)["x"].get<double>();
						tmp_point.y = points_json.at(k)["y"].get<double>();
						points.at(i).emplace_back(tmp_point);
					}
				}
			}
			else {
				return;
			}
		}
		else {
			std::cout << "Can not open " << lane_marking_path.at(i) << endl;
		}
		fin.close();
	}

	std::vector<cv::Mat> images_copy(4);
	for (int i = 0; i < 4; i++) {
		images_copy.at(i) = images.at(i).clone();
		for (size_t j = 0; j < points.at(i).size(); j++) {
			cv::circle(images_copy.at(i), points.at(i).at(j), 2, cv::Scalar(0, 0, 255), -1);
		}
		imshow(std::to_string(i + 1), images_copy.at(i));
	}



	string parameter_path = "images/dataset/dotted_line/crowded/v2/parameters/2_parameters.json";
	std::vector<cv::Mat> intrinsic_matrixs(4);
	std::vector<cv::Mat> rotation_matrixs(4);
	std::vector<cv::Mat> extrinsic_matrixs(4);
	std::vector<cv::Mat> camera_position(4);
	std::vector<std::vector<double>> distortion_coefficients(4);
	fin.open(parameter_path);
	json parameter_data = json::parse(fin);
	std::vector<std::vector<double>> intrinsic_double(4);
	std::vector<std::vector<double>> rotation_double(4);
	std::vector<std::vector<double>> camera_position_double(4);

	intrinsic_double.at(0) = parameter_data["k_front"].get<std::vector<double>>();
	intrinsic_double.at(1) = parameter_data["k_rear"].get<std::vector<double>>();
	intrinsic_double.at(2) = parameter_data["k_left"].get<std::vector<double>>();
	intrinsic_double.at(3) = parameter_data["k_right"].get<std::vector<double>>();

	rotation_double.at(0) = parameter_data["r_front"].get<std::vector<double>>();
	rotation_double.at(1) = parameter_data["r_rear"].get<std::vector<double>>();
	rotation_double.at(2) = parameter_data["r_left"].get<std::vector<double>>();
	rotation_double.at(3) = parameter_data["r_right"].get<std::vector<double>>();

	camera_position_double.at(0) = parameter_data["t_front"].get<std::vector<double>>();
	camera_position_double.at(1) = parameter_data["t_rear"].get<std::vector<double>>();
	camera_position_double.at(2) = parameter_data["t_left"].get<std::vector<double>>();
	camera_position_double.at(3) = parameter_data["t_right"].get<std::vector<double>>();

	distortion_coefficients.at(0) = parameter_data["d_front"].get<std::vector<double>>();
	distortion_coefficients.at(1) = parameter_data["d_rear"].get<std::vector<double>>();
	distortion_coefficients.at(2) = parameter_data["d_left"].get<std::vector<double>>();
	distortion_coefficients.at(3) = parameter_data["d_right"].get<std::vector<double>>();
	for (int i = 0; i < 4; i++) {
		intrinsic_matrixs.at(i) = cv::Mat::zeros(3, 3, CV_64FC1);
		rotation_matrixs.at(i) = cv::Mat::zeros(3, 3, CV_64FC1);
		extrinsic_matrixs.at(i) = cv::Mat::zeros(4, 4, CV_64FC1);
		camera_position.at(i) = cv::Mat::zeros(3, 1, CV_64FC1);
		for (int j = 0; j < 9; j++) {
			intrinsic_matrixs.at(i).at<double>(j / 3, j % 3) = intrinsic_double.at(i).at(j);
			rotation_matrixs.at(i).at<double>(j / 3, j % 3) = rotation_double.at(i).at(j);
			extrinsic_matrixs.at(i).at<double>(j / 3, j % 3) = rotation_double.at(i).at(j);
		}
		for (int j = 0; j < 3; j++) {
			camera_position.at(i).at<double>(j, 0) = camera_position_double.at(i).at(j);
		}
		camera_position.at(i).at<double>(0, 0) = -camera_position.at(i).at<double>(0, 0);
		cv::Mat tmp_t = rotation_matrixs.at(i) * camera_position.at(i);
		for (int j = 0; j < 3; j++) {
			extrinsic_matrixs.at(i).at<double>(j, 3) = tmp_t.at<double>(j, 0);
		}
		extrinsic_matrixs.at(i).at<double>(3, 3) = 1.0;
	}

	int project_image_height = 770;
	int project_image_width = 880;

	cv::Mat kg = cv::Mat::zeros(3, 3, CV_64FC1);
	kg = cv::Mat::zeros(3, 3, CV_64FC1);
	kg.at<double>(0, 0) = 70;
	kg.at<double>(1, 1) = 70;
	kg.at<double>(0, 2) = project_image_width / 2.0;
	kg.at<double>(1, 2) = project_image_height / 2.0;
	kg.at<double>(2, 2) = 1.0;

	cv::Mat merge_image;
	GenerateMergeImg(images, intrinsic_matrixs, extrinsic_matrixs, 
		distortion_coefficients, merge_image, kg, project_image_height, project_image_width);

	
	for (int i = 0; i < 4; i++) {
		for (size_t j = 0; j < points.at(i).size(); j++) {
			cv::Point2d tmp_point = ProjectFisheyePoint(
					points.at(i).at(j), intrinsic_matrixs.at(i), extrinsic_matrixs.at(i),
					kg, distortion_coefficients.at(i)
				);

			cv::circle(merge_image, tmp_point, 3 , cv::Scalar(255- 50 * i, 0, 50 + 50 * i), -1);
		}
	}
	imshow("merge_image", merge_image);
	cv::waitKey(0);
}

int main(int argc, char** argv){
	ReadSample();
	return 0;
}