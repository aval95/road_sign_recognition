/*
	Computer Vision Final Project - Road Sign Recognition
	Valente Alex - 1173742
	UniPD - A.A. 2017-2018
*/
#ifndef ROAD__SIGN__DETECTOR__H
#define ROAD__SIGN__DETECTOR__H

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>
#include <cmath>
#include <algorithm>   // needed for std::max() function
#include "DetectedMat.h"
#include "SVMclassifier.h"



class RoadSignDetector {
public:

	RoadSignDetector(std::string classifiers_path);

	RoadSignDetector(std::string classifiers_path, cv::Mat image);

	void loadImage(cv::Mat image);

	void loadImage(std::string filename);

	std::vector<DetectedMat> preprocess();

	std::vector<int> classifyRoadSigns();

	cv::Mat getClassificationResults();

	void setDebugMode(bool debug);

private:

	int NUM_CLASSES = 14;
	int MAX_OVERLAPPED_AREA = 30;
	int SIZE_THRESHOLD = 28;
	float ASPECT_RATIO_THRESHOLD = 1.5;

	SVMclassifier m_svm_classifier;
	cv::Mat m_src_image;
	cv::Mat m_dst_image;
	bool m_debug_mode_flag;
	cv::RNG rng; // Random number generator, for rectangles color
	std::vector<cv::Rect> m_found_rectangles;
	std::vector<DetectedMat> m_found_signs;
	bool m_preprocessed;
	
	void init(std::string classifiers_path);

	bool findCircles(cv::Mat image);

	void autoTunedCanny(cv::Mat image, int* lowThreshold, int* highThreshold);

	std::vector<cv::Rect> getNoOverlappingRectangles(std::vector<cv::Rect> rectVector, int max_overlap);

	cv::Mat findRedSigns(cv::Mat image);

	cv::Mat findBlueSigns(cv::Mat image);

};

#endif
