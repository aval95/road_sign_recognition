/*
Computer Vision Final Project - Road Sign Recognition
Valente Alex - 1173742
UniPD - A.A. 2017-2018

Support Vector Machine multiclass classifier for road sign recognition
*/
#ifndef SVM__CLASSIFIER__H
#define SVM__CLASSIFIER__H

#include <opencv2/objdetect.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>
#include "DetectedMat.h"

class SVMclassifier {
public:

	SVMclassifier();

	SVMclassifier(std::vector<cv::String> svm_classifier_files);

	void init();

	std::vector<cv::Ptr<cv::ml::SVM>> svmClassTraining(std::string classpath);

	int svmClassPrediction(DetectedMat image);

	std::string getClassName(int num_class);

	void setDebugMode(bool debug);

private:

	int HOG_DESCRIPTOR_SIZE = 2304;
	int NUM_CLASSES = 14;
	float RECOGNITION_THRESHOLD = 0.20;

	std::vector<cv::Ptr<cv::ml::SVM>> m_svm;  // Need for one classifier per road sign class
	std::vector<std::string> m_label_names;
	bool m_debug_mode_flag = false;
};

#endif

