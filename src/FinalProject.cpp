/*
Computer Vision Final Project - Road Sign Recognition
Valente Alex - 1173742
UniPD - A.A. 2017-2018
*/
#include "SVMclassifier.h"
#include "RoadSignDetector.h"

#include <cmath>
#include <cstring>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/features2d.hpp>
#include <opencv2/ml.hpp>

/*
The main function, entry point of the application
@param  arcg  the number of command-line arguments (including the name of the executable file)
@param  argv  the values of command-line arguments
*/
int main(int argc, char** argv) {
	std::cout << "Road Sign Detection" << std::endl;
	
	// Parameters have to be read from the command line, if available
	bool realtime = false;
	std::string images_path;
	std::string classifiers_path;

	// Uncomment to consider default paths
	/*images_path = "C:/Users/Alex/source/repos/cv_final_project/data/dataset/";
	classifiers_path = "C:/Users/Alex/source/repos/cv_final_project/data/classifiers/";*/

	if (argc > 2) {
		realtime = (std::strcmp(argv[1],"realtime") == 0);
		if (realtime) {
			classifiers_path = argv[2];
		}
		else {
			classifiers_path = argv[1];
			if (argc == 3)
				images_path = argv[2];
		}//if else
	}//if parameters

	if (classifiers_path.empty()) {
		std::cerr << "ERROR: need for classifiers path";
		return 1;
	}//if

	// Summary of parameters used
	std::cout << "\nRealtime: " << realtime << std::endl;
	std::cout << "Classifiers path: " << classifiers_path << std::endl;
	if(!realtime)
		std::cout << "Images path: " << images_path << std::endl;

	// Initialization
	std::cout << "\nInitialization in progress..." << std::endl;
	RoadSignDetector det(classifiers_path);

	// Flag to decide whether to view debug information or not
	det.setDebugMode(false);
	
	// Start of processing
	if (realtime) {
		// The images that will be passed to the detector are captured from a camera for realtime recognition
		int numcamera = 1;
		std::cout << "\nCamera number: ";
		std::cin >> numcamera;
		cv::VideoCapture cap(numcamera);
		if (!cap.isOpened())
			return 1;
		while (true) {
			cv::Mat image;
			cap >> image;
			det.loadImage(image);
			det.preprocess();
			cv::Mat result = det.getClassificationResults();
			cv::imshow("Result", result);

			// Need for a small pause, otherwise there will not be enough time to process an image before a new image becomes available
			cv::waitKey(10);
		}//while
	}
	else {
		std::vector<cv::String> paths;
		if (images_path.empty()) {
			std::cout << "\nInsert the path of the images to be processed: ";
			std::cin >> images_path;
		}//if

		// Load of the paths of all files in the selected folder
		cv::glob(images_path, paths);
		for (int i = 0; i < paths.size(); i++) {
			det.loadImage(paths[i]);
			det.preprocess();
			cv::Mat result = det.getClassificationResults();
			cv::imshow("Result", result);

			// Let the user view the resulting detection
			cv::waitKey(0);
		}//for
	}//if else

	return 0;

}//main