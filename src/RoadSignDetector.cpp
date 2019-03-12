/*
Computer Vision Final Project - Road Sign Recognition
Valente Alex - 1173742
UniPD - A.A. 2017-2018
*/
#include "RoadSignDetector.h"

/*
Structure used to give an order policy to the std::order function in preprocess()
*/
struct sort_order {
	inline bool operator() (const cv::Rect& rect1, const cv::Rect& rect2)
	{
		return (rect2.area() < rect1.area() && (rect2.x < rect1.x || rect2.y < rect1.y));
	}
};



/*
Constructor of RoadSignDetector class
@param  classifiers_path  the path of the files of the SVM trained classifiers
*/
RoadSignDetector::RoadSignDetector(std::string classifiers_path) {
	init(classifiers_path);
}//RoadSignDetector



/*
Constructor of RoadSignDetector class
@param  classifiers_path  the path of the files of the SVM trained classifiers
@param  image  the image to be processed to detect and classify road signs
*/
RoadSignDetector::RoadSignDetector(std::string classifiers_path, cv::Mat image) : m_src_image(image) {
	m_dst_image = m_src_image.clone();
	init(classifiers_path);
}//RoadSignDetector



/*
Function that loads an image to be processed by the detector
@param  image  the image to be processed to detect and classify road signs
*/
void RoadSignDetector::loadImage(cv::Mat image) {
	if (image.empty()) {
		std::cerr << "ERROR: empty image loaded" << std::endl;
		exit(0);
	}//if
	m_src_image = image;
	m_dst_image = m_src_image.clone();
	m_preprocessed = false;
	m_found_rectangles = std::vector<cv::Rect>();
	m_found_signs = std::vector<DetectedMat>();
}//loadImage



/*
Function that loads an image to be processed by the detector
@param  filename  the path and file name of the file containing the image to be processed
*/
void RoadSignDetector::loadImage(std::string filename) {
	m_src_image = cv::imread(filename);
	m_dst_image = m_src_image.clone();
	m_preprocessed = false;
	m_found_rectangles = std::vector<cv::Rect>();
	m_found_signs = std::vector<DetectedMat>();
}//loadImage



/*
Function used to initialize the detector, reading the classifiers files and creating an instance of the SVM classifier
@param  classifiers_path  the path of the files of the SVM trained classifiers  
*/
void RoadSignDetector::init(std::string classifiers_path) {
	std::vector<cv::String> classifiers_name(NUM_CLASSES);
	for (int i = 0; i < classifiers_name.size(); i++) {
		classifiers_name[i] = classifiers_path + "svm_trained_" + std::to_string(i) + ".xml";
	}//for
	m_svm_classifier = SVMclassifier(classifiers_name);
	std::cout << "Initialization completed" << std::endl;
	m_debug_mode_flag = false;
	m_svm_classifier.setDebugMode(false);
	rng = cv::RNG(12345);
	m_preprocessed = false;
	m_found_rectangles = std::vector<cv::Rect>();
	m_found_signs = std::vector<DetectedMat>();
}//init



/*
Sets the debug mode on or off, to view or not debug images and messages
@param  debug  the status of the debug mode: true = on, false = off
*/
void RoadSignDetector::setDebugMode(bool debug) {
	m_debug_mode_flag = debug;
	m_svm_classifier.setDebugMode(debug);
}//setDebugMode



/*
Extracts from the input image only the red parts, according to determined parameters
@param  image  the image where the red parts will be extracted
return  the mask indicating only the red parts of the input image
*/
cv::Mat RoadSignDetector::findRedSigns(cv::Mat image) {
	cv::Mat enhancedImage = image.clone();

	// Color enhancement of R component in RGB color space
	for (int x = 0; x < enhancedImage.cols; x++) {
		for (int y = 0; y < enhancedImage.rows; y++) {
			cv::Vec3b components = enhancedImage.at<cv::Vec3b>(y, x);
			int S = components[0] + components[1] + components[2];
			if (S > 0) {
				enhancedImage.at<cv::Vec3b>(y, x).val[0] = cv::max(0, cv::min(components[0] - components[1], components[0] - components[2]) / S);
			}//if not 0
		}//for
	}//for

	if (m_debug_mode_flag) {
		cv::namedWindow("Image Red");
		cv::imshow("Image Red", enhancedImage);
	}//if

	// Convert the red enhanced image to HSV color space in order to consider only effective red parts (depending on Hue value)
	cv::Mat hsvImage;
	cv::cvtColor(enhancedImage, hsvImage, CV_BGR2HSV);
	cv::Mat redMask(hsvImage.rows, hsvImage.cols, CV_8UC3, cv::Scalar(0));

	// Red extraction
	for (int x = 0; x < hsvImage.cols; x++) {
		for (int y = 0; y < hsvImage.rows; y++) {
			cv::Vec3b components = hsvImage.at<cv::Vec3b>(y, x);
			if ((components.val[0] <= 15 || components.val[0] >= 245) && components.val[1] >= 50 && components.val[2] > 20) {
				redMask.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
			}//if
		}//for
	}//for

	return redMask;
}//findRedSigns



 /*
 Extracts from the input image only the blue parts, according to determined parameters
 @param  image  the image where the blue parts will be extracted
 return  the mask indicating only the blue parts of the input image
 */
cv::Mat RoadSignDetector::findBlueSigns(cv::Mat image) {
	cv::Mat enhancedImage = image.clone();

	// Color enhancement of B component in RGB color space
	for (int x = 0; x < enhancedImage.cols; x++) {
		for (int y = 0; y < enhancedImage.rows; y++) {
			cv::Vec3b components = enhancedImage.at<cv::Vec3b>(y, x);
			int S = components[0] + components[1] + components[2];
			if (S > 0) {
				enhancedImage.at<cv::Vec3b>(y, x).val[2] = cv::max(0, cv::min(components[2] - components[1], components[2] - components[0]) / S);
			}//if not 0
		}//for
	}//for

	if (m_debug_mode_flag) {
		cv::namedWindow("Image Blue");
		cv::imshow("Image Blue", enhancedImage);
	}//if

	 // Convert the blue enhanced image to HSV color space in order to consider only effective blue parts (depending on Hue value)
	cv::Mat hsvImage;
	cv::cvtColor(enhancedImage, hsvImage, CV_BGR2HSV);
	cv::Mat blueMask(hsvImage.rows, hsvImage.cols, CV_8UC3, cv::Scalar(0));

	// Blue extraction
	for (int x = 0; x < hsvImage.cols; x++) {
		for (int y = 0; y < hsvImage.rows; y++) {
			cv::Vec3b components = hsvImage.at<cv::Vec3b>(y, x);
			if ((components.val[0] >= 100 && components.val[0] <= 150) && components.val[1] >= 100 && components.val[2] > 20) {
				blueMask.at<cv::Vec3b>(y, x) = cv::Vec3b(255, 255, 255);
			}//if
		}//for
	}//for

	return blueMask;
}//findBlueSigns



/*
Preprocesses the input image to help the classifiers recognize the class of the road sign detected
It uses:
- Red signs detection
- Blue signs detection
For each component image it uses:
- Canny edge detection with auto-tuning of parameters
- Contours detection
- Convex hull to have only convex regions (road signs have convex shapes only)
- Application of the detection mask to the original image
- Bounding with rectangles all the convex regions found
- Discarding of rectangles:
  - Because they are too small or with an aspect ratio too different from the square to contain a road sign
  - Because they contain other rectangles or they overlap another one (note that all signs have an outer shape 
    and the same shape inside, I maintain only the external one)
- Resize of rectangles to avoid cutting edges with too strict boundaries
- Extraction of the rectangular regions found from the original image
For each rectangular region extracted:
- Seach for circles (if one or more circles are present I can exclude class 13 (Warning) 
  for the classification phase, if no circles I know that classes 0-12 are not possible (but it can be a not 
  known road sign, not necessarly a class 13))
return  the vector with the detected rectangular regions where the signs have to be classified
*/
std::vector<DetectedMat> RoadSignDetector::preprocess() {
	
	cv::Mat image = m_src_image.clone();

	// Red and blue regions detection, masks calculated
	cv::Mat red_signs_mask = findRedSigns(image);
	cv::Mat blue_signs_mask = findBlueSigns(image);

	// I want that points in blue_signs_mask are not also part of red_signs_mask -> need for NAND
	cv::Mat neg_red_signs_mask;
	cv::bitwise_not(red_signs_mask, neg_red_signs_mask);
	cv::bitwise_and(blue_signs_mask, neg_red_signs_mask, blue_signs_mask);

	// Apply masks to the input image
	cv::Mat red_signs, blue_signs;
	cv::bitwise_and(image, red_signs_mask, red_signs);
	cv::bitwise_and(image, blue_signs_mask, blue_signs);

	// Postprocess to delete points that are isolated (they certainly will not be road signs)
	cv::medianBlur(red_signs, red_signs, 5);
	cv::medianBlur(blue_signs, blue_signs, 5);
	if (m_debug_mode_flag) {
		cv::imshow("Red image", red_signs);
		cv::imshow("Blue image", blue_signs);
	}//if


	std::vector<cv::Mat> signs_components = { red_signs, blue_signs };

	// Phases repeated for each color component
	for (int component_num = 0; component_num < signs_components.size(); component_num++) {

		// Auto-tuning Canny of the considered component image
		int lowThreshold;
		int highThreshold;
		autoTunedCanny(signs_components[component_num], &lowThreshold, &highThreshold);

		// Canny edge detection
		cv::Mat gray;
		cv::cvtColor(signs_components[component_num], gray, CV_BGR2GRAY);
		cv::Mat canny_output;
		cv::blur(gray, gray, cv::Size(3, 3));
		Canny(gray, canny_output, lowThreshold, highThreshold * 2, 3);


		// Find contours in the Canny image and draw them
		std::vector<std::vector<cv::Point>> contours;
		std::vector<cv::Vec4i> hierarchy;
		findContours(canny_output, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		cv::Mat contours_mat = cv::Mat::zeros(canny_output.size(), CV_8UC3);
		for (size_t i = 0; i < contours.size(); i++) {
			cv::Scalar color = cv::Scalar(255, 255, 255);
			drawContours(contours_mat, contours, i, color, CV_FILLED);
		}//for

		// Show in a window
		if (m_debug_mode_flag) {
			cv::namedWindow("Contours " + std::to_string(component_num), cv::WINDOW_AUTOSIZE);
			imshow("Contours " + std::to_string(component_num), contours_mat);
		}//if


		// Find the convex hull object for each contour
		std::vector<std::vector<cv::Point> >hull(contours.size());
		for (int i = 0; i < contours.size(); i++) {
			convexHull(cv::Mat(contours[i]), hull[i], false);
		}//for

		// Draw contours + hull results
		for (int i = 0; i < contours.size(); i++) {
			cv::Scalar color = cv::Scalar(255, 255, 255);
			drawContours(contours_mat, hull, i, color, CV_FILLED);
		}//for

		// Show in a window
		if (m_debug_mode_flag) {
			cv::namedWindow("Hull " + std::to_string(component_num), CV_WINDOW_AUTOSIZE);
			imshow("Hull " + std::to_string(component_num), contours_mat);
		}//if


		// Application of the detection mask to the original image
		cv::Mat masked;
		cv::bitwise_and(image, contours_mat, masked);

		if (m_debug_mode_flag) {
			cv::namedWindow("Masked with contours " + std::to_string(component_num), CV_WINDOW_AUTOSIZE);
			cv::imshow("Masked with contours " + std::to_string(component_num), masked);
		}//if


		// Find rectangles bounding convex regions found
		cv::Mat maskedPhase1 = masked.clone();
		std::vector<cv::Rect> boundRect(hull.size());
		for (size_t i = 0; i < hull.size(); i++) {
			boundRect[i] = cv::boundingRect(hull[i]);
			cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
			cv::rectangle(maskedPhase1, boundRect[i].tl(), boundRect[i].br(), color, 2);
		}//for

		if (m_debug_mode_flag) {
			cv::namedWindow("Masked with rectangles Phase1 " + std::to_string(component_num), CV_WINDOW_AUTOSIZE);
			cv::imshow("Masked with rectangles Phase1 " + std::to_string(component_num), maskedPhase1);
			printf("\n\nPhase 1\nNumber of rectangles = %d", boundRect.size());
			for (int i = 0; i < boundRect.size(); i++) {
				printf("\nx = %d  y = %d  w = %d  h = %d", boundRect[i].x, boundRect[i].y, boundRect[i].width, boundRect[i].height);
			}//for
			printf("\n");
		}//if


		// Discard rectangles that can not contain road signs because of size and aspect ratio
		cv::Mat maskedPhase2 = masked.clone();
		std::vector<cv::Rect> realBoundRect;
		for (int i = 0; i < boundRect.size(); i++) {
			cv::Rect testRect = boundRect[i];
			if (!(testRect.width < testRect.height / ASPECT_RATIO_THRESHOLD || testRect.height < testRect.width / ASPECT_RATIO_THRESHOLD || testRect.height < SIZE_THRESHOLD || testRect.width < SIZE_THRESHOLD)) {
				realBoundRect.push_back(testRect);
			}//if
		}//for

		// Draw maintained rectangles
		for (size_t i = 0; i < realBoundRect.size(); i++) {
			cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
			cv::rectangle(maskedPhase2, realBoundRect[i].tl(), realBoundRect[i].br(), color, 2);
		}//for

		if (m_debug_mode_flag) {
			cv::namedWindow("Masked with rectangles Phase2 " + std::to_string(component_num), CV_WINDOW_AUTOSIZE);
			cv::imshow("Masked with rectangles Phase2 " + std::to_string(component_num), maskedPhase2);
			printf("\n\nPhase 2\nNumber of rectangles = %d", realBoundRect.size());
			for (int i = 0; i < realBoundRect.size(); i++) {
				printf("\nx = %d  y = %d  w = %d  h = %d", realBoundRect[i].x, realBoundRect[i].y, realBoundRect[i].width, realBoundRect[i].height);
			}//for
			printf("\n");
		}//if


		// Discard overlapping rectangles
		std::vector<cv::Rect> realRealBoundRect;
		realRealBoundRect = getNoOverlappingRectangles(realBoundRect, MAX_OVERLAPPED_AREA);


		// Increase the size of maintained rectangles to help the SVM classify signs
		// The resize will be size-dependent: the biggest the size the less the increase in size
		for (int i = 0; i < realRealBoundRect.size(); i++) {
			int x = realRealBoundRect[i].x;
			int y = realRealBoundRect[i].y;
			int width = realRealBoundRect[i].width;
			int height = realRealBoundRect[i].height;

			int diff_width = std::max(0, (int)(20 - 0.2 * width));
			int diff_height = std::max(0, (int)(20 - 0.2 * height));

			int new_x = std::max(0, (int)(x - diff_width / 2));
			int new_y = std::max(0, (int)(y - diff_height / 2));
			int new_width = std::min(signs_components[component_num].cols - new_x - 1, (int)(width + diff_width));
			int new_height = std::min(signs_components[component_num].rows - new_y - 1, (int)(height + diff_height));

			realRealBoundRect[i].x = new_x;
			realRealBoundRect[i].y = new_y;
			realRealBoundRect[i].width = new_width;
			realRealBoundRect[i].height = new_height;
		}//for



		/* For each preserved rectangle after the discard phases:
		   - draw it
		   - find circles (if any)
		   - push the image contained in the rectangle in m_found_signs as DetectedMat (it contains the image and additional information)
		*/
		cv::Mat maskedPhase3 = masked.clone();
		for (size_t i = 0; i < realRealBoundRect.size(); i++) {
			cv::Scalar color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
			cv::rectangle(maskedPhase3, realRealBoundRect[i].tl(), realRealBoundRect[i].br(), color, 2);

			// Search for circles
			bool hasCircles = findCircles(m_src_image(realRealBoundRect[i]));

			// Detected Mat
			DetectedMat detected(m_src_image(realRealBoundRect[i]), component_num == 0, component_num == 1, hasCircles);

			// Push
			m_found_signs.push_back(detected);
			m_found_rectangles.push_back(realRealBoundRect[i]);
		}//for m_found_signs

		if (m_debug_mode_flag) {
			cv::namedWindow("Masked with rectangles Phase3 " + std::to_string(component_num), CV_WINDOW_AUTOSIZE);
			cv::imshow("Masked with rectangles Phase3 " + std::to_string(component_num), maskedPhase3);
			printf("\n\nPhase 3\nNumber of rectangles = %d", realRealBoundRect.size());
			for (int i = 0; i < realRealBoundRect.size(); i++) {
				printf("\nx = %d  y = %d  w = %d  h = %d", realRealBoundRect[i].x, realRealBoundRect[i].y, realRealBoundRect[i].width, realRealBoundRect[i].height);
			}//for
			printf("\n");
		}//if

	}//for signs_components

	// Pre-processing done
	m_preprocessed = true;

	return m_found_signs;

}//preprocess



/*
Function that, given the images of the found signs, returns the original image with the detected signs bounded by rectangles and the result of the classification
return  the output image with signs bounded and the name of the recognized class (if any)
*/
cv::Mat RoadSignDetector::getClassificationResults() {
	if(!m_preprocessed)
		preprocess();

	// Start classification of detected road signs
	std::vector<int> labels = classifyRoadSigns();

	// Draw in destination image
	for (int i = 0; i < labels.size(); i++) {
		if (labels[i] != -1) {
			cv::rectangle(m_dst_image, m_found_rectangles[i].tl(), m_found_rectangles[i].br(), cv::Scalar(0, 255, 0), 2);
			cv::putText(m_dst_image, m_svm_classifier.getClassName(labels[i]), cv::Point(m_found_rectangles[i].x + 5, m_found_rectangles[i].y + m_found_rectangles[i].height - 5), cv::FONT_HERSHEY_TRIPLEX, 0.5, cv::Scalar(0, 255, 0));
			if(m_debug_mode_flag)
				printf("\nClass found: %d", labels[i]);
		}//if recognized
	}//for
	return m_dst_image;
}//getClassificationResults



/*
Function that, given the images of the found signs, returns the labels assigned to the images
return  the vector that contains the assigned labels (-1 = not recognized image)
*/
std::vector<int> RoadSignDetector::classifyRoadSigns() {
	if (!m_preprocessed)
		preprocess();

	std::vector<int> classesFound(m_found_signs.size());
	for (int i = 0; i < m_found_signs.size(); i++) {
		int classFound = m_svm_classifier.svmClassPrediction(m_found_signs[i]);
		classesFound[i] = classFound;
	}//for
	return classesFound;
}//classifySigns



/*
Function that discards the rectangles that have an overlapping area that is greater that a threshold, all taken from the same vector of rectangles
@param  rectVector  the vector of rectangles
@param  max_overlap  the maximum overlap area allowed
return  a vector that contains only the rectangles of the input vector that does not overlap (or whose overlap area is less or equal that the threshold)
*/
std::vector<cv::Rect> RoadSignDetector::getNoOverlappingRectangles(std::vector<cv::Rect> rectVector, int max_overlap) {
	// Firstly, it is needed to order the rectangles by increasing area
	// Bigger rectangles are probably the external part of the road sign and must be preserved
	// Smaller rectangles that are part of a bigger rectangle are not important, they contain features that are already in the bigger one -> no loss of information
	std::vector<cv::Rect> result(rectVector);
	std::sort(result.begin(), result.end(), sort_order());

	// Now I can delete overlapping rectangles from the vector
	bool overlap = false;
	for (int i = 0; i < result.size(); i++) {
		for (int j = i + 1; j < result.size(); j++) {
			cv::Rect overlappingRect = result[i] & result[j];
			if (overlappingRect.area() > max_overlap) {
				// Overlapping
				result.erase(result.begin() + j);
				j--;
			}//if overlap
		}//for j
	}//for i
	return result;
}//getNoOverlappingRectangles



/*
Function that finds the circles in the given image
@param  image  the image where to find circles
return  true if at least one circle has been found, false otherwise
*/
bool RoadSignDetector::findCircles(cv::Mat image) {

	// Auto-tuning of Canny parameters
	cv::Mat hough_circles_image;
	int lowThreshold;
	int highThreshold;
	autoTunedCanny(image, &lowThreshold, &highThreshold);

	// Vector that will contain found circles in the form (center_x, center_y, radius)
	std::vector<cv::Vec3f> circles;

	// Detect circles in the image
	// Note that HoughCircles has a built-in Canny edge detector, so there is no need to perform an edge detection before
	cv::cvtColor(image, hough_circles_image, CV_BGR2GRAY);
	int minRadius = std::min(hough_circles_image.rows, hough_circles_image.cols) / 2 - 20;
	cv::HoughCircles(hough_circles_image, circles, cv::HoughModes::HOUGH_GRADIENT, 2, std::max(hough_circles_image.rows, hough_circles_image.cols), highThreshold, 100, minRadius, minRadius + 50);

	if(m_debug_mode_flag)
		std::cout << "\n\nCircles: " << circles.size() << std::endl;

	return circles.size() > 0;
}//findCircles



/*
Function that automatically chooses the best Canny edge detector parameters
@param  image  the image where Canny will be applied
@param  lowThreshold  used as return value, the low threshold calculated
@param  highThreshold  used as return value, the high threshold calculated
*/
void RoadSignDetector::autoTunedCanny(cv::Mat image, int* lowThreshold, int* highThreshold) {
	// Auto-tuning Canny of the considered image
	double sigma = 0.33;
	cv::Mat gray;
	cv::cvtColor(image, gray, CV_BGR2GRAY);
	cv::Scalar meanValue = cv::mean(gray);
	*lowThreshold = (int)(cv::max(0.0, (1.0 - sigma) * meanValue.val[0]));
	*highThreshold = (int)(cv::min(255.0, (1.0 + sigma) * meanValue.val[0]));

	if (m_debug_mode_flag)
		printf("\n\nThresholds: %d %d %f\n\n", *lowThreshold, *highThreshold, meanValue.val[0]);

}//autoTunedCanny


