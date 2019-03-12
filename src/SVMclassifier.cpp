/*
Computer Vision Final Project - Road Sign Recognition
Valente Alex - 1173742
UniPD - A.A. 2017-2018
*/
#include "SVMclassifier.h"

/*
Constructor of SVMclassifier class
*/
SVMclassifier::SVMclassifier() {
	m_svm = std::vector<cv::Ptr<cv::ml::SVM>>(NUM_CLASSES);
	for (int i = 0; i < m_svm.size(); i++) {
		m_svm[i] = cv::ml::SVM::create();
	}//for
	init();
}//SVMclassifier



/*
Constructor of SVMclassifier class
@param  classifiers_path  the path of the files of the SVM trained classifiers
*/
SVMclassifier::SVMclassifier(std::vector<cv::String> svm_classifier_files) {
	if (svm_classifier_files.empty() || svm_classifier_files.size() < NUM_CLASSES) {
		// Not possible to load train data file
		// Creation of new binary classifiers to be trained with svmClassTraining
		m_svm = std::vector<cv::Ptr<cv::ml::SVM>>(NUM_CLASSES);
		for (int i = 0; i < m_svm.size(); i++) {
			m_svm[i] = cv::ml::SVM::create();
		}//for
	}
	else {
		// Initialization of pre-trained Support Vector Machine classifiers
		m_svm = std::vector<cv::Ptr<cv::ml::SVM>>(NUM_CLASSES);
		for (int i = 0; i < m_svm.size(); i++) {
			m_svm[i] = cv::ml::SVM::load(svm_classifier_files[i]);
		}//for
	}//if else
	init();
}//SVMclassifier



/*
Sets the debug mode on or off, to view or not debug images and messages
@param  debug  the status of the debug mode: true = on, false = off
*/
void SVMclassifier::setDebugMode(bool debug) {
	m_debug_mode_flag = debug;
}//setDebugMode



/*
Function used to initialize the classifier, setting the names of the classes that will be detected
*/
void SVMclassifier::init() {
	m_label_names = std::vector<std::string>(NUM_CLASSES);
	m_label_names[0] = "Speed limit 10";
	m_label_names[1] = "Speed limit 20";
	m_label_names[2] = "Speed limit 30";
	m_label_names[3] = "Speed limit 40";
	m_label_names[4] = "Speed limit 50";
	m_label_names[5] = "Speed limit 60";
	m_label_names[6] = "Speed limit 70";
	m_label_names[7] = "Speed limit 80";
	m_label_names[8] = "Speed limit 90";
	m_label_names[9] = "Speed limit 100";
	m_label_names[10] = "Speed limit 120";
	m_label_names[11] = "No parking";
	m_label_names[12] = "Snow chains";
	m_label_names[13] = "Warning";
}//init



/*
Function used for the training of the NUM_CLASSES SVM classifiers
@param  classpath  the path of the folder containing the images for training, divided by class
return  the trained SVM classifier
*/
std::vector<cv::Ptr<cv::ml::SVM>> SVMclassifier::svmClassTraining(std::string classpath) {
	// Import files names divided by class
	int num_files = 0;
	std::vector<std::vector<cv::String>> imagesName(NUM_CLASSES);
	for (int num_class = 0; num_class < NUM_CLASSES; num_class++) {
		std::string path = classpath + "/" + std::to_string(num_class) + "/";
		cv::glob(path, imagesName[num_class]);
		num_files += imagesName[num_class].size();
	}//for

	printf("Num files: %d\n", num_files);

	// Training of each classifier
	for (int num_class_classifier = 0; num_class_classifier < NUM_CLASSES; num_class_classifier++) {

		// Initialization of variables needed to store training data and labels
		cv::Mat training_mat(num_files, HOG_DESCRIPTOR_SIZE, CV_32FC1);
		cv::Mat labels_mat(num_files, 1, CV_32SC1, cv::Scalar(0));
		int processedImages = 0;

		// Initialization of HOG (Histogram of Gradients) descriptor
		cv::HOGDescriptor desc(cv::Size(64, 64), cv::Size(4, 4), cv::Size(4, 4), cv::Size(4, 4), 9);

		for (int num_class = 0; num_class < NUM_CLASSES; num_class++) {

			// Initialization of labels of class
			// Classify the current class as 1 = POSITIVE, all the other classes as -1 = NEGATIVE
			if (num_class == num_class_classifier) {
				labels_mat(cv::Range(processedImages, processedImages + imagesName[num_class].size()), cv::Range::all()) = 1;
			}
			else {
				labels_mat(cv::Range(processedImages, processedImages + imagesName[num_class].size()), cv::Range::all()) = -1;
			}//if else

			// Processing of all images read
			for (int file_num = 0; file_num < imagesName[num_class].size(); file_num++) {
				cv::Mat img_mat = cv::imread(imagesName[num_class][file_num], 0); // 0 for greyscale
				cv::resize(img_mat, img_mat, cv::Size(64, 64));

				// HOG (Histogram of Gradient) descriptor calculation
				std::vector<float> descriptorsValues;
				std::vector<cv::Point> locations;
				desc.compute(img_mat, descriptorsValues, cv::Size(4, 4), cv::Size(0, 0), locations);

				// HOG gives a vector as output, no need for vectorization
				// Vector added to the training_mat
				int jj = 0; // Current column in training_mat
				for (int i = 0; i < descriptorsValues.size(); i++) {
					training_mat.at<float>(processedImages, jj++) = descriptorsValues[i];
				}//for i
				processedImages++;
			}//for file_num

		}//for num_class

		// SVM initialization and training for each classifier
		printf("\nStarting training for class %d\n", num_class_classifier);
		m_svm[num_class_classifier]->setType(cv::ml::SVM::C_SVC); // Support Vector Machine for Classification
		m_svm[num_class_classifier]->setKernel(cv::ml::SVM::LINEAR); // Linear Kernel
		m_svm[num_class_classifier]->setTermCriteria(cv::TermCriteria(cv::TermCriteria::MAX_ITER, 500, 1e-6));
		m_svm[num_class_classifier]->train(training_mat, cv::ml::ROW_SAMPLE, labels_mat);
		m_svm[num_class_classifier]->save("svm_trained_" + std::to_string(num_class_classifier) +".xml");
		printf("End of training\n");

	}//for num_class_classifier

	return m_svm;

}//svmClassTraining



/*
Function used to predict the class (if any) of the detected image
@param  image  the detected image to be classified
return  the index of the detected class (or -1 if no class detected)
*/
int SVMclassifier::svmClassPrediction(DetectedMat image) {

	// Check if all SVMs loaded are trained
	for (int i = 0; i < m_svm.size(); i++) {
		if (!(m_svm[i]->isTrained())) {
			std::cerr << "ERROR: trying to predict without having trained the classifier" << std::endl;
			return -1;
		}//if
	}//for	

	// Preparation of image for HOG descriptor calculation
	cv::Mat testImage = image.clone();
	cv::cvtColor(testImage, testImage, CV_BGR2GRAY);
	cv::resize(testImage, testImage, cv::Size(64, 64));

	// Initialization of HOG (Histogram of Gradients) descriptor
	cv::HOGDescriptor desc(cv::Size(64, 64), cv::Size(4, 4), cv::Size(4, 4), cv::Size(4, 4), 9);

	// Calculation of HOG descriptor
	std::vector<float> descriptorsValues;
	std::vector<cv::Point> locations;
	std::vector<cv::Rect> found;
	desc.compute(testImage, descriptorsValues, cv::Size(4, 4), cv::Size(0, 0), locations);

	// Vectorization of descriptor
	cv::Mat test_mat(1, HOG_DESCRIPTOR_SIZE, CV_32FC1);
	int kk = 0; // Current column in testImage
	for (int i = 0; i < descriptorsValues.size(); i++) {
		test_mat.at<float>(0, kk++) = descriptorsValues[i];
	}//for i

	// Start class prediction with SVM and additional information known
	float maxval = 0;
	float maxclass = -1;
	bool foundPositiveMatch = false;
	std::vector<float> results_confidence(NUM_CLASSES);
	for (int num_class_classifier = 0; num_class_classifier < NUM_CLASSES; num_class_classifier++) {
		cv::Mat results; // The distance of the classified point from the hyperplane
		m_svm[num_class_classifier]->predict(test_mat, results, cv::ml::StatModel::RAW_OUTPUT);
		float result = m_svm[num_class_classifier]->predict(test_mat);

		// Add a penalty to wrong classifications
		if (image.isDetectedWithBlueMask()) {
			if (image.hasCircles() && num_class_classifier != 12) {
				// The only blue circular sign is 12 (Snow chains)
				result = -1;
				results.at<float>(0, 0) = 100;
			}
			else if (!image.hasCircles()) {
				// Unrecognized, useless to try predicting
				result = -1;
				results.at<float>(0, 0) = 100;
			}//if else
		} 
		else if (image.isDetectedWithRedMask()) {
			if (num_class_classifier == 12) {
				// 12 (Snow chains) can not be detected with red mask
				result = -1;
				results.at<float>(0, 0) = 100;
			}
			else if (image.hasCircles() && num_class_classifier == 13) {
				// 13 (Warning) can not contain circles
				result = -1;
				results.at<float>(0, 0) = 100;
			}
			else if (!image.hasCircles() && num_class_classifier != 13) {
				// All road signs except 13 (Warning) are circular
				result = -1;
				results.at<float>(0, 0) = 100;
			}//if else
		}//if else

		// Consider the match with highest confidence
		if (result == 1 && cv::abs(results.at<float>(0,0)) > maxval) {
			// One of the classifier has classified the road sign with 1 = positive match
			// In case of multiple positive matching, we consider the one that has more distance from the hyperplane of the SVM
			maxval = cv::abs(results.at<float>(0, 0));
			maxclass = num_class_classifier;
			foundPositiveMatch = true;
		}
		else {
			// No-one of the classifiers has a positive match
			// I consider as positive match the negative match with lower distance from the hyperplane, considering a conservative threshold
			// If all points are too far from the separating hyperplane, I consider the road sign as not classificable and it is not recognized
			// It can be a false positive or a road sign whose class is different from the 14 classes considered
			if (result == -1 && !foundPositiveMatch && cv::abs(results.at<float>(0, 0)) <= RECOGNITION_THRESHOLD && -cv::abs(results.at<float>(0, 0)) < maxval) {
				maxval = cv::abs(results.at<float>(0, 0));
				maxclass = num_class_classifier;
			}//if
		}//if else
		
		if(m_debug_mode_flag)
			std::cout << "Result with class " << num_class_classifier << ": " << result << " " << results << std::endl;
	}//for

	return maxclass;
}//svmClassPrediction


/*
Returns the name of the class from its index
@param  the index of the class
return  the name of the class
*/
std::string SVMclassifier::getClassName(int num_class) {
	if (num_class >= 0 && num_class < NUM_CLASSES)
		return m_label_names[num_class];
	else
		return std::string();
}//getClassName
