#ifndef DETECTED__MAT__H
#define DETECTED__MAT__H

#include <opencv2/highgui.hpp>

class DetectedMat : public cv::Mat {
public:
	DetectedMat(cv::Mat image, bool detectedWithRedMask, bool detectedWithBlueMask, bool hasCircles = false, bool hasTriangles = false);

	bool isDetectedWithRedMask();

	bool isDetectedWithBlueMask();

	bool hasCircles();

	bool hasTriangles();

private:
	bool m_detectedWithRedMask;
	bool m_detectedWithBlueMask;
	bool m_hasCircles;
	bool m_hasTriangles;
};

#endif
