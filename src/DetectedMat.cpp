/*
Computer Vision Final Project - Road Sign Recognition
Valente Alex - 1173742
UniPD - A.A. 2017-2018
*/
#include "DetectedMat.h"

/*
Constructor of DetectedMat
@param  image  the image detected
@param  detectedWithRedMask  indicates whether the image has been detected with the red mask or not
@param  detectedWithBlueMask  indicates whether the image has been detected with the blue mask or not
@param  hasCircles  indicates whether the image contains a circular sign or not
@param  hasTriangles  indicates whether the image contains a triangle-shaped sign or not
*/
DetectedMat::DetectedMat(cv::Mat image, bool detectedWithRedMask, bool detectedWithBlueMask, bool hasCircles, bool hasTriangles) : cv::Mat(image) {
	m_detectedWithRedMask = detectedWithRedMask;
	m_detectedWithBlueMask = detectedWithBlueMask;
	m_hasCircles = hasCircles;
	m_hasTriangles = hasTriangles;
}//DetectedMat



/*Getter function for isDetectedWithRedMask
return  indicates whether the image has been detected with the red mask or not
*/
bool DetectedMat::isDetectedWithRedMask() {
	return m_detectedWithRedMask;
}//isDetectedWithRedMask



/*Getter function for detectedWithBlueMask
return  indicates whether the image has been detected with the blue mask or not
*/
bool DetectedMat::isDetectedWithBlueMask() {
	return m_detectedWithBlueMask;
}//isDetectedWithBlueMask



/*Getter function for hasCircles
return  indicates whether the image contains a circular sign or not
*/
bool DetectedMat::hasCircles() {
	return m_hasCircles;
}//hasCircles



/*Getter function for hasTriangles
return  indicates whether the image contains a triangle-shaped sign or not
*/
bool DetectedMat::hasTriangles() {
	return m_hasTriangles;
}//hasTriangles
