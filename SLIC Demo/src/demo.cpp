// Demo.cpp
// Demonstrates SLIC
// Author: Chandler Calkins

#include <string>
#include <direct.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ximgproc/slic.hpp>
using namespace cv;

// main - Generates superpixels for an images using SLIC and displays those superpixels on the image.
//
// Preconditions:
//
// There is a valid .jpg or .png file in the project folder and the `imread()` call that creates the `input_image` object reads from that file.
//
// Postconditions:
//
// A file called output.jpg should be in the project folder.
int main(int argc, char* argv[])
{
	// Move out of build/Debug into root of project folder
	// Use this for VSCode, comment out for Visual Studio / actual submission
	_chdir("../../");

	// Reads the input image
	const Mat input_image = imread("cosmo.png");

	// Creates window to display output to
	const String window_name = "Feature Matches";
	namedWindow(window_name);

	Ptr<ximgproc::SuperpixelSLIC> slic = ximgproc::createSuperpixelSLIC(input_image);

	// Creates the output image of superpixels
	Mat output;
	// TODO: create superpixel output

	// Displays output to a window
	const unsigned char SCALE = 8;
	resizeWindow(window_name, output.cols / SCALE, output.rows / SCALE);
	imshow(window_name, input_image);
	waitKey(0);
	
	// Write output to an image file
	imwrite("output.jpg", output);

	return 0;
}
