#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <ctime>

#define SAVE_PROGRESS 1
#define SHOW_PROGRESS 0

using namespace std;
using namespace cv;

Mat src, src_bin, src_tmp, src_con, src_aft;
vector< vector<Point> > contours;

const int bin_threshold = 91;

const int EROSION_TYPE = 2;// 0: Rect, 1: Cross, 2: Ellipse
const int EROSION_SIZE = 1;
const int EROSION_TIMES = 2;
const int DILATION_TYPE = 2;// 0: Rect, 1: Cross, 2: Ellipse
const int DILATION_SIZE = 1;
const int DILATION_TIMES = 2;

const int CONTOUR_MIN = 8;
const int CONTOUR_MAX = 50;

int main() {

	//refresh random seed
	unsigned seed;
    seed = (unsigned)time(NULL);
	srand(seed);

	//get input file name
	char* input = new char[256];
	string fileName;
	cout << "Please Enter the Image Name: ";
	cin.getline(input, 256, '\n');
	fileName = input;

	//creat a window
	namedWindow("img", CV_WINDOW_AUTOSIZE);

	//load image
	src = imread(fileName, CV_LOAD_IMAGE_GRAYSCALE);
	if(SHOW_PROGRESS)
	{
		imshow("img", src);
		waitKey(0);
	}

	//binarization
	src.copyTo(src_bin);
	threshold(src_bin, src_bin, bin_threshold, 255, CV_THRESH_BINARY);
	if(SHOW_PROGRESS)
	{
		imshow("img", src_bin);
		waitKey(0);
	}
	if(SAVE_PROGRESS)
	{
		imwrite("a01.bmp", src_bin);
	}

	//erosion and dilation element
	Mat erosion_element = getStructuringElement(EROSION_TYPE, Size(2*EROSION_SIZE + 1, 2*EROSION_SIZE+1), Point(EROSION_SIZE, EROSION_SIZE));
	Mat dilation_element = getStructuringElement(DILATION_TYPE, Size(2*DILATION_SIZE + 1, 2*DILATION_SIZE+1), Point(DILATION_SIZE, DILATION_SIZE));

	src_bin.copyTo(src_tmp);

	//closing
	for(size_t i = 0; i < DILATION_TIMES; ++i)
		dilate(src_tmp, src_tmp, dilation_element);
	for(size_t i = 0; i < EROSION_TIMES; ++i)
		erode(src_tmp, src_tmp, erosion_element);
	if(SHOW_PROGRESS)
	{
		imshow("img", src_tmp);
		waitKey(0);
	}
	if(SAVE_PROGRESS)
	{
		imwrite("a02.bmp", src_tmp);
	}

	//opening
	for(size_t i = 0; i < EROSION_TIMES; ++i)
		erode(src_tmp, src_tmp, erosion_element);
	for(size_t i = 0; i < DILATION_TIMES; ++i)
		dilate(src_tmp, src_tmp, dilation_element);
	dilate(src_tmp, src_tmp, dilation_element);
	if(SHOW_PROGRESS)
	{
		imshow("img", src_tmp);
		waitKey(0);
	}
	if(SAVE_PROGRESS)
	{
		imwrite("a03.bmp", src_tmp);
	}

	//find contours
	findContours(src_tmp, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	//draw contours and calculate dice points
	int dicePoints = 0;
	src_con = Mat::zeros(src.size(), CV_8UC3);
	for(size_t i=0; i<contours.size(); ++i) {
		int b = rand()%256;
		int g = rand()%256;
		int r = rand()%256;
		if((int)contours.at(i).size() <= CONTOUR_MAX && (int)contours.at(i).size() >= CONTOUR_MIN) {
			drawContours(src_con, contours, i, cv::Scalar(b, g, r), 1);
			dicePoints++;
		}
	}
	if(SHOW_PROGRESS)
	{
		imshow("img", src_con);
		waitKey(0);
	}
	if(SAVE_PROGRESS)
	{
		imwrite("a04.bmp", src_con);
	}

	src.copyTo(src_aft);
	
	//draw result
	stringstream ss;
	ss << "Dice Points: " << dicePoints;
	putText(src_aft, ss.str(), Point(0, 20), FONT_HERSHEY_COMPLEX, 0.75, Scalar(0, 0, 0));
	imshow("img", src_aft);
	waitKey(0);

	imwrite("result.bmp", src_aft);

	return 0;
}