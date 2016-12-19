#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <ctime>

#include <opencv2/opencv.hpp>

#define SHOW_PROGRESS 1
#define SAVE_PROGRESS 1

using namespace std;
using namespace cv;

const int BIN_THRESHOLD = 120;

const int EROSION_TYPE = 2;// 0: Rect, 1: Cross, 2: Ellipse
const int EROSION_SIZE = 1;
const int EROSION_TIMES = 2;
const int DILATION_TYPE = 2;// 0: Rect, 1: Cross, 2: Ellipse
const int DILATION_SIZE = 1;
const int DILATION_TIMES = 2;

const int CONTOUR_SIZE_MIN = 8;
const int CONTOUR_SIZE_MAX = 50;

string getString(string message);

int main()
{
	Mat src, src_bin, src_tmp, src_con, src_result;
	vector< vector<Point> > contours;

	//refresh random seed
	srand((unsigned)time(NULL));

	//get input file name
	string fileIn = getString("Please Enter the Dice File Name: ");

	//read pic
	cout << endl << "Loading " << fileIn << "..." << endl;
	src = imread(fileIn, CV_LOAD_IMAGE_GRAYSCALE);
	if(src.empty())
	{
		cout << "Can't Find File!" << endl;
		system("pause");
		return 0;
	}
	cout << "Load Done." << endl << endl;

	//show image
	namedWindow("img", CV_WINDOW_AUTOSIZE);
	if(SHOW_PROGRESS)
	{
		imshow("img", src);
		waitKey();
	}



	src.copyTo(src_bin);

	//binarization
	threshold(src_bin, src_bin, BIN_THRESHOLD, 255, CV_THRESH_BINARY);
	if(SHOW_PROGRESS)
	{
		imshow("img", src_bin);
		waitKey();
	}
	if(SAVE_PROGRESS)
	{
		string tmp = fileIn;
		tmp.insert(fileIn.find('.'), "_01");
		imwrite(tmp, src_bin);
	}

	//create erosion and dilation element
	Mat erosion_element = getStructuringElement(EROSION_TYPE, Size(2*EROSION_SIZE + 1, 2*EROSION_SIZE+1), Point(EROSION_SIZE, EROSION_SIZE));
	Mat dilation_element = getStructuringElement(DILATION_TYPE, Size(2*DILATION_SIZE + 1, 2*DILATION_SIZE+1), Point(DILATION_SIZE, DILATION_SIZE));

	src_bin.copyTo(src_tmp);

	//do closing
	for(size_t i = 0; i < DILATION_TIMES; ++i)
		dilate(src_tmp, src_tmp, dilation_element);
	for(size_t i = 0; i < EROSION_TIMES; ++i)
		erode(src_tmp, src_tmp, erosion_element);
	if(SHOW_PROGRESS)
	{
		imshow("img", src_tmp);
		waitKey();
	}
	if(SAVE_PROGRESS)
	{
		string tmp = fileIn;
		tmp.insert(fileIn.find('.'), "_02");
		imwrite(tmp, src_tmp);
	}

	//do opening
	for(size_t i = 0; i < EROSION_TIMES; ++i)
		erode(src_tmp, src_tmp, erosion_element);
	for(size_t i = 0; i < DILATION_TIMES; ++i)
		dilate(src_tmp, src_tmp, dilation_element);
	dilate(src_tmp, src_tmp, dilation_element);
	if(SHOW_PROGRESS)
	{
		imshow("img", src_tmp);
		waitKey();
	}
	if(SAVE_PROGRESS)
	{
		string tmp = fileIn;
		tmp.insert(fileIn.find('.'), "_03");
		imwrite(tmp, src_tmp);
	}

	//find contours
	findContours(src_tmp, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);

	//draw contours and calculate total dice points
	int totalDicePoints = 0;
	src_con = Mat::zeros(src.size(), CV_8UC3);
	for(size_t i=0; i<contours.size(); ++i)
	{
		int b = rand()%256;
		int g = rand()%256;
		int r = rand()%256;
		if((int)contours.at(i).size() <= CONTOUR_SIZE_MAX && (int)contours.at(i).size() >= CONTOUR_SIZE_MIN)
		{
			drawContours(src_con, contours, i, cv::Scalar(b, g, r));
			totalDicePoints++;
		}
	}
	if(SHOW_PROGRESS)
	{
		imshow("img", src_con);
		waitKey();
	}
	if(SAVE_PROGRESS)
	{
		string tmp = fileIn;
		tmp.insert(fileIn.find('.'), "_04");
		imwrite(tmp, src_con);
	}

	src.copyTo(src_result);

	//draw result and show
	stringstream ss;
	ss << "Total Dice Points: " << totalDicePoints;
	putText(src_result, ss.str(), Point(0, 20), FONT_HERSHEY_COMPLEX, 0.75, Scalar(0, 0, 0));
	imshow("img", src_result);
	waitKey();

	//save pic
	string fileOut = fileIn;
	fileOut.insert(fileIn.find('.'), "_points");
	cout << "Saving " << fileOut << "..." << endl;
	imwrite(fileOut, src_result);
	cout << "Save Done." << endl;

	return 0;
}

string getString(string message)
{
	cout << message;

	char* input = new char[256];
	cin.getline(input, 256, '\n');

	string fileName = input;

	return fileName;
}
