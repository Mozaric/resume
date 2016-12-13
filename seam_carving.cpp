#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include "cvaux.h"
#include <iostream>
#include <cmath>
#include <fstream>

#define SEAMS_NUM 100

using namespace cv;
using namespace std;

void RGB2Gray(IplImage* in, IplImage* out);
void SobelOperation(IplImage* in, IplImage* out);
IplImage* SeamCarving(IplImage* in_gray, IplImage* in_color, int seams=30);
void TestUchar(IplImage* in);

void TwoDAlloc_bool(bool** ptr, int row, int column);
void TwoDAlloc_int(int** ptr, int row, int column);

/*=============================
IplImage* cvLoadImage(char* file_name, int flag);
flags:
	CV_LOAD_IMAGE_UNCHANGED	   -1  原圖影像
	CV_LOAD_IMAGE_GRAYSCALE		0  灰階
	CV_LOAD_IMAGE_COLOR			1  彩色
----	---------------------------
image_in = cvLoadImage("test.bmp", CV_LOAD_IMAGE_UNCHANGED);
===============================
IplImage* cvCreateImage(CvSize size, int depth, int channels);
size:
	cvSize(width, height)
depth:
	IPL_DEPTH_8U - unsigned 8-bit integer. Equivalent to CV_8U in matrix types.
	IPL_DEPTH_8S - signed 8-bit integer. Equivalent to CV_8S in matrix types.
	IPL_DEPTH_16U - unsigned 16-bit integer. Equivalent to CV_16U in matrix types.
	IPL_DEPTH_16S - signed 8-bit integer. Equivalent to CV_16S in matrix types.
	IPL_DEPTH_32S - signed 32-bit integer. Equivalent to CV_32S in matrix types.
	IPL_DEPTH_32F - single-precision floating-point number. Equivalent to CV_32F in matrix types.
	IPL_DEPTH_64F - double-precision floating-point number. Equivalent to CV_64F in matrix types.
channels:
	gray level image -> channels = 1
	color level image -> channels = 3
----	---------------------------
IplImage *image_gray = cvCreateImage(cvSize(800, 600), IPL_DEPTH_8U, 1);
=============================*/

int main(int argc, char** argv)
{
	//Get Image File Name
	cout << "Please Enter the Image File Name: ";
	char* chInput = new char[256];
	cin.getline(chInput, 256, '\n');

	//File Name Processing
	int dot_pos;
	string strInput = chInput;
	string str_gray, str_sobel, str_seamc;
	dot_pos = strInput.find('.', 0);
	str_gray = strInput;
	str_sobel = strInput;
	str_seamc = strInput;
	str_gray.insert(dot_pos, "_gray");
	str_sobel.insert(dot_pos, "_sobel");
	str_seamc.insert(dot_pos, "_seamc");

	//Load Image File
	IplImage* image_in;
	image_in = cvLoadImage(chInput, CV_LOAD_IMAGE_COLOR);

	//Seam Carving Loop
	//有一張原始彩色圖(m*n)
	//使用其彩色圖(m*n)計算灰階圖(m*n)
	//使用灰階圖(m*n)計算Sobel Edge Detection(m*n)
	//使用sobel(m*n), 彩色圖(m*n) 進行seam carving 可得一彩色圖((m-1)*n)
	//
	//再使用此彩色圖計算其灰階圖
	//......
	
	IplImage* image_color = cvCloneImage(image_in);
	IplImage* image_out;

	for(int i=0; i<SEAMS_NUM; i++)
	{
		cout << "Round: " << i << endl;
		IplImage* image_gray = cvCreateImage(cvSize(image_color->width, image_color->height), IPL_DEPTH_8U, 1);
		IplImage* image_sobel = cvCreateImage(cvSize(image_color->width, image_color->height), IPL_DEPTH_8U, 1);
		
		RGB2Gray(image_color, image_gray);
		SobelOperation(image_gray, image_sobel);
		IplImage* image_seamc = SeamCarving(image_sobel, image_color, 1);

		image_color = cvCloneImage(image_seamc);

		//Process the file name
		string str_out = strInput;
		stringstream ss1;
		if(i<10)	ss1 << "_seamc0" << i;
		else		ss1 << "_seamc" << i;
		str_out.insert(dot_pos, ss1.str());
		//Save Process files
		cvSaveImage(str_out.c_str(), image_seamc);

		//Save Result Image
		if(i==SEAMS_NUM-1)
			image_out = cvCloneImage(image_seamc);

		cvReleaseImage(&image_gray);
		cvReleaseImage(&image_sobel);
		cvReleaseImage(&image_seamc);
	}

	//Show the Images
	cvShowImage("Original Image", image_in);
	cvShowImage("Seam Carving", image_out);
	cvWaitKey(0);
	
	//Destroy Windows
	cvDestroyWindow("Original Image");
	cvDestroyWindow("Seam Carving");
	
	//Release Images
	cvReleaseImage(&image_in);
	cvReleaseImage(&image_out);

    return 0;
}

void RGB2Gray(IplImage* in, IplImage* out)
{
	uchar blue(0), green(0), red(0);

	//Gray Level = (Blue + Green + Red)/3
	for(int i=0; i<in->height; i++)
	{
        	for(int j=0; j<in->width*3; j=j+3)
		{
            		blue = in->imageData[i*in->widthStep+j];
            		green = in->imageData[i*in->widthStep+j+1];
            		red = in->imageData[i*in->widthStep+j+2];
			out->imageData[i*out->widthStep+j/3] = (blue+green+red)/3;
		}
	}
}

void SobelOperation(IplImage* in, IplImage* out)
{
	int sobel_filter_x[3][3]= {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
	int sobel_filter_y[3][3]= {{1, 2, 1}, {0, 0, 0}, {-1, -2, -1}};
	double Gx(0), Gy(0);// Gx -> Gradient of x direction, Gy -> Gradient of y direction
	int height = in->height;
	int width = in->width;
	int widthStep = in->widthStep;

	//Dynamic allocation
	int** Gradient;
	Gradient = new int*[height];
	for(int i=0; i<height; i++)
		*(Gradient+i) = new int[width];

	//Compute the Gradient Image
	for(int i=0; i<height; i++)
	{
		for(int j=0; j<width; j++)
		{
			if ( i==0 || j==0 || i==height-1 || j==width-1 )//Do not use Sobel Operator on Edge Points
				Gradient[i][j] = 0;
			else
			{
				Gx = 0;
				Gy = 0;
				for(int m=i-1; m<i+2; m++)
				{
					for(int n=j-1; n<j+2; n++)
					{
						Gx += (uchar)(in->imageData[m*widthStep+n])*sobel_filter_x[m-i+1][n-j+1];//sobel operation
						Gy += (uchar)(in->imageData[m*widthStep+n])*sobel_filter_y[m-i+1][n-j+1];
					}
				}
				Gradient[i][j] = (int)sqrt(Gx*Gx+Gy*Gy);//G = sqrt(Gx^2 + Gy^2)
			}
		}
	}

	//Change the Scale: Min ~ Max -> 0 ~ 255
	//Find the Maximum and Minimum Gradient value
	int Gradient_Max(0), Gradient_Min(0);
	Gradient_Max = Gradient[1][1];
	Gradient_Min = Gradient[1][1];
	for(int i=1; i<height-1; i++)
	{
		for(int j=1; j<width-1; j++)
		{
			if(Gradient[i][j] > Gradient_Max) Gradient_Max = Gradient[i][j];
			else if(Gradient[i][j] < Gradient_Min) Gradient_Min = Gradient[i][j];
		}
	}

	//Change the scale
	//0------------255			scaleA
	//Minimum------Maximum		scaleB
	//scaleB = 255*(scaleA-Min)/(Max-Min)
	double diff = (double)Gradient_Max-Gradient_Min;
	for(int i=0; i<height; i++)
	{
		for(int j=0; j<width; j++)
		{
			Gradient[i][j] = (int)((Gradient[i][j]-Gradient_Min)*255/diff);
			///////////////////////////////////////////////////
			//if(Gradient[i][j]<10) Gradient[i][j] = 0;
			out->imageData[i*widthStep+j] = (uchar)Gradient[i][j];
		}
	}

	//Release pointer
	for(int i=0; i<height; i++)
		delete [] *(Gradient+i);
	delete [] Gradient;
}

IplImage* SeamCarving(IplImage* in_gray, IplImage* in_color, int seams)
{
	int ori_width = in_color->width;
	int ori_height = in_color->height;
	int ori_widthStep = in_color->widthStep;

	//Create and Initialize the Seam Map
	bool** seam_map;
	seam_map = new bool*[ori_height];
	for(int i=0; i<ori_height; i++)
		*(seam_map+i) = new bool[ori_width];
	for(int i=0; i<ori_height; i++)
		for(int j=0; j<ori_width; j++)
			seam_map[i][j] = false;

	//Create the array to copy image data
	int** M;
	M = new int*[ori_height];
	for(int i=0; i<ori_height; i++)
		*(M+i) = new int[ori_width];
	//Copy the image data
	for(int i=0; i<ori_height; i++)
		for(int j=0; j<ori_width; j++)
			M[i][j] = abs(in_gray->imageData[i*in_gray->widthStep+j]);

	//Use Gradient Image to find the M Image
	//計算累積梯度值
	for(int i=2; i<ori_height-1; i++)
	{
		for(int j=1; j<ori_width-1; j++)
		{
			if(j==1)
				M[i][j] += min(M[i-1][j], M[i-1][j+1]);
			else if(j==ori_width-2)
				M[i][j] += min(M[i-1][j-1], M[i-1][j]);
			else
				M[i][j] += min(M[i-1][j-1], min(M[i-1][j], M[i-1][j+1]));
		}
	}

	//Copy the bottom value
	int* bot_value	= new int[ori_width-2];
	int* bot_no		= new int[ori_width-2];
	for(int i=1; i<ori_width-1; i++)
	{
		bot_no[i-1]		= i;
		bot_value[i-1]	= M[ori_height-2][i];
	}

	//Bubble Sort
	int temp(0);
	for(int i=0; i<ori_width-2; i++)
	{
		for(int j=0; j<ori_width-3-i; j++)
		{
			if(bot_value[j] > bot_value[j+1])
			{
				temp			= bot_value[j];
				bot_value[j]	= bot_value[j+1];
				bot_value[j+1]	= temp;
				temp			= bot_no[j];
				bot_no[j]		= bot_no[j+1];
				bot_no[j+1]		= temp;
			}
		}
	}

	//Fine the Seams
	for(int i=0; i<seams; i++)
	{
		int curr_pos_1 = 0;
		int curr_pos_2 = 0;
		int minimum;
		int n1(0), n2(0), n3(0);

		curr_pos_1 = ori_height-2;
		curr_pos_2 = bot_no[i];

		seam_map[curr_pos_1][curr_pos_2] = true;

		for(int m=ori_height-3; m>0; m--)
		{
			if(seam_map[m][curr_pos_2-1])
				n1 = 100000;
			else
				n1 = M[m][curr_pos_2-1];

			if(seam_map[m][curr_pos_2])
				n2 = 100000;
			else
				n2 = M[m][curr_pos_2];

			if(seam_map[m][curr_pos_2+1])
				n3 = 100000;
			else
				n3 = M[m][curr_pos_2+1];

			if(curr_pos_2==ori_width-2)
				n3 = 100000;
			if(curr_pos_2==1)
				n1 = 100000;

			minimum = min(n1, min(n2, n3));
			if(minimum==n1)
				curr_pos_2 = curr_pos_2-1;
			else if(minimum==n2)
				curr_pos_2 = curr_pos_2;
			else
				curr_pos_2 = curr_pos_2+1;

			curr_pos_1 = m;
			seam_map[curr_pos_1][curr_pos_2] = true;
		}
	}

	IplImage* out_2 = cvCreateImage(cvSize(ori_width-1, ori_height), IPL_DEPTH_8U, 3);

	int aft_width = out_2->width;
	int aft_widthStep = out_2->widthStep;

	int j2(0);
	for(int i=0; i<ori_height; i++)
	{
		j2 = 0;
		for(int j=0; j<ori_width*3; j+=3)
		{
			int index1 = i*ori_widthStep+j;
			int index2 = i*aft_widthStep+j2;
			if(seam_map[i][j/3])
				;
			else
			{
				out_2->imageData[index2] = in_color->imageData[index1];
				out_2->imageData[index2+1] = in_color->imageData[index1+1];
				out_2->imageData[index2+2] = in_color->imageData[index1+2];
				j2 += 3;
			}
		}
	}

	return out_2;
}

void TestUchar(IplImage* in)
{
	IplImage* test_i = cvCreateImage(cvSize(in->width, in->height), IPL_DEPTH_8U, 1);
	IplImage* test_u = cvCreateImage(cvSize(in->width, in->height), IPL_DEPTH_8U, 1);
	int** test_int;
	uchar** test_uch;

	test_int = new int*[in->height];
	test_uch = new uchar*[in->height];
	for(int i=0; i<in->height; i++)
	{
		*(test_int+i) = new int[in->width];
		*(test_uch+i) = new uchar[in->width];
	}

	for(int i=0; i<in->height; i++)
	{
		for(int j=0; j<in->width; j++)
		{
			test_int[i][j] = in->imageData[i*in->widthStep+j];
			test_uch[i][j] = in->imageData[i*in->widthStep+j];
			test_int[i][j] = test_int[i][j]/10;
			test_uch[i][j] = test_uch[i][j]/10;
		}
	}

	for(int i=0; i<in->height; i++)
	{
		for(int j=0; j<in->width; j++)
		{
			test_i->imageData[i*in->widthStep+j] = test_int[i][j];
			test_u->imageData[i*in->widthStep+j] = test_uch[i][j];
		}
	}

	cvShowImage("Origin", in);
	cvShowImage("Test Int", test_i);
	cvShowImage("Test Uchar", test_u);
	cvWaitKey(0);
	//Destroy Windows
	cvDestroyWindow("Origin");
	cvDestroyWindow("Test Int");
	cvDestroyWindow("Test Uchar");
	//Release Images
	cvReleaseImage(&test_i);
	cvReleaseImage(&test_u);

	//結果顯示，用Uchar Copy資料再進行處理會比較好!!
}
