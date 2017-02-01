#include <iostream>
#include <string>
#include <sstream>
using namespace std;

// OpenCV includes
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

// OpenCV command line parser functions
// Keys accecpted by command line parser
const char* keys =
{
"{help h usage ? | | print this message}"
"{@video | | Video file, if not defined try to use webcamera}"
};

Mat calculateLightPattern(Mat img)
{
	Mat pattern;
	// Basic and effective way to calculate the light pattern from one image
	blur(img, pattern, Size(img.cols/3,img.cols/3));
	return pattern;
}

Mat removeLight(Mat img, Mat pattern, int method)
{
	Mat aux;
	// if method is normalization
	if(method==1)
	{
		// Require change our image to 32 float for division
		Mat img32, pattern32;
		img.convertTo(img32, CV_32F);
		pattern.convertTo(pattern32, CV_32F);
		// Divide the image by the pattern
		aux= 1-(img32/pattern32);
		// Scale it to convert to 8bit format
		aux=aux*255;
		// Convert 8 bits format
		aux.convertTo(aux, CV_8U);
	}else{
		aux= pattern-img;
	}
	return aux;
}

Mat removeNoise(Mat source){
	Mat img_noise;
	medianBlur(source, img_noise, 3);
	return img_noise;
}

Mat convertToBlackAndWhite(Mat source){
	Mat im_gray;
	cvtColor(source,im_gray,CV_RGB2GRAY);
	return im_gray;
}

Scalar getRandomRgbColor(){
	RNG rng(12345);
	return Scalar(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
}

void connectedComponentsStats(Mat img)
{
	// Use connected components with stats
	Mat labels, stats, centroids;
	int num_objects= connectedComponentsWithStats(img, labels, stats,
	centroids);
	// Check the number of objects detected
	if(num_objects < 2 ){
		cout << "No objects detected" << endl;
		return;
	}else{
		cout << "Number of objects detected: " << num_objects - 1 << endl;
	}
	// Create output image coloring the objects and show area
	Mat output= Mat::zeros(img.rows,img.cols, CV_8UC3);
	RNG rng( 0xFFFFFFFF );
	for(int i=1; i<num_objects; i++){
		cout << "Object "<< i << " with pos: " << centroids.at<Point2d>(i)
		<< " with area " << stats.at<int>(i, CC_STAT_AREA) << endl;
		Mat mask= labels==i;
		output.setTo(getRandomRgbColor(), mask);
		// draw text with area
		stringstream ss;
		ss << "area: " << stats.at<int>(i, CC_STAT_AREA);
		putText(output,
		ss.str(),
		centroids.at<Point2d>(i),
		FONT_HERSHEY_SIMPLEX,
		0.4,
		Scalar(255,255,255));
	}
	imshow("Video", output);
}

int main( int argc, const char** argv )
{
	CommandLineParser parser(argc, argv, keys);
	parser.about("Chapter 2. v1.0.0");
	//If requires help show
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	String videoFile= parser.get<String>(0);
	// Check if params are correctly parsed in his variables
	if (!parser.check())
	{
		parser.printErrors();
		return 0;
	}
	VideoCapture cap; // open the default camera
	if(videoFile != "")
		cap.open(videoFile);
	else
		cap.open(0);
	if(!cap.isOpened()) // check if we succeeded
		return -1;
	namedWindow("Video",1);
	for(;;)
	{
		Mat frame;
		cap >> frame; // get a new frame from camera

		// Reducimos el noise de la imagen
		frame = removeNoise(frame);

		// Removemos el background por aproximacion
		frame = removeLight(frame, calculateLightPattern(frame), 1);

		// Convertimos la imagen a escala de grises
		frame = convertToBlackAndWhite(frame);

		// Conseguimos los componentes de la imagen
		imshow("Video", frame);
		//connectedComponentsStats(frame);
		waitKey(100);
	}
	// Release the camera or video cap
	cap.release();
	return 0;
}
