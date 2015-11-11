#include <stdio.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

const float mChannelRange[2] = {0.0, 255.0};
RNG rng(12345);

struct quadrangle{
  Point min_y_pt;
  Point max_y_pt;
  Point min_x_pt;
  Point max_x_pt;
};

quadrangle getQuadrangle(Mat src){
  quadrangle q;
  vector<Point> points;
  Mat_<uchar>::iterator it = src.begin<uchar>();
  Mat_<uchar>::iterator end = src.end<uchar>();
  

  for(; it != end; ++it){
    if(*it){
      //assign initial value to all
      if (q.min_y_pt.x == 0 && q.min_y_pt.y == 0) {
        q.min_x_pt = it.pos();
        q.max_x_pt = it.pos();
        q.min_y_pt = it.pos();
        q.max_y_pt = it.pos();
        continue;
      }
      

      int x = it.pos().x;
      int y = it.pos().y;
      
      if (y < q.min_y_pt.y) {
        q.min_y_pt = it.pos();
      } else if (y > q.max_y_pt.y) {
        q.max_y_pt = it.pos();
      }

      if (x < q.min_x_pt.x) {
        q.min_x_pt = it.pos();
      } else if (x > q.max_x_pt.x) {
        q.max_x_pt = it.pos();
      }
    }
  }
  return q;
}

Mat BackProject(Mat image, MatND hist) {
  Mat result = image.clone();
  const float* channel_ranges[] = {mChannelRange, mChannelRange, mChannelRange};

  int* mChannelNumbers = new int[3];
  for (int count=0; count<3; count++) {
    mChannelNumbers[count] = count;
  }

  calcBackProject(&image, 1, mChannelNumbers, hist, result, channel_ranges, 255.0);
  return result;
}

int main(int argc, char** argv )
{
  string imgsDir = "Books/";
  string imgFile = "BookView09.JPG"; 


  Mat sample, dst;
  sample = imread("bluesample.png");  
  if ( !sample.data) {
    cout << ("No image data for sample blue.\n");
    return -1;
  }
  Mat hls_sample;
  cvtColor(sample, hls_sample, CV_BGR2HLS);
  
  //Get the 3D histogram of the hls_sample image
  MatND sampleHist;
  int number_of_bins = 10;
  int sampleNumberChannels = hls_sample.channels();
  int* sampleChannelNumbers = new int[sampleNumberChannels];
  int* sampleNumberBins = new int[sampleNumberChannels];
  
  for (int count=0; count<sampleNumberChannels; count++) {
    sampleChannelNumbers[count] = count;
    sampleNumberBins[count] = number_of_bins;
  }
  
  const float* channel_ranges[] = {mChannelRange, mChannelRange, mChannelRange};
  calcHist(&hls_sample, 1, sampleChannelNumbers, Mat(), sampleHist, sampleNumberChannels, sampleNumberBins, channel_ranges);
  
  //Normalize the histogram of the samples and backproject the src image
  normalize(sampleHist, sampleHist, 1.0);
  
  Mat src, hls_src, backprojected_src;
  src = imread(imgsDir + imgFile,1);
  cvtColor(src, hls_src, CV_BGR2HLS);

  calcBackProject(&hls_src, 1, sampleChannelNumbers, sampleHist, backprojected_src, channel_ranges, 255.0);
  imshow("after backprojection", backprojected_src);

  Mat bin_src;
  threshold( backprojected_src, bin_src, 25, 255, THRESH_BINARY);  
  
  //Closing (dilation -> erosion) in order to get rid of small objects that may still remain in the background
  Mat closed_src;
  Mat element = getStructuringElement( MORPH_RECT, Size( 5, 5));
  morphologyEx(bin_src, closed_src, MORPH_CLOSE, element, Point(-1,-1), 1);
  imshow("after closing", closed_src);  
  
  //Draw the quadrangle
  Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
  quadrangle q = getQuadrangle(bin_src);
  line(bin_src, q.max_y_pt, q.min_x_pt, color, 1, 8);
  line(bin_src, q.max_y_pt, q.max_x_pt, color, 1, 8);
  line(bin_src, q.min_y_pt, q.min_x_pt, color, 1, 8);
  line(bin_src, q.min_y_pt, q.max_x_pt, color, 1, 8);
  imshow("Rects", bin_src);

  waitKey(0);

  return 0;
}
