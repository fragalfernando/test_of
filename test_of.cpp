#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iomanip>
#include <vector>
#include <cstdio>
#include <ctime>
#include <string>
#include <unistd.h>
#include "opencv2/xfeatures2d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include "CycleTimer.h"

//#define DEBUG

#ifdef DEBUG
/* When debugging is enabled, these form aliases to useful functions */
#define dbg_printf(...) printf(__VA_ARGS__); 
#else
/* When debugging is disnabled, no code gets generated for these */
#define dbg_printf(...)
#endif

#define SUCCESS 0
#define INVALID_PATCH_SIZE 1
#define OUT_OF_FRAME 2
#define ZERO_DENOMINATOR 3

using namespace cv;
using namespace std;

/* Total images*/
int total_images = 205;
int keypoints = 18;

/* Image and coordinates directory */
string test_dir = "./data/";
string output_dir = "./output/";
string format_img = ".jpg";
string coordinates_fname = "output.log";

const char *coordinate_file = "./data/output.log";

char compute_lk(vector<float> &ix, vector<float> &iy,
                vector<float> &it, pair<float,float> &delta)
{
    float sum_xx = 0.0, sum_yy = 0.0, sum_xt = 0.0,
                        sum_yt = 0.0, sum_xy = 0.0;
    float num_u,num_v, den, u, v;

    /* Calculate sums */
    for (int i = 0; i < ix.size(); i++)
    {
        sum_xx += ix[i] * ix[i];
        sum_yy += iy[i] * iy[i];
        sum_xy += ix[i] * iy[i];
        sum_xt += ix[i] * it[i];
        sum_yt += iy[i] * it[i];
    }

    /* Get numerator and denominator of u and v */
    den = (sum_xx*sum_yy) - (sum_xy * sum_xy);

    if (den == 0.0) return ZERO_DENOMINATOR;

    num_u = (-1.0 * sum_yy * sum_xt) + (sum_xy * sum_yt);
    num_v = (-1.0 * sum_xx * sum_yt) + (sum_xt * sum_xy);

    u = num_u / den;
    v = num_v / den;
    delta.first = u;
    delta.second = v;

    return SUCCESS;
}

void get_vectors(vector< vector<float> > &patch, vector< vector<float> > &patch_it,
                 int patch_size, vector<float> &ix, vector<float> &iy, vector<float> &it)
{
    for (int i = 1; i <= patch_size; i++)
        for (int j = 1; j <= patch_size; j++)
        {
            it.push_back(patch_it[i][j]);
            ix.push_back((patch[i][j+1] - patch[i][j-1])/2.0);
            iy.push_back((patch[i+1][j] - patch[i-1][j])/2.0);
        }
}

char extract_patch(int x, int y, int patch_size,
                   Mat &image, vector< vector<float> > &patch)
{
    int radix = patch_size / 2;

    if ( ((x - (radix + 1)) < 0) ||
         ((x + (radix + 1)) >= image.cols) ||
         ((y - (radix + 1)) < 0) ||
         ((y + (radix + 1)) >= image.rows))
        return OUT_OF_FRAME;

    for (int i = -radix-1; i <= radix+1; i++)
        for (int j = -radix-1; j <= radix+1; j++)
            patch[i+radix+1][j+radix+1] = image.at<float>(y+i,x+j);

    return SUCCESS;

}

void get_opt_flow(vector<Point2f> &coord_in,
                  vector<Point2f> &coord_out,
                  Mat &prev,
                  Mat &next,
                  vector<char> &status,
                  int patch_size = 5)
{
    /* Empty coordinates */
    if (coord_in.size() == 0)
        return;

    Mat It = next - prev;

    /* Even width of patch not valid */
    if (patch_size % 2 == 0)
        status[0] = INVALID_PATCH_SIZE;

    /* Process all pixel requests */
    for (int i = 0; i < coord_in.size(); i++)
    {
        /* Extract a patch around the image */
        vector< vector<float> > patch(patch_size + 2,
                                    vector<float>(patch_size + 2));
        vector< vector<float> > patch_it(patch_size + 2,
                                    vector<float>(patch_size + 2));
  
        status[i] = extract_patch((int)coord_in[i].x,(int)coord_in[i].y,
                      patch_size, prev, patch);


        if (status[i]) {cout<<"UPSP!"<<endl; continue;} 

        status[i] = extract_patch((int)coord_in[i].x,(int)coord_in[i].y,
                      patch_size, It, patch_it);

        if (status[i]) {cout<<"UPSP2!"<<endl; continue;} 

        /* Get the Ix, Iy and It vectors */
        vector<float> ix, iy, it;
        get_vectors(patch, patch_it, patch_size, ix, iy, it);

        /* Calculate optical flow */
        pair<float,float> delta;
        status[i] = compute_lk(ix, iy, it, delta);

        if (status[i]) {cout<<"UPSOF!"<<endl; continue;}

        /* OPTICAL FLOW SUCEED */
        coord_out[i].x =  delta.first + coord_in[i].x;
        coord_out[i].y =  delta.second + coord_in[i].y;
    }

}

void drawKeyPoints(Mat image, int* x, int* y, int n, std::string output_file){

    Mat target;
    cv::cvtColor(image, target, CV_GRAY2BGR);

    for(int i=0;i<n;i++,x++,y++){
        if (!*x && !*y) continue;

        Point center = Point(*x, *y);

        cv::circle(target, center, 3, Scalar(255, 0, 0), 1);
    }

    imwrite(output_file, target);
}

void draw_both(Mat image, int* x, int* y, int n, vector<Point2f> &of, std::string output_file){

    Mat target;
    cv::cvtColor(image, target, CV_GRAY2BGR);

    for(int i=0;i<n;i++,x++,y++){
        if (!*x && !*y) continue;

        Point center = Point(*x, *y);

        cv::circle(target, center, 3, Scalar(255, 0, 0), 1);
    }

    for (int i = 0; i < of.size(); i++)
        cv::circle(target,of[i],3,Scalar(0,255,0),1);

    imwrite(output_file, target);
}

void draw_all(Mat image, int* x, int* y, int n, vector<Point2f> &of, vector<Point2f> &custom, 
              std::string output_file){

    Mat target;
    cv::cvtColor(image, target, CV_GRAY2BGR);

    for(int i=0;i<n;i++,x++,y++){
        if (!*x && !*y) continue;

        Point center = Point(*x, *y);

        cv::circle(target, center, 3, Scalar(255, 0, 0), 1);
    }

    for (int i = 0; i < of.size(); i++)
        cv::circle(target,of[i],3,Scalar(0,255,0),1);

    for (int i = 0; i < custom.size(); i++)
        cv::circle(target,custom[i],3,Scalar(0,0,255),1);

    imwrite(output_file, target);
}


void points2crd(int* x, int* y, int n, vector<Point2f> &output_crd){

    for(int i=0;i<n;i++,x++,y++)
    {
        if (!*x && !*y) continue;

        Point p = Point2d((float)*x, (float)*y);
        output_crd.push_back(p);
    }

}

int main(int argc, char ** argv) 
{

    Mat input, input_float;
    
    freopen(coordinate_file,"r",stdin);
    Mat prev, current, fprev, fcurrent;
    vector<Point2f> tracking_points[2];
    vector<Point2f> custom_points[2];

    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
    Size winSize(21,21);
    int x[keypoints*30];
    int y[keypoints*30];
    float fx, fy, conf;

    for (int i = 0; i < total_images; i++)
    {
        int persons;
        cout<<"Processing Frame "<<i<<endl;

        string number = ""; 
        char buffer[15];
        sprintf(buffer,"%d",i);
        string temp(buffer);
 
        if (i > 99) number = temp;
        else if (i > 9) number = "0" + temp;
        else number = "00" + temp;
        
        /* Load image */
        string img = test_dir + number + format_img; 
        string output_image = output_dir + number + format_img;        
        input = cv::imread(img, CV_LOAD_IMAGE_GRAYSCALE);                                      
        input.convertTo(input_float, CV_32F);
        input_float *= (1/255.00);

        cin>>persons;
        
        memset(x,0,keypoints*30*sizeof(int));
        memset(y,0,keypoints*30*sizeof(int));
	
        for (int p = 0; p < persons; p++)
        {
	    /* Add all keypoints */
	    for (int k = 0; k < keypoints; k++)
	    {	      	    
	        cin>>fx>>fy>>conf;
                if (conf < 0.01)
	            continue;	

                x[p*keypoints + k] = (int) fx;
                y[p*keypoints + k] = (int) fy;
            }
	}
        /* Set tracking points */
        if (i == 50)
        {
            fcurrent = input_float.clone();
            current = input;
            points2crd(x, y,keypoints*persons, tracking_points[1]);
            points2crd(x, y,keypoints*persons, custom_points[1]);
            custom_points[0].resize(custom_points[1].size());
        } 
        /* Draw OF keypoint estimates for frames (3,15) */
        if (i > 50 && i < 200)
        {
            vector<uchar> status;
            vector<float> err;
            vector<char> mystatus(custom_points[0].size());
    
            std::swap(tracking_points[1], tracking_points[0]);   
            std::swap(custom_points[1], custom_points[0]);
            cv::swap(fcurrent, fprev);
            cv::swap(current, prev);

            fcurrent = input_float.clone();
            current = input;

            //for (int r = 0; r < fcurrent.rows; r++)
            //    for (int s = 0; s < fcurrent.cols; s++)
            //        if (fcurrent.at<float>(r,s) != fprev.at<float>(r,s))
            //            cout<<"Found one element different"<<endl;
                
            calcOpticalFlowPyrLK(prev, current, tracking_points[0], tracking_points[1], 
                                 status, err, winSize, 0, termcrit, 0, 0.001);
            get_opt_flow(custom_points[0],custom_points[1],fprev, fcurrent, mystatus, 51);
            //draw_both(input_float,x,y,keypoints * persons,tracking_points[1],output_image);
            draw_all(input_float,x,y,keypoints * persons,tracking_points[1],custom_points[1], output_image);
        }     
        else    
            drawKeyPoints(input_float, x, y, keypoints * persons, output_image);
    }
    return 0;

}
