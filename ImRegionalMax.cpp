#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>

using namespace cv;

/**
 * @brief Checks whether the pixel is part of Peak/ is Peak, We are looking for 8-conn points
 * 
 * @param mxz 
 * @param conn_points 
 * @return true 
 * @return false 
 */
bool isPeak(cv::Mat mx[], std::vector<cv::Point>& conn_points){
    cv::Point poi_point = conn_points.back();
    int row = poi_point.y;
    int col = poi_point.x;
    float poi_val = mx[0].at<float>(poi_point);
    bool isPeakEle = true;
    for(int mask_row = row - 1; mask_row <= row + 1 ; mask_row++){
        for(int mask_col = col - 1; mask_col <= col + 1 ; mask_col++){
            if(mask_row == row && mask_col == col){
                continue;
            }
            float conn_pt_val = mx[0].at<float>(mask_row, mask_col);
            if(poi_val < conn_pt_val){
                isPeakEle = false;
                break;
            }
            if(poi_val == conn_pt_val){
                int Peak_status = mx[1].at<uchar>(mask_row, mask_col);
                if(Peak_status == 0){
                    isPeakEle = false;
                    break;
                }
                else if(Peak_status == 1){
                    isPeakEle = true;
                    break;
                }
                else{
                    cv::Point p(mask_col, mask_row);
                    std::vector<cv::Point>::iterator it;
                    it = std::find (conn_points.begin(), conn_points.end(), p);
                    if(it == conn_points.end()){
                        conn_points.push_back(p);    
                        isPeakEle = isPeakEle &&  isPeak(mx, conn_points);
                    }
                }
            }            
        }
        if(isPeakEle ==  false){
            break;
        }
    }
    return isPeakEle;
}

/**
 * @brief This is equivalent to imregionalmax(img, conn = 8) of Matlab
 * It takes floating point matrix, finds all local maximas and put them in 8UC1 matrix
 * pls refer: https://in.mathworks.com/help/images/ref/imregionalmax.html for imregionalmax
 * eg: I/P
 *      10    10    10    10    10    10    10    10    10    10
        10    22    22    22    10    10    44    10    10    10
        10    22    22    22    10    10    10    45    10    10
        10    22    22    22    10    10    10    10    44    10
        10    10    10    10    10    10    10    10    10    10
        10    10    10    10    10    33    33    33    10    10
        10    10    10    10    10    33    33    33    10    10
        10    10    10    10    10    33    33    33    10    10
        10    10    10    10    10    10    10    10    10    10
        10    10    10    10    10    10    10    10    10    10        
 * O/P
        0   0   0   0   0   0   0   0   0   0
        0   1   1   1   0   0   0   0   0   0
        0   1   1   1   0   0   0   1   0   0
        0   1   1   1   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   1   1   1   0   0
        0   0   0   0   0   1   1   1   0   0
        0   0   0   0   0   1   1   1   0   0
        0   0   0   0   0   0   0   0   0   0
        0   0   0   0   0   0   0   0   0   0
 * @param src 
 * @param conn 
 * @return cv::Mat 
 */
cv::Mat imregionalmax(cv::Mat& src){
    cv::Mat padded;
    cv::copyMakeBorder(src, padded, 1, 1, 1, 1, BORDER_CONSTANT, Scalar::all(-1));
    Mat mx_ch1(padded.rows, padded.cols, CV_8UC1, Scalar(2)); //Peak elements will be represented by 1, others by 0, initializing Mat with 2 for differentiation 
    cv::Mat mx[2] = {padded, mx_ch1}; //mx[0] is padded image, mx[1] is regional maxima matrix
    int mx_rows = mx[0].rows;
    int mx_cols = mx[0].cols;
    cv::Mat dest(mx[0].size(), CV_8UC1);

    //Check each pixel for local max
    for(int row = 1; row < mx_rows - 1 ; row++){
        for(int col = 1; col < mx_cols - 1 ; col++){
            std::vector<cv::Point> conn_points; //this vector holds all connected points for candidate pixel
            cv::Point p(col, row);
            conn_points.push_back(p);
            bool isPartOfPeak = isPeak(mx, conn_points);
            if(isPartOfPeak){
               mx[1].at<uchar>(row, col) = 1; 
            }            
            else{
               mx[1].at<uchar>(row, col) = 0; 
            }
        }
    }
    dest = mx[1](cv::Rect(1, 1,src.cols, src.rows));
    return dest;
}


int main(){
    Mat src = (Mat_<float>(10, 10) << 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 
                                    10, 22, 22, 22, 10, 10, 44, 10, 10, 10,
                                    10, 22, 22, 22, 10, 10, 10, 45, 10, 10,
                                    10, 22, 22, 22, 10, 10, 10, 10, 44, 10,
                                    10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                    10, 10, 10, 10, 10, 33, 33, 33, 10, 10,
                                    10, 10, 10, 10, 10, 33, 33, 33, 10, 10,
                                    10, 10, 10, 10, 10, 33, 33, 33, 10, 10,
                                    10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                                    10, 10, 10, 10, 10, 10, 10, 10, 10, 10
                                );
    cv::Mat peaks = imregionalmax(src);

    for(int row = 0 ; row < peaks.rows ; row++) {
        for(int col = 0 ; col < peaks.cols ; col++) {
            std::cout << (int)peaks.at<uchar>(row, col)<< ", ";
        }
        std::cout<<std::endl;
    }
}