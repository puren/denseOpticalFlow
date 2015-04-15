#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>
#include "H5Cpp.h"
#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif
using namespace cv;
using namespace std;



void drawOptFlowMap (const Mat& flow, Mat& cflowmap, int step, const Scalar& color) {
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const Point2f& fxy = flow.at< Point2f>(y, x);
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), 1, color, -1);
        }
}

int writeOpticalFlow(char* fileName, int nx, int ny, int rank, const Mat& flow)
{
    /*
     * Try block to detect exceptions raised by any of the calls inside it
     */
    int x,y;
    try
    {
        /*
         * Turn off the auto-printing when failure occurs so that we can
         * handle the errors appropriately
         */
        //Exception::dontPrint();
        H5File* file = new H5File( fileName, H5F_ACC_TRUNC );
        
        Group* group = new Group( file->createGroup( "/Data" ));
        /*
         * Create property list for a dataset and set up fill values.
         */
        float fillvalue = 0.0;   /* Fill value for the dataset */
        DSetCreatPropList plist;
        plist.setFillValue(PredType::NATIVE_FLOAT, &fillvalue);

        /*
         * Create dataspace for the dataset in the file.
         */
        hsize_t fdim[] = {nx, ny}; // dim sizes of ds (on disk)
        DataSpace fspace_x( rank, fdim );
        DataSpace fspace_y( rank, fdim );

        /*
         * Create dataset and write it into the file.
         */
        DataSet* dataset_x = new DataSet(file->createDataSet(
                                            "optical_flow_x", PredType::NATIVE_FLOAT, fspace_x, plist));
        DataSet* dataset_y = new DataSet(file->createDataSet(
                                                             "optical_flow_y", PredType::NATIVE_FLOAT, fspace_y, plist));
        
        float values_x[nx][ny];
        float values_y[nx][ny];
        for (y=0; y<ny; y++) {
            for (x=0; x<nx; x++) {
                const Point2f& fxy = flow.at< Point2f>(y, x);
                values_x[x][y]=fxy.x;
                values_y[x][y]=fxy.y;
            }
        }
        
        //dataset_x->write( values_x, PredType::NATIVE_FLOAT, H5S_ALL, H5S_ALL);
        //dataset_y->write( values_y, PredType::NATIVE_FLOAT, H5S_ALL, H5S_ALL);
        delete dataset_x;
        delete dataset_y;
        delete file;

    }// end of try block
    // catch failure caused by the H5File operations
    catch( FileIException error )
    {
        error.printError();
        return -1;
    }

    // catch failure caused by the DataSet operations
    catch( DataSetIException error )
    {
        error.printError();
        return -1;
    }

    // catch failure caused by the DataSpace operations
    catch( DataSpaceIException error )
    {
        error.printError();
        return -1;
    }
    return 0;
}


int main()
{
    //cv::namedWindow( "View", CV_WINDOW_NORMAL | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED );
    
    Mat prv, nxt;
    Mat prv_show, nxt_show;
    
    int start_frame = 180;
    char fileName1[100];
    sprintf(fileName1, "/Users/puren/projects/pillow_data/puren_deformable/rec1/png1/frame_%07d.png", start_frame);
    
    VideoCapture stream1(fileName1);
    
    if (!(stream1.read(prv))) {
        std::cout<<"no such file"<<fileName1<<std::endl;
        return 0;
    }
    std::cout<<fileName1<<std::endl;
    cvtColor(prv, prv, CV_BGR2GRAY);
    
    imshow( "View", prv );
    cv::waitKey(100);
    
    int i = start_frame;
    while (true) {
        i++;
        char fileName2[100];
        sprintf(fileName2, "/Users/puren/projects/pillow_data/puren_deformable/rec1/png1/frame_%07d.png", i);
        std::cout<<fileName2<<std::endl;
        VideoCapture stream2(fileName2);
        if (!(stream2.read(nxt))) {
            std::cout<<"no such file "<<fileName2<<std::endl;
            break;
        }
        cvtColor(nxt, nxt, CV_BGR2GRAY);
        imshow( "View", nxt );
        
        Mat_<Point2f> flow;
        Ptr<DenseOpticalFlow> tvl1 = createOptFlow_DualTVL1();
        tvl1->calc(prv, nxt, flow);
        
       bool isWrite=true;
        Mat cflow;
        cvtColor(prv, cflow, CV_GRAY2BGR);
        drawOptFlowMap(flow, cflow, 40, CV_RGB(0, 255, 0));
        imshow("OpticalFlowFarneback", cflow);

        cv::waitKey(1);
        
        //write to hdf5 file
//        char file_of[100];
//        sprintf(file_of, "/Users/puren/projects/pillow_data/puren_deformable/rec1/png1/flow_pur/frame_%07d.h5", i);
//        int nx = prv.cols;
//        int ny = prv.rows;
//        writeOpticalFlow(file_of, nx, ny, 2, flow);
        
        prv = nxt.clone();
    }
}