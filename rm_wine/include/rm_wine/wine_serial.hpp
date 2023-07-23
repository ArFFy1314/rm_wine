#pragma once 

#include <memory>
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "auto_aim_interfaces/msg/target.hpp"
#include "rm_wine/yolov7_kpt.hpp"


namespace rm_wine{

yolo_kpt DEMO;
std::vector<yolo_kpt::Object> result;
cv::TickMeter meter;

class Wine :public rclcpp::Node
{
public:
    Wine(const rclcpp::NodeOptions & options);
    ~Wine();
    
private:
    // static void signalHandler(int signum);
    // static bool running;

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    
    //Camera params
    cv::Mat CAMERA_MATRIX;    //IntrinsicMatrix		  fx,fy,cx,cy
    cv::Mat DISTORTION_COEFF; //DistortionCoefficients k1,k2,p1,p2

    float yaw;  
    float pitch; 
};
}


