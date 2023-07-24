#pragma once 

#include <memory>
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "auto_aim_interfaces/msg/target.hpp"
#include "rm_wine/yolov7_kpt.hpp"

#define VIRTUAL //开启可视化调图

namespace rm_wine{

yolo_kpt DEMO;
std::vector<yolo_kpt::Object> result;//检测结果集合
cv::TickMeter meter;

class Wine :public rclcpp::Node
{
public:
    struct Winepoints {
        std::vector<cv::Point2f> centerpoints;
        std::vector<cv::Point2f> helppoints;
        cv::Point2f targetpoint;
        cv::Point2f centerpoint;

        std::vector<cv::Point2f> helppoints_polar;
        cv::Point2f targetpoint_polar;
        cv::Point2f centerpoint_polar;
        Winepoints()
        {
            centerpoints.resize(0);
            helppoints.resize(0);
            targetpoint.x = 0;
            targetpoint.y = 0;
            centerpoint.x = 0;
            centerpoint.y = 0;
            helppoints_polar.resize(0);
            targetpoint_polar.x = 0;
            targetpoint_polar.y = 0;
            centerpoint_polar.x = 0;
            centerpoint_polar.y = 0;
        }
        cv::Point2f calculateCenterpointsAverage() {
            cv::Point2f average(0, 0);
            int size = centerpoints.size();
            if (size == 0) {
                return average;
            }
            for (const auto& point : centerpoints) {
                average += point;
            }
            average.x /= size;
            average.y /= size;
            return average;
        }
    };
    
    Wine(const rclcpp::NodeOptions & options);
    ~Wine();
    
private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    Winepoints detectPoint(const cv::Mat& src_img,const std::vector<yolo_kpt::Object>& ObjR);

    Winepoints detectpoints;
};
}


