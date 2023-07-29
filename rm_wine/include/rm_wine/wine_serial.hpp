#pragma once 

#include <memory>
#include "opencv2/opencv.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/float32.hpp"
#include "std_msgs/msg/float64.hpp"
#include "cv_bridge/cv_bridge.h"
#include "auto_aim_interfaces/msg/target.hpp"
#include "rm_wine/yolov7_kpt.hpp"
#include "opencv2/video/tracking.hpp"

#define VIRTUAL //开启可视化调图

namespace rm_wine{

yolo_kpt DEMO;
std::vector<yolo_kpt::Object> result;//检测结果集合
cv::TickMeter meter;
struct Point9f {
    cv::Point3f target_allangle;
    cv::Point2f current_position;
    cv::Point2f current_vector;
    cv::Point2d current_time;
};
std::vector<Point9f> target_infos;
const int stateNum=2; 
const int measureNum=2; 	
cv::Mat measurement = cv::Mat::zeros(measureNum, 1, CV_32F);
cv::Point2f nextPosition;

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
        std::vector<cv::Point3f> tmp_infos;
        Winepoints()
        {
            centerpoints.resize(0);
            helppoints.resize(0);
            targetpoint.x = -1;
            targetpoint.y = -1;
            centerpoint.x = 0;
            centerpoint.y = 0;
            helppoints_polar.resize(0);
            targetpoint_polar.x = 0;
            targetpoint_polar.y = 0;
            centerpoint_polar.x = 0;
            centerpoint_polar.y = 0;
            tmp_infos.resize(0);
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
        float calculateAngleAverage() {
            int size = tmp_infos.size();
            if (size == 0) {
                return 0.0f;
            }
            float totalAngle = std::accumulate(tmp_infos.begin(), tmp_infos.end(), 0.0f,
                [](float sum, const cv::Point3f& info) {
                    return sum + info.z; // 将 angle 加到总和中
                });
            return totalAngle / size;
        }
    };
    Wine(const rclcpp::NodeOptions & options);
    
    ~Wine();
    
private:
    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_velocity;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr publisher_timestamp;
    rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr publisher_angle;
    Point9f detectPoint(const cv::Mat& src_img,const std::vector<yolo_kpt::Object>& ObjR,builtin_interfaces::msg::Time_<std::allocator<void>> & timestamp);
    void predictPoint(const cv::Mat& src_img,const Point9f& temp,std::vector<Point9f> temp_target_infos);
    cv::Point2f calNextPosition(cv::Point2f point, cv::Point2f org, float rotate_angle);
    double deltaAngle(double time);
    void fittingCurve(const Point9f& temp,std::vector<Point9f> temp_target_infos);
    void fitting_a_w(std::vector<Point9f> temp_target_infos);
    void fitting_t(std::vector<Point9f> temp_target_infos);
    double get_F_s(int n, double f_k, int k, int _N);
    double get_F_c(int n, double f_k, int k, int _N);
    double get_F(int n, int _N,std::vector<Point9f> temp_target_infos);
    double get_integral(double t_,std::vector<Point9f> temp_target_infos);
    void clearData(std::vector<Point9f> temp_target_infos);

    Point9f detectpoints;

    float wine_rotate=0;
    float distance_threshold = 15;
    std::vector<cv::Point2f> helppoints_polar_last;
    cv::Point2f targetpoint_polar_last;
    cv::Point3d time_now_last_stamp;
    cv::Point2f temp_center_abs;
    float angle_average_last=0;
    bool is_time_start_set = false;
    double current_time_start=0;
    float temp_angular_velocity_last=0;

    double _a = 0.9;  // 振幅 [0.780, 1.045]
    double _w = 1.9;  // 频率 [1.884, 2.000]
    double t_0 = 0.0; // 初相

    double MAX_T0 = 3.34; // 最大周期
    double T0_N = 30;     // 相位采样数
    double DT = 0.01;     // 采样时间间隔，单位：秒
    double N = 400;       // 角速度采样数
    double DELAY_TIME = 0.37; // 预测时间，单位：秒|需要调整
    int DN = 1; // 逐差法测速度间距

    double start_time;
    bool is_Inited = false;           // 大符拟合是否初始化
};
}


