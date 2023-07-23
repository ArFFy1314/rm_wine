#include <iostream>
#include "rm_wine/wine_serial.hpp"

namespace rm_wine{

Wine::Wine(const rclcpp::NodeOptions & options) : Node("rm_wine_serial", options) {
    RCLCPP_INFO(this->get_logger(), "wine is starting");

    // std::signal(SIGINT, &Wine::signalHandler);

    // 订阅原始图像消息
    auto callback = std::bind(&Wine::imageCallback, this, std::placeholders::_1);
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>("/virtual/raw_img", 10, callback);
}

void Wine::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
    // 将ROS图像消息转换为OpenCV图像
    cv_bridge::CvImagePtr image_ptr;
    try {
        image_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
        RCLCPP_ERROR(this->get_logger(), "can not transport image: %s", e.what());
        return;
    }

    // 图像处理
    meter.start(); // 计时开始
    result = DEMO.work(image_ptr->image);
    meter.stop(); // 计时结束
    RCLCPP_INFO(this->get_logger(), "Time: %f\n", meter.getTimeMilli());
    meter.reset();
}
Wine::~Wine(){}

}

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    rclcpp::spin(std::make_shared<rm_wine::Wine>(options));
    rclcpp::shutdown();
    return 0;
}