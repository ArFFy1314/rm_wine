#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include <opencv2/highgui/highgui.hpp>
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.h"
#include "image_transport/image_transport.hpp"
#include <filesystem>

#define VIDEO_PATH "C++_inference_openvino_kpt/222.mp4"
using namespace std::chrono_literals;

class Imgpub : public rclcpp::Node {
public:
    Imgpub(const rclcpp::NodeOptions & options)
        : Node("virtual_imgpub", options)
    {
        RCLCPP_INFO(this->get_logger(), "virtual_imgpub is starting");
        std::filesystem::path videoPath(VIDEO_PATH);
        if (std::filesystem::exists(videoPath)) {
            openVideo();
        } else {
            RCLCPP_INFO(this->get_logger(), "无法找到视频文件");
            return;
        }
        double frame_rate = this->declare_parameter("frame_rate", 50.0);
        timer_interval_ = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(1.0 / frame_rate));

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>("/virtual/raw_img", 10);
        timer_ = this->create_wall_timer(timer_interval_, std::bind(&Imgpub::timer_callback, this));
    }

private:
    void openVideo()
    {
        cap.open(VIDEO_PATH);
        if (!cap.isOpened()) {
            RCLCPP_INFO(this->get_logger(), "无法打开视频文件");
        }
    }

    void timer_callback()
    {
        if (!cap.isOpened()) {
            openVideo(); // 重新打开视频文件
            if (!cap.isOpened()) {
                RCLCPP_INFO(this->get_logger(), "文件读取完毕，正在重新加载");
                return;
            }
        }

        cv::Mat src_img;
        bool ret = cap.read(src_img);
        if (!ret) {
            cap.release(); // 释放视频资源
            return;
        }

        sensor_msgs::msg::Image::UniquePtr message = std::make_unique<sensor_msgs::msg::Image>();
        message->header.stamp = this->now();

        rclcpp::Time timestamp = message->header.stamp;
        double timestamp_value = timestamp.seconds();
        RCLCPP_INFO(this->get_logger(),"Timestamp_S_value: %f\n", timestamp_value);
        // double nanos = static_cast<float>(message->header.stamp.nanosec) / 1e9;
        // RCLCPP_INFO(this->get_logger(),"Timestamp_N_value: %f\n", nanos);
        
        message->header.frame_id = "frame";
        message->encoding = "bgr8";
        message->height = src_img.rows;
        message->width = src_img.cols;
        message->step = src_img.cols * src_img.elemSize();
        size_t size = src_img.cols * src_img.rows * src_img.elemSize();
        message->data.resize(size);
        memcpy(&message->data[0], src_img.data, size);

        publisher_->publish(std::move(message));
    }

    cv::VideoCapture cap;
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;
    std::chrono::nanoseconds timer_interval_;
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions options;
    rclcpp::spin(std::make_shared<Imgpub>(options));
    rclcpp::shutdown();
    return 0;
}
