#include <iostream>
#include "rm_wine/wine_serial.hpp"

namespace rm_wine{

Wine::Wine(const rclcpp::NodeOptions & options) : Node("rm_wine_serial", options) {
    RCLCPP_INFO(this->get_logger(), "wine is starting");
    // 订阅原始图像消息
    auto callback = std::bind(&Wine::imageCallback, this, std::placeholders::_1);
    subscription_ = this->create_subscription<sensor_msgs::msg::Image>("/virtual/raw_img", 10, callback);
}

void Wine::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg){
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
    
    detectpoints = detectPoint(image_ptr->image,result);
    //DFT 差分拟合预测

    //小孔或者cj距离

}
Wine::Winepoints Wine::detectPoint(const cv::Mat& src_img,const std::vector<yolo_kpt::Object>& ObjR){
    Winepoints temppoints;
    //循环至多5片扇叶
    for (const yolo_kpt::Object& object : ObjR){
        cv::Point2f fun_point;
        std::vector<bool> valid_keypoints(5, false);
        for (int i = 0; i < static_cast<int>(object.kpt.size()); i++) {
            if (i != 2 && object.kpt[i].x != 0 && object.kpt[i].y != 0) {
                valid_keypoints[i] = true;
            }
        }
        if (valid_keypoints[0] && valid_keypoints[3]) {
            if (valid_keypoints[1] && valid_keypoints[4]) {
                fun_point = (object.kpt[0] + object.kpt[1] + object.kpt[3] + object.kpt[4]) * 0.25;
            } else {
                fun_point = (object.kpt[0] + object.kpt[3]) * 0.5;
            }
        } else if (valid_keypoints[1] && valid_keypoints[4]) {
            
            fun_point = (object.kpt[1] + object.kpt[4]) * 0.5;
        } else {
            fun_point = cv::Point2f(object.rect.x + object.rect.width / 2, object.rect.y + object.rect.height / 2);
        }

        if (object.label == 0 || object.label == 2){
            temppoints.targetpoint=fun_point;
        }
        else
        {
            temppoints.helppoints.push_back(fun_point);
        }
        temppoints.centerpoints.push_back(object.kpt[2]);
#ifdef VIRTUAL
        float x0 = object.rect.x;
        float y0 = object.rect.y;
        cv::rectangle(src_img, cv::Point(x0, y0), cv::Point(x0+ object.rect.width, y0+ object.rect.height), cv::Scalar(255, 255, 255), 1);//目标框
        cv::circle(src_img, temppoints.targetpoint, 2, cv::Scalar(255, 255, 255), 2);//待击打靶装甲中心
        for (int i = 0; i < KPT_NUM; i++)//循环遍历识别的角点
            if (DETECT_MODE == 1){
                if (i == 2)
                    cv::circle(src_img, object.kpt[i], 4, cv::Scalar(163, 164, 163), 4);//R字
                else
                    cv::circle(src_img, object.kpt[i], 3, cv::Scalar(0, 255, 0), 3);//检测点
            }
        std::string label = std::to_string(object.label)  + ": " + cv::format("%.2f", object.prob);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.25, 1, &baseLine);
        y0 = std::max(int(y0), labelSize.height);
        cv::rectangle(src_img, cv::Point(x0, y0 - round(1.5 * labelSize.height)),
                    cv::Point(x0 + round(2 * labelSize.width), y0 + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(src_img, label, cv::Point(x0, y0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(), 1.5);
#endif
    }
    temppoints.centerpoint = temppoints.calculateCenterpointsAverage();
#ifdef VIRTUAL
    cv::circle(src_img, temppoints.centerpoint, 8, cv::Scalar(255, 0, 0), 2);//R字  
    cv::imshow("Inference test", src_img);
    cv::waitKey(1);
#endif
    return temppoints;
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