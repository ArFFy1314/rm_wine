#include <iostream>
#include <algorithm> // for std::transform
#include <iterator> // for std::back_inserter
#include "rm_wine/wine_serial.hpp"

namespace rm_wine{
cv::KalmanFilter KF(stateNum, measureNum, 0);//状态值测量值5×1向量(x,y,△x,△y,distance)
Wine::Wine(const rclcpp::NodeOptions & options) : Node("rm_wine_serial", options) {
    RCLCPP_INFO(this->get_logger(), "wine is starting");
    
    KF.transitionMatrix = (cv::Mat_<float>(stateNum, stateNum) <<  1,1,0,1);
    KF.measurementMatrix = (cv::Mat_<float>(measureNum, measureNum)<<  1,0,0,1);
    KF.processNoiseCov = (cv::Mat_<float>(stateNum, stateNum) <<  1e-7,0,0,1e-7);
    KF.measurementNoiseCov = (cv::Mat_<float>(stateNum, stateNum) <<  1e-2,0,0,1e-2);
	setIdentity(KF.errorCovPost, cv::Scalar::all(1)); 
    randn(KF.statePost, cv::Scalar::all(0), cv::Scalar::all(KF.measurementNoiseCov.at<float>(0)));

    auto callback = std::bind(&Wine::imageCallback, this, std::placeholders::_1);

    subscription_ = this->create_subscription<sensor_msgs::msg::Image>("/virtual/raw_img", 10, callback);
    publisher_velocity = this->create_publisher<std_msgs::msg::Float32>("/wine/velocity", 10);
    publisher_timestamp = this->create_publisher<std_msgs::msg::Float64>("/wine/timestamp", 10);
    publisher_angle = this->create_publisher<std_msgs::msg::Float32>("/wine/angle", 10);
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

     // 获取时间戳
    auto timestamp = msg->header.stamp;
    // 图像处理
    meter.start(); // 计时开始
    result = DEMO.work(image_ptr->image);
    meter.stop(); // 计时结束
    // RCLCPP_INFO(this->get_logger(), "Time: %f\n", meter.getTimeMilli());
    meter.reset();
    
    detectpoints = detectPoint(image_ptr->image,result,timestamp);
    //DFT 差分拟合预测
    this->predictPoint(image_ptr->image,detectpoints,target_infos);
    //小孔或者cj距离

}
Point9f Wine::detectPoint(const cv::Mat& src_img,const std::vector<yolo_kpt::Object>& ObjR,builtin_interfaces::msg::Time_<std::allocator<void>> & timestamp){
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
            temppoints.helppoints.push_back(fun_point);
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

    temppoints.centerpoint_polar=temppoints.centerpoint-temppoints.centerpoint;
    std::transform(temppoints.helppoints.begin(), temppoints.helppoints.end(), 
        std::back_inserter(temppoints.helppoints_polar), [&](const cv::Point2f& point) {
            return point - temppoints.centerpoint;});
    temppoints.targetpoint=(temppoints.targetpoint==cv::Point2f(-1,-1))?temppoints.centerpoint:temppoints.targetpoint;
    temppoints.targetpoint_polar=temppoints.targetpoint-temppoints.centerpoint;
    
    for (const auto& new_point : temppoints.helppoints_polar) {
        for (const auto& old_point : helppoints_polar_last) {
            float distance = cv::norm(old_point - new_point);
            // RCLCPP_INFO(this->get_logger(), "distance:%f",distance);
            if (distance <= distance_threshold) {
                float angle_old=std::atan2(old_point.y,old_point.x);
                float angle_new=std::atan2(new_point.y,new_point.x);
                if (angle_old>-CV_PI&&angle_old<0) angle_old+=2*CV_PI;
                if (angle_new>-CV_PI&&angle_new<0) angle_new+=2*CV_PI;
                float angle=angle_new-angle_old;
                angle = fabsf32(angle) > CV_PI ? (angle < 0 ? angle + 2*CV_PI : angle - 2*CV_PI) : angle;
                // RCLCPP_INFO(this->get_logger(), "angle:%f",angle);
                temppoints.tmp_infos.push_back(cv::Point3f(new_point.x,new_point.y,angle));
                // temppoints.tmp_infos.emplace_back(new_point.x - old_point.x, new_point.y - old_point.y, [&]() {
                //     float angle = std::atan2(new_point.y, new_point.x) - std::atan2(old_point.y, old_point.x);
                //     if (angle < 0) angle += 2 * CV_PI;
                //     RCLCPP_INFO(this->get_logger(), "angle:%f",angle);
                //     return angle;
                // }());
                break;
            }
        }
    }
    
    //时间解算
    rclcpp::Time curremt_timestamp = timestamp;
    time_now_last_stamp.x = curremt_timestamp.seconds()-current_time_start;
    time_now_last_stamp.z=time_now_last_stamp.x-time_now_last_stamp.y;
    // RCLCPP_INFO(this->get_logger(), "time_now_last_stamp:%lf,%lf,%lf",time_now_last_stamp.x,time_now_last_stamp.y,time_now_last_stamp.z);
    //统计旋转角度平均
    Point9f temp_data;
    float temp_angle = temppoints.calculateAngleAverage();
    if(temp_angle){
        if (!is_time_start_set) {  
            current_time_start = time_now_last_stamp.x;
            time_now_last_stamp.x = 0;
            is_time_start_set = true; 
        }

        wine_rotate += temp_angle;
        wine_rotate = (wine_rotate > 100 || wine_rotate < -100) ? wine_rotate * 0.1 : wine_rotate;
        if(temp_angle*wine_rotate<=0){
            if (!target_infos.empty()) {
                Point9f last_value = target_infos.back();
                temp_angle=last_value.target_allangle.x;
            }
        }

        // float alpha=0.1f;
        // temp_angle=alpha*temp_angle+(1-alpha)*angle_average_last;
        // angle_average_last=temp_angle;

        double temp_time_interval = time_now_last_stamp.z;
        double temp_angular_velocity = temp_angle / temp_time_interval;
        double temp_angular_acceleration = (temp_angular_velocity-temp_angular_velocity_last) / temp_time_interval;
        temp_angular_velocity_last=temp_angular_velocity;

         /*
        卡尔曼
        */
        cv::Mat prediction = KF.predict();
        cv::Point predict_pt = cv::Point(prediction.at<float>(0),prediction.at<float>(1) );
        measurement.at<float>(0) = temp_angle;
		measurement.at<float>(1) = temp_angular_velocity;
        KF.correct(measurement);

        // RCLCPP_INFO(this->get_logger(), "temp_angular_velocity:%lf,%lf,%lf",temp_angle,temp_angular_velocity,temp_angular_acceleration);

        //发布速度topic
        auto message_velocity=std_msgs::msg::Float32();
        message_velocity.data=temp_angular_acceleration;
        publisher_velocity->publish(message_velocity);
        //发布时间间隔
        auto message_timestamp=std_msgs::msg::Float64();
        message_timestamp.data=time_now_last_stamp.z;
        publisher_timestamp->publish(message_timestamp);
        //发布角度
        // float topic_angle=std::atan2(temppoints.targetpoint_polar.y,temppoints.targetpoint_polar.x);
        // RCLCPP_INFO(this->get_logger(), "target:%lf,%lf",temppoints.targetpoint_polar.x,temppoints.targetpoint_polar.y);
        // if (topic_angle>-CV_PI&&topic_angle<0) topic_angle+=2*CV_PI;
        auto message_angle=std_msgs::msg::Float32();
        message_angle.data=KF.statePost.at<float>(0)*180.f/CV_PI;
        publisher_angle->publish(message_angle);

        // 存储角度、角速度和角加速度
        temp_data.target_allangle = cv::Point3f(temp_angle, temp_angular_velocity, temp_angular_acceleration);
        temp_data.current_position = temppoints.targetpoint_polar;
        temp_data.current_vector = targetpoint_polar_last-temppoints.targetpoint_polar;
        temp_data.current_time = cv::Point2d(time_now_last_stamp.x, time_now_last_stamp.z);
        // RCLCPP_INFO(this->get_logger(), "temp_data.current_time:%lf,%lf",temp_data.current_time.x,temp_data.current_time.y);
        target_infos.push_back(temp_data);
        while (target_infos.size() > N)
            target_infos.erase(target_infos.begin());
    }
    // 更新击打信息
    temp_center_abs=temppoints.centerpoint;
    helppoints_polar_last=temppoints.helppoints_polar;
    targetpoint_polar_last=temppoints.targetpoint_polar;
    time_now_last_stamp.y=time_now_last_stamp.x;
#ifdef VIRTUAL
    cv::circle(src_img, temppoints.centerpoint, 8, cv::Scalar(255, 0, 0), 2);//R字  
#endif

    return temp_data;
} 
void Wine::predictPoint(const cv::Mat& src_img,const Point9f& temp,std::vector<Point9f> temp_target_infos){
    if(!temp_target_infos.empty()) {
        fittingCurve(temp,temp_target_infos);//初步拟合
        RCLCPP_INFO(this->get_logger(), "size:%d",temp_target_infos.size());
        if (is_Inited)//正常是看有没有拟合好曲线
        {
            double delta = deltaAngle(temp.current_time.x);
            nextPosition=calNextPosition(temp.current_position, temp_center_abs, delta);
            RCLCPP_ERROR(this->get_logger(), "is_Inited:0  %f,%f",nextPosition.x,nextPosition.y);
        }
        else
        {
            double delta = CV_PI / 3 * DELAY_TIME;//小符旋转速度CV_PI/3
            nextPosition=calNextPosition(temp.current_position, temp_center_abs, delta);
            RCLCPP_INFO(this->get_logger(), "is_Inited:1  %f,%f",nextPosition.x,nextPosition.y);
        }
    }
    //nextPosition 默认是(0,0),与电控协商好，具体的追踪方式
#ifdef VIRTUAL
    cv::circle(src_img, nextPosition, 8, cv::Scalar(0, 255, 255), 2);//R字  
    cv::imshow("Inference test", src_img);
    cv::waitKey(1);
#endif
}
cv::Point2f Wine::calNextPosition(cv::Point2f point, cv::Point2f org, float rotate_angle)
{
    float angle_target_polar=std::atan2(point.y,point.x);
    if (angle_target_polar>-CV_PI&&angle_target_polar<0) angle_target_polar+=2*CV_PI;
    RCLCPP_INFO(this->get_logger(), "angle_target_polar  %lf:",angle_target_polar);
    RCLCPP_INFO(this->get_logger(), "rotate_angle  %lf:",rotate_angle);
    float next_angle=angle_target_polar + rotate_angle;
    // angle = fabsf32(angle) > CV_PI ? (angle < 0 ? angle + 2*CV_PI : angle - 2*CV_PI) : angle;
    float radius=cv::norm(point);
    RCLCPP_INFO(this->get_logger(), "radius  %lf:",radius);
    return cv::Point2f(cos(next_angle) * radius, sin(next_angle) * radius) + org;
}
double Wine::deltaAngle(double time)
{
    double t = (double)(time - start_time);
    RCLCPP_ERROR(this->get_logger(), "time  %lf:",time);
    RCLCPP_ERROR(this->get_logger(), "start_time  %lf:",start_time);
    RCLCPP_INFO(this->get_logger(), "deltaAngle_t  %lf:",t);
    return (-_a / _w) * (cos(_w * (t + DELAY_TIME + t_0)) - cos(_w * (t + t_0))) + (2.090 - _a) * DELAY_TIME;
}
void Wine::fittingCurve(const Point9f& temp,std::vector<Point9f> temp_target_infos)
{
    DT=temp.current_time.y;
    if (temp_target_infos.empty())   return;
    RCLCPP_INFO(this->get_logger(), "fittingCurve:-2  %lf:",temp_target_infos[temp_target_infos.size() - 1].current_time.x);
    RCLCPP_INFO(this->get_logger(), "fittingCurve:-3  %lf:",temp_target_infos[0].current_time.x);
    RCLCPP_INFO(this->get_logger(), "fittingCurve:-4  %lf:",(N - 1) * DT - 1);
    if (temp_target_infos[temp_target_infos.size() - 1].current_time.x - 
        temp_target_infos[0].current_time.x >= (N - 1) * DT  - 1)
    {
        fitting_a_w(temp_target_infos);
        if (isnan(_a))
        {
            _a = 0.9;
            _w = 1.9;
            t_0 = 0;
            clearData(temp_target_infos);
            return;
        }
        fitting_t(temp_target_infos);
        is_Inited = true;
    }
}
void Wine::fitting_a_w(std::vector<Point9f> temp_target_infos)
{
    int n_min = 1.884 / (2 * M_PI) * N;
    int n_max = 2.0 / (2 * M_PI) * N + 1;

    double max_i = n_min;
    double max_value = get_F(n_min, N,temp_target_infos), value = 0.0;
    for (int i = n_min + 1; i < n_max; i++)
    {
        value = get_F(i, N,temp_target_infos);
        if (value > max_value)
        {
            max_i = (double)i;
            max_value = value;
        }
    }
    _w = max_i / (double)N * 2.0 * M_PI;
    _a = max_value / N * 2;
    if (_a > 1.045) _a = 1.045;
    else if (_a < 0.780) _a = 0.780;
}
void Wine::fitting_t(std::vector<Point9f> temp_target_infos)
{
    double max_value = 0.0, value = 0.0;
    int max_i = 0;
    for (int i = 0; i < T0_N + 1; i++)
    {
        value = get_integral((double)i * MAX_T0 / T0_N,temp_target_infos);
        if (value > max_value)
        {
            max_i = i;
            max_value = value;
        }
    }
    t_0 = (double)max_i * MAX_T0 / T0_N;
    start_time = temp_target_infos[0].current_time.x;
}
/**
 *  @brief  离散傅里叶获得正弦项值
 */
double Wine::get_F_s(int n, double f_k, int k, int _N)
{
    return f_k * sin(2.0 * M_PI * (double)n / (double)_N * (double)k * DT);
}

/**
 *  @brief  离散傅里叶获得余弦项值
 */
double Wine::get_F_c(int n, double f_k, int k, int _N)
{
    return f_k * cos(2.0 * M_PI * (double)n / (double)_N * (double)k * DT);
}
/**
 *  @brief 离散傅里叶获得第n项的值，规整化速度值
 *  @return 模的平方
 */
double Wine::get_F(int n, int _N,std::vector<Point9f> temp_target_infos)
{
    double c = 0.0, s = 0.0;
    if (wine_rotate>0)
        for (int i = 0; i < temp_target_infos.size(); i++)
        {
            c += get_F_c(n, (temp_target_infos[i].target_allangle.y - (2.090 - _a)), i, N);
            s += get_F_s(n, (temp_target_infos[i].target_allangle.y - (2.090 - _a)), i, N);
        }
    else
        for (int i = 0; i < temp_target_infos.size(); i++)
        {
            c += get_F_c(n, (-temp_target_infos[i].target_allangle.y - (2.090 - _a)), i, N);
            s += get_F_s(n, (-temp_target_infos[i].target_allangle.y - (2.090 - _a)), i, N);
        }

    return sqrt(c * c + s * s);
}

/**
 *  @brief  求不同相位时的积分,规整化速度值
 */
double Wine::get_integral(double t_,std::vector<Point9f> temp_target_infos)
{
    double sum = 0;
    if (wine_rotate>0)
        for (int i = 0; i < temp_target_infos.size(); i++)
        {
            sum += sin((i * DT + t_) * _w) * (temp_target_infos[i].target_allangle.y - (2.090 - _a)) / _a;
        }
    else
        for (int i = 0; i < temp_target_infos.size(); i++)
        {
            sum += sin((i * DT + t_) * _w) * (-temp_target_infos[i].target_allangle.y - (2.090 - _a)) / _a;
        }

    return sum;
}
void Wine::clearData(std::vector<Point9f> temp_target_infos)
{
    temp_target_infos.clear();
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