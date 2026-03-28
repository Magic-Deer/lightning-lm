//
// Created by xiang on 25-9-8.
//

#ifndef LIGHTNING_LOC_SYSTEM_H
#define LIGHTNING_LOC_SYSTEM_H

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/int32.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>

#include <atomic>
#include <deque>
#include <mutex>

#include "livox_ros_driver2/msg/custom_msg.hpp"

#include "common/eigen_types.h"
#include "common/imu.h"
#include "common/keyframe.h"

namespace lightning {

namespace loc {
class Localization;
struct LocalizationResult;
}

class LocSystem {
   public:
    struct Options {
        bool publish_global_tf_ = true;  // 是否发布 map -> odom 和 odom -> base
        bool publish_imu_tf_ = true;     // 是否发布 base -> imu
        bool publish_odom_ = true;       // 是否发布odom
    };

    explicit LocSystem(Options options);
    ~LocSystem();

    /// 初始化，地图路径在yaml里配置
    bool Init(const std::string& yaml_path);

    /// 设置初始化位姿
    void SetInitPose(const SE3& pose);

    /// 处理IMU
    void ProcessIMU(const lightning::IMUPtr& imu);

    /// 处理点云
    void ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud);
    void ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud);

    /// 实时模式下的spin
    void Spin();

   private:
    void HandleInitialPose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr& pose_msg);
    void HandleLocalOdom(const NavState& state);
    void HandleGlobalLoc(const loc::LocalizationResult& result);
    void PublishLocState(const std_msgs::msg::Int32& state);
    void PublishGlobalTransform(const SE3& map_to_odom, double timestamp);

    Options options_;

    std::shared_ptr<loc::Localization> loc_ = nullptr;  // 定位接口

    std::atomic_bool map_loaded_ = false;   // 地图是否已载入

    /// 实时模式下的ros2 node, subscribers
    rclcpp::Node::SharedPtr node_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_ = nullptr;
    std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_ = nullptr;

    std::string map_frame_ = "map";
    std::string odom_frame_ = "odom";
    std::string base_frame_ = "base_link";
    std::string imu_frame_ = "base_link";
    SE3 base_to_imu_ = SE3();

    std::string imu_topic_;
    std::string cloud_topic_;
    std::string livox_topic_;
    std::string odom_topic_ = "/odom";
    std::string initialpose_topic_ = "/initialpose";
    std::string status_topic_ = "/lightning/status";

    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_ = nullptr;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_ = nullptr;
    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr livox_sub_ = nullptr;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initialpose_sub_ = nullptr;

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_ = nullptr;
    rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr status_pub_ = nullptr;

    static constexpr size_t kMaxLocalOdomHistorySize = 200;

    std::mutex state_mutex_;
    std::deque<NavState> local_odom_history_;
    NavState latest_local_odom_state_;
    bool has_latest_local_odom_ = false;
    SE3 latest_map_to_odom_ = SE3();
    bool has_latest_map_to_odom_ = false;
    SE3 pending_initial_pose_map_to_imu_ = SE3();
    bool has_pending_initial_pose_ = false;
    Vec3d latest_angular_velocity_ = Vec3d::Zero();
    bool has_angular_velocity_ = false;
    std::atomic_bool warned_imu_frame_mismatch_ = false;
};

};  // namespace lightning

#endif  // LIGHTNING_LOC_SYSTEM_H
