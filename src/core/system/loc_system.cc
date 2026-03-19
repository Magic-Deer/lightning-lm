//
// Created by xiang on 25-9-12.
//

#include "core/system/loc_system.h"

#include <algorithm>
#include <yaml-cpp/yaml.h>

#include "core/lightning_math.hpp"
#include "core/localization/localization.h"
#include "wrapper/ros_utils.h"

namespace lightning {

namespace {

std::string GetStringOr(const YAML::Node& node, const char* key, const std::string& fallback) {
    if (node && node[key]) {
        return node[key].as<std::string>();
    }
    return fallback;
}

bool GetBoolOr(const YAML::Node& node, const char* key, bool fallback) {
    if (node && node[key]) {
        return node[key].as<bool>();
    }
    return fallback;
}

geometry_msgs::msg::TransformStamped MakeTransform(const SE3& pose, double timestamp, const std::string& parent_frame,
                                                   const std::string& child_frame) {
    geometry_msgs::msg::TransformStamped msg;
    msg.header.stamp = math::FromSec(timestamp);
    msg.header.frame_id = parent_frame;
    msg.child_frame_id = child_frame;
    msg.transform.translation.x = pose.translation().x();
    msg.transform.translation.y = pose.translation().y();
    msg.transform.translation.z = pose.translation().z();
    msg.transform.rotation.x = pose.unit_quaternion().x();
    msg.transform.rotation.y = pose.unit_quaternion().y();
    msg.transform.rotation.z = pose.unit_quaternion().z();
    msg.transform.rotation.w = pose.unit_quaternion().w();
    return msg;
}

nav_msgs::msg::Odometry MakeOdometry(const NavState& state, const Vec3d& angular_velocity, const std::string& odom_frame,
                                     const std::string& base_frame) {
    nav_msgs::msg::Odometry msg;
    msg.header.stamp = math::FromSec(state.timestamp_);
    msg.header.frame_id = odom_frame;
    msg.child_frame_id = base_frame;
    msg.pose.pose.position.x = state.pos_.x();
    msg.pose.pose.position.y = state.pos_.y();
    msg.pose.pose.position.z = state.pos_.z();
    msg.pose.pose.orientation.x = state.rot_.unit_quaternion().x();
    msg.pose.pose.orientation.y = state.rot_.unit_quaternion().y();
    msg.pose.pose.orientation.z = state.rot_.unit_quaternion().z();
    msg.pose.pose.orientation.w = state.rot_.unit_quaternion().w();

    const Vec3d linear_velocity = state.GetRot().inverse() * state.GetVel();
    msg.twist.twist.linear.x = linear_velocity.x();
    msg.twist.twist.linear.y = linear_velocity.y();
    msg.twist.twist.linear.z = linear_velocity.z();
    msg.twist.twist.angular.x = angular_velocity.x();
    msg.twist.twist.angular.y = angular_velocity.y();
    msg.twist.twist.angular.z = angular_velocity.z();
    return msg;
}

}  // namespace

LocSystem::LocSystem(LocSystem::Options options) : options_(options) {
    /// handle ctrl-c
    signal(SIGINT, lightning::debug::SigHandle);
}

LocSystem::~LocSystem() {
    if (loc_ != nullptr) {
        loc_->Finish();
    }
}

bool LocSystem::Init(const std::string &yaml_path) {
    loc::Localization::Options opt;
    opt.online_mode_ = true;
    loc_ = std::make_shared<loc::Localization>(opt);

    auto yaml = YAML::LoadFile(yaml_path);
    const auto ros_config = yaml["ros"];
    const auto system_config = yaml["system"];

    std::string map_path = yaml["system"]["map_path"].as<std::string>();

    LOG(INFO) << "online mode, creating ros2 node ... ";

    /// subscribers
    node_ = std::make_shared<rclcpp::Node>("lightning_loc");

    imu_topic_ = yaml["common"]["imu_topic"].as<std::string>();
    cloud_topic_ = yaml["common"]["lidar_topic"].as<std::string>();
    livox_topic_ = yaml["common"]["livox_lidar_topic"].as<std::string>();

    map_frame_ = GetStringOr(ros_config, "map_frame", map_frame_);
    odom_frame_ = GetStringOr(ros_config, "odom_frame", odom_frame_);
    base_frame_ = GetStringOr(ros_config, "base_frame", base_frame_);
    odom_topic_ = GetStringOr(ros_config, "odom_topic", odom_topic_);
    initialpose_topic_ = GetStringOr(ros_config, "initialpose_topic", initialpose_topic_);
    status_topic_ = GetStringOr(ros_config, "status_topic", status_topic_);
    options_.pub_tf_ = GetBoolOr(ros_config, "publish_tf",
                                 system_config && system_config["pub_tf"] ? system_config["pub_tf"].as<bool>()
                                                                          : options_.pub_tf_);
    options_.publish_odom_ = GetBoolOr(ros_config, "publish_odom", options_.publish_odom_);

    rclcpp::QoS qos(10);

    if (options_.publish_odom_) {
        odom_pub_ = node_->create_publisher<nav_msgs::msg::Odometry>(odom_topic_, 10);
    }

    status_pub_ = node_->create_publisher<std_msgs::msg::Int32>(status_topic_, 10);

    imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic_, qos, [this](sensor_msgs::msg::Imu::SharedPtr msg) {
            IMUPtr imu = std::make_shared<IMU>();
            imu->timestamp = ToSec(msg->header.stamp);
            imu->linear_acceleration =
                Vec3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
            imu->angular_velocity = Vec3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

            ProcessIMU(imu);
        });

    cloud_sub_ = node_->create_subscription<sensor_msgs::msg::PointCloud2>(
        cloud_topic_, qos, [this](sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
            Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
        });

    livox_sub_ = node_->create_subscription<livox_ros_driver2::msg::CustomMsg>(
        livox_topic_, qos, [this](livox_ros_driver2::msg::CustomMsg ::SharedPtr cloud) {
            Timer::Evaluate([&]() { ProcessLidar(cloud); }, "Proc Lidar", true);
        });

    initialpose_sub_ = node_->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        initialpose_topic_, 10, [this](geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr pose_msg) {
            HandleInitialPose(pose_msg);
        });

    if (options_.pub_tf_) {
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
    }

    loc_->SetLocalOdomCallback([this](const NavState& state) { HandleLocalOdom(state); });
    loc_->SetGlobalLocCallback([this](const loc::LocalizationResult& result) { HandleGlobalLoc(result); });
    loc_->SetLocStateCallback([this](const std_msgs::msg::Int32& state) { PublishLocState(state); });

    bool ret = loc_->Init(yaml_path, map_path);
    map_loaded_ = ret;
    if (ret) {
        LOG(INFO) << "online loc node has been created.";
    }

    return ret;
}

void LocSystem::SetInitPose(const SE3 &pose) {
    LOG(INFO) << "set init pose: " << pose.translation().transpose() << ", "
              << pose.unit_quaternion().coeffs().transpose();

    if (loc_ != nullptr) {
        loc_->SetExternalPose(pose.unit_quaternion(), pose.translation());
    }
}

void LocSystem::ProcessIMU(const IMUPtr &imu) {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        latest_angular_velocity_ = imu->angular_velocity;
        has_angular_velocity_ = true;
    }

    if (map_loaded_ && loc_ != nullptr) {
        loc_->ProcessIMUMsg(imu);
    }
}

void LocSystem::ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr &cloud) {
    if (map_loaded_ && loc_ != nullptr) {
        loc_->ProcessLidarMsg(cloud);
    }
}

void LocSystem::ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr &cloud) {
    if (map_loaded_ && loc_ != nullptr) {
        loc_->ProcessLivoxLidarMsg(cloud);
    }
}

void LocSystem::HandleInitialPose(const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr& pose_msg) {
    Eigen::Quaterniond q(pose_msg->pose.pose.orientation.w, pose_msg->pose.pose.orientation.x,
                         pose_msg->pose.pose.orientation.y, pose_msg->pose.pose.orientation.z);
    if (q.norm() < 1e-6) {
        q = Eigen::Quaterniond::Identity();
    } else {
        q.normalize();
    }

    const Vec3d t(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y, pose_msg->pose.pose.position.z);
    SetInitPose(SE3(q, t));
}

void LocSystem::HandleLocalOdom(const NavState& state) {
    Vec3d angular_velocity = Vec3d::Zero();
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!local_odom_history_.empty() && state.timestamp_ < local_odom_history_.back().timestamp_) {
            LOG(WARNING) << "local odom time moved backwards, clearing history: " << state.timestamp_ << " < "
                         << local_odom_history_.back().timestamp_;
            local_odom_history_.clear();
        }

        local_odom_history_.push_back(state);
        while (local_odom_history_.size() > kMaxLocalOdomHistorySize) {
            local_odom_history_.pop_front();
        }

        if (has_angular_velocity_) {
            angular_velocity = latest_angular_velocity_ - state.Getbg();
        }
    }

    if (options_.publish_odom_ && odom_pub_ != nullptr) {
        odom_pub_->publish(MakeOdometry(state, angular_velocity, odom_frame_, base_frame_));
    }

    if (options_.pub_tf_ && tf_broadcaster_ != nullptr) {
        tf_broadcaster_->sendTransform(MakeTransform(state.GetPose(), state.timestamp_, odom_frame_, base_frame_));
    }
}

void LocSystem::HandleGlobalLoc(const loc::LocalizationResult& result) {
    if (!result.valid_ || !options_.pub_tf_ || tf_broadcaster_ == nullptr) {
        return;
    }

    std::deque<NavState> local_odom_history;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        local_odom_history = local_odom_history_;
    }

    if (local_odom_history.size() < 2) {
        return;
    }

    SE3 odom_pose;
    NavState best_match;
    if (!math::PoseInterp<NavState>(result.timestamp_, local_odom_history,
                                    [](const NavState& state) { return state.timestamp_; },
                                    [](const NavState& state) { return state.GetPose(); }, odom_pose, best_match)) {
        LOG_EVERY_N(WARNING, 50) << "failed to align local odom with global loc at t=" << result.timestamp_;
        return;
    }
    (void)best_match;

    const SE3 map_to_odom = result.pose_ * odom_pose.inverse();
    tf_broadcaster_->sendTransform(MakeTransform(map_to_odom, result.timestamp_, map_frame_, odom_frame_));
}

void LocSystem::PublishLocState(const std_msgs::msg::Int32& state) {
    if (status_pub_ != nullptr) {
        status_pub_->publish(state);
    }
}

void LocSystem::Spin() {
    if (node_ != nullptr) {
        spin(node_);
    }
}

}  // namespace lightning
