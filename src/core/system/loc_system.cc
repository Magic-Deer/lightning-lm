//
// Created by xiang on 25-9-12.
//

#include "core/system/loc_system.h"

#include <algorithm>
#include <iomanip>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "core/lightning_math.hpp"
#include "core/localization/localization.h"
#include "wrapper/ros_utils.h"

namespace lightning {

namespace {

std::string NormalizeFrameId(std::string frame_id) {
    while (!frame_id.empty() && frame_id.front() == '/') {
        frame_id.erase(frame_id.begin());
    }
    return frame_id;
}

std::string GetStringOr(const YAML::Node& node, const char* key, const std::string& fallback) {
    if (node && node[key]) {
        return node[key].as<std::string>();
    }
    return fallback;
}

std::string GetFrameOr(const YAML::Node& node, const char* key, const std::string& fallback) {
    return NormalizeFrameId(GetStringOr(node, key, fallback));
}

bool GetBoolOr(const YAML::Node& node, const char* key, bool fallback) {
    if (node && node[key]) {
        return node[key].as<bool>();
    }
    return fallback;
}

Vec3d GetVec3Or(const YAML::Node& node, const char* key, const Vec3d& fallback) {
    if (!(node && node[key])) {
        return fallback;
    }

    try {
        const auto values = node[key].as<std::vector<double>>();
        if (values.size() != 3) {
            LOG(WARNING) << "config " << key << " expects 3 values, got " << values.size() << ", using fallback";
            return fallback;
        }
        return Vec3d(values[0], values[1], values[2]);
    } catch (const YAML::Exception& e) {
        LOG(WARNING) << "failed to parse " << key << ": " << e.what() << ", using fallback";
        return fallback;
    }
}

Quatd GetQuatOr(const YAML::Node& node, const char* key, const Quatd& fallback) {
    if (!(node && node[key])) {
        return fallback;
    }

    try {
        const auto values = node[key].as<std::vector<double>>();
        if (values.size() != 4) {
            LOG(WARNING) << "config " << key << " expects 4 values, got " << values.size() << ", using fallback";
            return fallback;
        }

        Quatd q(values[3], values[0], values[1], values[2]);
        if (q.norm() < 1e-6) {
            LOG(WARNING) << "config " << key << " has near-zero quaternion, using fallback";
            return fallback;
        }
        q.normalize();
        return q;
    } catch (const YAML::Exception& e) {
        LOG(WARNING) << "failed to parse " << key << ": " << e.what() << ", using fallback";
        return fallback;
    }
}

SE3 GetSE3Or(const YAML::Node& node, const char* key, const SE3& fallback) {
    if (!(node && node[key])) {
        return fallback;
    }

    const auto se3_node = node[key];
    const Vec3d translation = GetVec3Or(se3_node, "translation", fallback.translation());
    const Quatd rotation = GetQuatOr(se3_node, "rotation_xyzw", fallback.unit_quaternion());
    return SE3(rotation, translation);
}

bool IsIdentity(const SE3& pose, double tolerance = 1e-9) {
    return pose.translation().norm() < tolerance && pose.so3().log().norm() < tolerance;
}

SE3 ComputeMapToOdom(const SE3& map_to_tracking, const SE3& odom_to_tracking) {
    return map_to_tracking * odom_to_tracking.inverse();
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

nav_msgs::msg::Odometry MakeOdometry(const SE3& pose, double timestamp, const Vec3d& linear_velocity,
                                     const Vec3d& angular_velocity, const std::string& odom_frame,
                                     const std::string& base_frame) {
    nav_msgs::msg::Odometry msg;
    msg.header.stamp = math::FromSec(timestamp);
    msg.header.frame_id = odom_frame;
    msg.child_frame_id = base_frame;
    msg.pose.pose.position.x = pose.translation().x();
    msg.pose.pose.position.y = pose.translation().y();
    msg.pose.pose.position.z = pose.translation().z();
    msg.pose.pose.orientation.x = pose.unit_quaternion().x();
    msg.pose.pose.orientation.y = pose.unit_quaternion().y();
    msg.pose.pose.orientation.z = pose.unit_quaternion().z();
    msg.pose.pose.orientation.w = pose.unit_quaternion().w();
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
    std::string map_path = yaml["system"]["map_path"].as<std::string>();

    LOG(INFO) << "online mode, creating ros2 node ... ";

    /// subscribers
    node_ = std::make_shared<rclcpp::Node>("lightning_loc");

    imu_topic_ = yaml["common"]["imu_topic"].as<std::string>();
    cloud_topic_ = yaml["common"]["lidar_topic"].as<std::string>();
    livox_topic_ = yaml["common"]["livox_lidar_topic"].as<std::string>();

    map_frame_ = GetFrameOr(ros_config, "map_frame", map_frame_);
    odom_frame_ = GetFrameOr(ros_config, "odom_frame", odom_frame_);
    base_frame_ = GetFrameOr(ros_config, "base_frame", base_frame_);
    tracking_frame_ = GetFrameOr(ros_config, "tracking_frame", base_frame_);
    odom_topic_ = GetStringOr(ros_config, "odom_topic", odom_topic_);
    initialpose_topic_ = GetStringOr(ros_config, "initialpose_topic", initialpose_topic_);
    status_topic_ = GetStringOr(ros_config, "status_topic", status_topic_);
    base_to_tracking_ = GetSE3Or(ros_config, "base_to_tracking", base_to_tracking_);
    options_.publish_global_tf_ = GetBoolOr(ros_config, "publish_global_tf", options_.publish_global_tf_);
    options_.publish_tracking_tf_ = GetBoolOr(ros_config, "publish_tracking_tf", options_.publish_tracking_tf_);
    options_.publish_odom_ = GetBoolOr(ros_config, "publish_odom", options_.publish_odom_);

    if (base_frame_ == tracking_frame_ && !IsIdentity(base_to_tracking_)) {
        LOG(WARNING) << "base_frame and tracking_frame are identical, ignoring non-identity base_to_tracking";
        base_to_tracking_ = SE3();
    }

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

    if (options_.publish_global_tf_) {
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
    }
    if (options_.publish_tracking_tf_ && base_frame_ != tracking_frame_) {
        static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node_);
    }

    loc_->SetLocalOdomCallback([this](const NavState& state) { HandleLocalOdom(state); });
    loc_->SetGlobalLocCallback([this](const loc::LocalizationResult& result) { HandleGlobalLoc(result); });
    loc_->SetLocStateCallback([this](const std_msgs::msg::Int32& state) { PublishLocState(state); });

    bool ret = loc_->Init(yaml_path, map_path);
    map_loaded_ = ret;
    if (ret) {
        if (static_tf_broadcaster_ != nullptr) {
            auto tracking_tf = MakeTransform(base_to_tracking_, 0.0, base_frame_, tracking_frame_);
            tracking_tf.header.stamp = node_->get_clock()->now();
            static_tf_broadcaster_->sendTransform(tracking_tf);
        }
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
    const std::string frame_id = NormalizeFrameId(pose_msg->header.frame_id);
    if (frame_id != map_frame_) {
        LOG(WARNING) << "ignore initial pose from frame " << pose_msg->header.frame_id
                     << ", expected " << map_frame_;
        return;
    }

    Eigen::Quaterniond q(pose_msg->pose.pose.orientation.w, pose_msg->pose.pose.orientation.x,
                         pose_msg->pose.pose.orientation.y, pose_msg->pose.pose.orientation.z);
    if (q.norm() < 1e-6) {
        q = Eigen::Quaterniond::Identity();
    } else {
        q.normalize();
    }

    const SE3 map_to_base(q, Vec3d(pose_msg->pose.pose.position.x, pose_msg->pose.pose.position.y,
                                   pose_msg->pose.pose.position.z));
    const SE3 map_to_tracking = map_to_base * base_to_tracking_;

    SetInitPose(map_to_tracking);

    SE3 map_to_odom;
    double tf_timestamp = 0.0;
    bool publish_global_tf = false;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        has_latest_map_to_odom_ = false;

        if (has_latest_local_odom_) {
            latest_map_to_odom_ = ComputeMapToOdom(map_to_tracking, latest_local_odom_state_.GetPose());
            has_latest_map_to_odom_ = true;
            has_pending_initial_pose_ = false;
            map_to_odom = latest_map_to_odom_;
            tf_timestamp = latest_local_odom_state_.timestamp_;
            publish_global_tf = true;
        } else {
            pending_initial_pose_map_to_tracking_ = map_to_tracking;
            has_pending_initial_pose_ = true;
        }
    }

    if (publish_global_tf) {
        LOG(INFO) << "applied initial pose immediately using local odom at t=" << std::fixed
                  << std::setprecision(12) << tf_timestamp;
        PublishGlobalTransform(map_to_odom, tf_timestamp);
    } else {
        LOG(INFO) << "received initial pose but local odom is not ready yet, waiting to update map->odom";
    }
}

void LocSystem::HandleLocalOdom(const NavState& state) {
    Vec3d angular_velocity_tracking = Vec3d::Zero();
    SE3 map_to_odom;
    bool publish_global_tf = false;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!local_odom_history_.empty() && state.timestamp_ < local_odom_history_.back().timestamp_) {
            LOG(WARNING) << "local odom time moved backwards, clearing history: " << state.timestamp_ << " < "
                         << local_odom_history_.back().timestamp_;
            local_odom_history_.clear();
        }

        latest_local_odom_state_ = state;
        has_latest_local_odom_ = true;
        local_odom_history_.push_back(state);
        while (local_odom_history_.size() > kMaxLocalOdomHistorySize) {
            local_odom_history_.pop_front();
        }

        if (has_angular_velocity_) {
            angular_velocity_tracking = latest_angular_velocity_ - state.Getbg();
        }

        if (has_pending_initial_pose_) {
            latest_map_to_odom_ = ComputeMapToOdom(pending_initial_pose_map_to_tracking_, state.GetPose());
            has_latest_map_to_odom_ = true;
            has_pending_initial_pose_ = false;
            map_to_odom = latest_map_to_odom_;
            publish_global_tf = true;
        } else if (has_latest_map_to_odom_) {
            map_to_odom = latest_map_to_odom_;
            publish_global_tf = true;
        }
    }

    const SE3 odom_to_tracking = state.GetPose();
    const SE3 odom_to_base = odom_to_tracking * base_to_tracking_.inverse();
    const Vec3d linear_velocity_tracking = state.GetRot().inverse() * state.GetVel();
    const Vec3d angular_velocity_base = base_to_tracking_.so3() * angular_velocity_tracking;
    // Apply the fixed rigid-body offset so /odom is expressed in base_frame, not tracking_frame.
    const Vec3d linear_velocity_base =
        base_to_tracking_.so3() * linear_velocity_tracking - angular_velocity_base.cross(base_to_tracking_.translation());

    if (options_.publish_odom_ && odom_pub_ != nullptr) {
        odom_pub_->publish(
            MakeOdometry(odom_to_base, state.timestamp_, linear_velocity_base, angular_velocity_base, odom_frame_,
                         base_frame_));
    }

    if (options_.publish_global_tf_ && tf_broadcaster_ != nullptr) {
        tf_broadcaster_->sendTransform(MakeTransform(odom_to_base, state.timestamp_, odom_frame_, base_frame_));
    }

    if (publish_global_tf) {
        PublishGlobalTransform(map_to_odom, state.timestamp_);
    }
}

void LocSystem::HandleGlobalLoc(const loc::LocalizationResult& result) {
    if (!result.valid_ || !options_.publish_global_tf_ || tf_broadcaster_ == nullptr) {
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

    SE3 odom_to_tracking;
    NavState best_match;
    if (!math::PoseInterp<NavState>(result.timestamp_, local_odom_history,
                                    [](const NavState& state) { return state.timestamp_; },
                                    [](const NavState& state) { return state.GetPose(); }, odom_to_tracking,
                                    best_match)) {
        LOG_EVERY_N(WARNING, 50) << "failed to align local odom with global loc at t=" << result.timestamp_;
        return;
    }
    (void)best_match;

    const SE3 map_to_odom = ComputeMapToOdom(result.pose_, odom_to_tracking);
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        latest_map_to_odom_ = map_to_odom;
        has_latest_map_to_odom_ = true;
    }
    PublishGlobalTransform(map_to_odom, result.timestamp_);
}

void LocSystem::PublishLocState(const std_msgs::msg::Int32& state) {
    if (status_pub_ != nullptr) {
        status_pub_->publish(state);
    }
}

void LocSystem::PublishGlobalTransform(const SE3& map_to_odom, double timestamp) {
    if (options_.publish_global_tf_ && tf_broadcaster_ != nullptr) {
        tf_broadcaster_->sendTransform(MakeTransform(map_to_odom, timestamp, map_frame_, odom_frame_));
    }
}

void LocSystem::Spin() {
    if (node_ != nullptr) {
        spin(node_);
    }
}

}  // namespace lightning
