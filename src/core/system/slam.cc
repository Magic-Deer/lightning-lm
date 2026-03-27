//
// Created by xiang on 25-5-6.
//

#include "core/system/slam.h"
#include "core/g2p5/g2p5.h"
#include "core/lio/laser_mapping.h"
#include "core/loop_closing/loop_closing.h"
#include "core/maps/tiled_map.h"
#include "ui/pangolin_window.h"
#include "wrapper/ros_utils.h"

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <yaml-cpp/yaml.h>

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

double GetDoubleOr(const YAML::Node& node, const char* key, double fallback) {
    if (node && node[key]) {
        return node[key].as<double>();
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

}  // namespace

SlamSystem::SlamSystem(lightning::SlamSystem::Options options) : options_(options) {
    /// handle ctrl-c
    signal(SIGINT, lightning::debug::SigHandle);
}

bool SlamSystem::Init(const std::string& yaml_path) {
    lio_ = std::make_shared<LaserMapping>();
    if (!lio_->Init(yaml_path)) {
        LOG(ERROR) << "failed to init lio module";
        return false;
    }

    auto yaml = YAML::LoadFile(yaml_path);
    const auto ros_config = yaml["ros"];
    options_.with_loop_closing_ = yaml["system"]["with_loop_closing"].as<bool>();
    options_.with_visualization_ = yaml["system"]["with_ui"].as<bool>();
    options_.with_2dvisualization_ = yaml["system"]["with_2dui"].as<bool>();
    options_.with_gridmap_ = yaml["system"]["with_g2p5"].as<bool>();
    options_.step_on_kf_ = yaml["system"]["step_on_kf"].as<bool>();
    map_topic_ = GetStringOr(ros_config, "map_topic", map_topic_);
    map_frame_ = GetFrameOr(ros_config, "map_frame", map_frame_);
    base_frame_ = GetFrameOr(ros_config, "base_frame", base_frame_);
    keyframe_cloud_topic_ = GetStringOr(ros_config, "keyframe_cloud_topic", keyframe_cloud_topic_);
    map_cloud_topic_ = GetStringOr(ros_config, "map_cloud_topic", map_cloud_topic_);
    map_cloud_voxel_size_ = GetDoubleOr(ros_config, "map_cloud_voxel_size", map_cloud_voxel_size_);
    publish_global_tf_ = GetBoolOr(ros_config, "publish_global_tf", publish_global_tf_);

    if (options_.with_loop_closing_) {
        LOG(INFO) << "slam with loop closing";
        LoopClosing::Options options;
        options.online_mode_ = options_.online_mode_;
        lc_ = std::make_shared<LoopClosing>(options);
        lc_->Init(yaml_path);
        lc_->SetLoopClosedCB([this]() {
            if (g2p5_ != nullptr) {
                g2p5_->RedrawGlobalMap();
            }
            RebuildMapCloud();
        });
    }

    if (options_.with_visualization_) {
        LOG(INFO) << "slam with 3D UI";
        ui_ = std::make_shared<ui::PangolinWindow>();
        ui_->Init();

        lio_->SetUI(ui_);
    }

    if (options_.with_gridmap_) {
        g2p5::G2P5::Options opt;
        opt.online_mode_ = options_.online_mode_;

        g2p5_ = std::make_shared<g2p5::G2P5>(opt);
        g2p5_->Init(yaml_path);

        g2p5_->SetMapUpdateCallback([this](g2p5::G2P5MapPtr map) { HandleMapUpdate(map); });
    }

    if (options_.online_mode_) {
        LOG(INFO) << "online mode, creating ros2 node ... ";

        /// subscribers
        node_ = std::make_shared<rclcpp::Node>("lightning_slam");

        if (publish_global_tf_) {
            tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
        }

        imu_topic_ = yaml["common"]["imu_topic"].as<std::string>();
        cloud_topic_ = yaml["common"]["lidar_topic"].as<std::string>();
        livox_topic_ = yaml["common"]["livox_lidar_topic"].as<std::string>();

        rclcpp::QoS qos(10);
        // qos.best_effort();

        imu_sub_ = node_->create_subscription<sensor_msgs::msg::Imu>(
            imu_topic_, qos, [this](sensor_msgs::msg::Imu::SharedPtr msg) {
                IMUPtr imu = std::make_shared<IMU>();
                imu->timestamp = ToSec(msg->header.stamp);
                imu->linear_acceleration =
                    Vec3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
                imu->angular_velocity =
                    Vec3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);

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

        if (options_.with_gridmap_) {
            rclcpp::QoS map_qos(rclcpp::KeepLast(1));
            map_qos.reliable();
            map_qos.transient_local();
            map_pub_ = node_->create_publisher<nav_msgs::msg::OccupancyGrid>(map_topic_, map_qos);
        }

        auto keyframe_cloud_qos = rclcpp::SensorDataQoS().keep_last(1);
        if (!keyframe_cloud_topic_.empty()) {
            keyframe_cloud_pub_ =
                node_->create_publisher<sensor_msgs::msg::PointCloud2>(keyframe_cloud_topic_, keyframe_cloud_qos);
        }

        rclcpp::QoS map_cloud_qos(rclcpp::KeepLast(1));
        map_cloud_qos.reliable();
        map_cloud_qos.transient_local();
        if (!map_cloud_topic_.empty()) {
            map_cloud_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>(map_cloud_topic_, map_cloud_qos);
        }

        savemap_service_ = node_->create_service<SaveMapService>(
            "lightning/save_map", [this](const SaveMapService::Request::SharedPtr& req,
                                         SaveMapService::Response::SharedPtr res) { SaveMap(req, res); });

        LOG(INFO) << "online slam node has been created.";
    }

    return true;
}

SlamSystem::~SlamSystem() {
    if (ui_) {
        ui_->Quit();
    }
}

void SlamSystem::StartSLAM(std::string map_name) {
    map_name_ = map_name;
    running_ = true;
}

void SlamSystem::SaveMap(const SaveMapService::Request::SharedPtr request,
                         SaveMapService::Response::SharedPtr response) {
    map_name_ = request->map_id;
    std::string save_path = "./data/" + map_name_ + "/";

    SaveMap(save_path);
    response->response = 0;
}

void SlamSystem::SaveMap(const std::string& path) {
    std::string save_path = path;
    if (save_path.empty()) {
        save_path = "./data/" + map_name_ + "/";
    }

    LOG(INFO) << "slam map saving to " << save_path;

    if (!std::filesystem::exists(save_path)) {
        std::filesystem::create_directories(save_path);
    } else {
        std::filesystem::remove_all(save_path);
        std::filesystem::create_directories(save_path);
    }

    // auto global_map_no_loop = lio_->GetGlobalMap(true);
    auto global_map = lio_->GetGlobalMap(!options_.with_loop_closing_);
    // auto global_map_raw = lio_->GetGlobalMap(!options_.with_loop_closing_, false, 0.1);

    TiledMap::Options tm_options;
    tm_options.map_path_ = save_path;

    TiledMap tm(tm_options);
    SE3 start_pose = lio_->GetAllKeyframes().front()->GetOptPose();
    tm.ConvertFromFullPCD(global_map, start_pose, save_path);

    pcl::io::savePCDFileBinaryCompressed(save_path + "/global.pcd", *global_map);
    // pcl::io::savePCDFileBinaryCompressed(save_path + "/global_no_loop.pcd", *global_map_no_loop);
    // pcl::io::savePCDFileBinaryCompressed(save_path + "/global_raw.pcd", *global_map_raw);

    if (options_.with_gridmap_) {
        /// 存为ROS兼容的模式
        auto map = g2p5_->GetNewestMap()->ToROS();
        const int width = map.info.width;
        const int height = map.info.height;

        cv::Mat nav_image(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            const int rowStartIndex = y * width;
            for (int x = 0; x < width; ++x) {
                const int index = rowStartIndex + x;
                int8_t data = map.data[index];
                if (data == 0) {                                   // Free
                    nav_image.at<uchar>(height - 1 - y, x) = 255;  // White
                } else if (data == 100) {                          // Occupied
                    nav_image.at<uchar>(height - 1 - y, x) = 0;    // Black
                } else {                                           // Unknown
                    nav_image.at<uchar>(height - 1 - y, x) = 128;  // Gray
                }
            }
        }

        cv::imwrite(save_path + "/map.pgm", nav_image);

        /// yaml
        std::ofstream yamlFile(save_path + "/map.yaml");
        if (!yamlFile.is_open()) {
            LOG(ERROR) << "failed to write map.yaml";
            return;  // 文件打开失败
        }

        try {
            YAML::Emitter emitter;
            emitter << YAML::BeginMap;
            emitter << YAML::Key << "image" << YAML::Value << "map.pgm";
            emitter << YAML::Key << "mode" << YAML::Value << "trinary";
            emitter << YAML::Key << "width" << YAML::Value << map.info.width;
            emitter << YAML::Key << "height" << YAML::Value << map.info.height;
            emitter << YAML::Key << "resolution" << YAML::Value << map.info.resolution;
            std::vector<double> orig{map.info.origin.position.x, map.info.origin.position.y, 0};
            emitter << YAML::Key << "origin" << YAML::Value << orig;
            emitter << YAML::Key << "negate" << YAML::Value << 0;
            emitter << YAML::Key << "occupied_thresh" << YAML::Value << 0.65;
            emitter << YAML::Key << "free_thresh" << YAML::Value << 0.25;

            emitter << YAML::EndMap;

            yamlFile << emitter.c_str();
            yamlFile.close();
        } catch (...) {
            yamlFile.close();
            return;
        }
    }

    LOG(INFO) << "map saved";
}

void SlamSystem::HandleMapUpdate(g2p5::G2P5MapPtr map) {
    if (map == nullptr) {
        return;
    }

    if (map_pub_ != nullptr && node_ != nullptr) {
        auto ros_map = map->ToROS();
        ros_map.header.stamp = node_->now();
        ros_map.header.frame_id = map_frame_;
        map_pub_->publish(ros_map);
    }

    if (options_.with_2dvisualization_) {
        cv::Mat image = map->ToCV();
        cv::imshow("map", image);

        if (options_.step_on_kf_) {
            cv::waitKey(0);
        } else {
            cv::waitKey(10);
        }
    }
}

void SlamSystem::PublishPoseTransform(const NavState& state) {
    if (!publish_global_tf_ || tf_broadcaster_ == nullptr || !state.pose_is_ok_) {
        return;
    }

    SE3 map_to_base = state.GetPose();
    if (cur_kf_ != nullptr) {
        map_to_base = cur_kf_->GetOptPose() * cur_kf_->GetLIOPose().inverse() * state.GetPose();
    }

    tf_broadcaster_->sendTransform(MakeTransform(map_to_base, state.timestamp_, map_frame_, base_frame_));
}

CloudPtr SlamSystem::TransformKeyframeCloud(const Keyframe::Ptr& keyframe) const {
    if (keyframe == nullptr) {
        return nullptr;
    }

    auto cloud = keyframe->GetCloud();
    if (cloud == nullptr || cloud->empty()) {
        return nullptr;
    }

    CloudPtr transformed(new PointCloudType());
    pcl::transformPointCloud(*cloud, *transformed, keyframe->GetOptPose().matrix());
    transformed->is_dense = false;
    transformed->height = 1;
    transformed->width = transformed->size();
    return transformed;
}

void SlamSystem::PublishCloud(const CloudPtr& cloud,
                              const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& publisher,
                              const builtin_interfaces::msg::Time& stamp) const {
    if (cloud == nullptr || cloud->empty() || publisher == nullptr) {
        return;
    }

    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*cloud, msg);
    msg.header.stamp = stamp;
    msg.header.frame_id = map_frame_;
    publisher->publish(msg);
}

void SlamSystem::PublishMapCloud() {
    if (node_ == nullptr || map_cloud_pub_ == nullptr) {
        return;
    }

    UL lock(map_cloud_mutex_);
    PublishCloud(map_cloud_, map_cloud_pub_, node_->now());
}

void SlamSystem::PublishKeyframeCloud(const Keyframe::Ptr& keyframe) {
    if (node_ == nullptr) {
        return;
    }

    auto transformed = TransformKeyframeCloud(keyframe);
    if (transformed == nullptr) {
        return;
    }

    const auto stamp = node_->now();
    PublishCloud(transformed, keyframe_cloud_pub_, stamp);

    if (map_cloud_pub_ == nullptr) {
        return;
    }

    UL lock(map_cloud_mutex_);
    *map_cloud_ += *transformed;

    if (map_cloud_voxel_size_ > 0.0 && !map_cloud_->empty()) {
        pcl::VoxelGrid<PointType> voxel;
        voxel.setLeafSize(map_cloud_voxel_size_, map_cloud_voxel_size_, map_cloud_voxel_size_);
        voxel.setInputCloud(map_cloud_);

        CloudPtr filtered(new PointCloudType());
        voxel.filter(*filtered);
        filtered->is_dense = false;
        filtered->height = 1;
        filtered->width = filtered->size();
        map_cloud_ = filtered;
    } else {
        map_cloud_->is_dense = false;
        map_cloud_->height = 1;
        map_cloud_->width = map_cloud_->size();
    }

    PublishCloud(map_cloud_, map_cloud_pub_, stamp);
}

void SlamSystem::RebuildMapCloud() {
    if (node_ == nullptr || lio_ == nullptr || map_cloud_pub_ == nullptr) {
        return;
    }

    auto keyframes = lio_->GetAllKeyframes();
    CloudPtr rebuilt(new PointCloudType());

    for (const auto& keyframe : keyframes) {
        auto transformed = TransformKeyframeCloud(keyframe);
        if (transformed == nullptr) {
            continue;
        }
        *rebuilt += *transformed;
    }

    if (map_cloud_voxel_size_ > 0.0 && !rebuilt->empty()) {
        pcl::VoxelGrid<PointType> voxel;
        voxel.setLeafSize(map_cloud_voxel_size_, map_cloud_voxel_size_, map_cloud_voxel_size_);
        voxel.setInputCloud(rebuilt);

        CloudPtr filtered(new PointCloudType());
        voxel.filter(*filtered);
        filtered->is_dense = false;
        filtered->height = 1;
        filtered->width = filtered->size();
        rebuilt = filtered;
    } else {
        rebuilt->is_dense = false;
        rebuilt->height = 1;
        rebuilt->width = rebuilt->size();
    }

    {
        UL lock(map_cloud_mutex_);
        map_cloud_ = rebuilt;
    }

    PublishMapCloud();
}

void SlamSystem::ProcessIMU(const lightning::IMUPtr& imu) {
    if (running_ == false) {
        return;
    }
    lio_->ProcessIMU(imu);
}

void SlamSystem::ProcessLidar(const sensor_msgs::msg::PointCloud2::SharedPtr& cloud) {
    if (running_ == false) {
        return;
    }

    lio_->ProcessPointCloud2(cloud);
    lio_->Run();
    PublishPoseTransform(lio_->GetState());

    auto kf = lio_->GetKeyframe();
    if (kf != cur_kf_) {
        cur_kf_ = kf;
    } else {
        return;
    }

    if (cur_kf_ == nullptr) {
        return;
    }

    if (options_.with_loop_closing_) {
        lc_->AddKF(cur_kf_);
    }

    if (options_.with_gridmap_) {
        g2p5_->PushKeyframe(cur_kf_);
    }

    PublishKeyframeCloud(cur_kf_);

    if (ui_) {
        ui_->UpdateKF(cur_kf_);
    }
}

void SlamSystem::ProcessLidar(const livox_ros_driver2::msg::CustomMsg::SharedPtr& cloud) {
    if (running_ == false) {
        return;
    }

    lio_->ProcessPointCloud2(cloud);
    lio_->Run();
    PublishPoseTransform(lio_->GetState());

    auto kf = lio_->GetKeyframe();
    if (kf != cur_kf_) {
        cur_kf_ = kf;
    } else {
        return;
    }

    if (cur_kf_ == nullptr) {
        return;
    }

    if (options_.with_loop_closing_) {
        lc_->AddKF(cur_kf_);
    }

    if (options_.with_gridmap_) {
        g2p5_->PushKeyframe(cur_kf_);
    }

    PublishKeyframeCloud(cur_kf_);

    if (ui_) {
        ui_->UpdateKF(cur_kf_);
    }
}

void SlamSystem::Spin() {
    if (options_.online_mode_ && node_ != nullptr) {
        spin(node_);
    }
}

}  // namespace lightning
