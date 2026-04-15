#include <pcl/common/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include "core/localization/lidar_loc/lidar_loc.h"
#include "core/localization/localization.h"
#include "core/localization/pose_graph/pgo.h"
#include "io/yaml_io.h"
#include "ui/pangolin_window.h"

namespace lightning::loc {

// ！ 构造函数
Localization::Localization(Options options) { options_ = options; }

// ！初始化函数
bool Localization::Init(const std::string& yaml_path, const std::string& global_map_path) {
    UL lock(global_mutex_);
    if (lidar_loc_ != nullptr) {
        // 若已经启动，则变为初始化
        Finish();
    }

    YAML_IO yaml(yaml_path);
    options_.with_ui_ = yaml.GetValue<bool>("system", "with_ui");

    /// lidar odom前端
    LaserMapping::Options opt_lio;
    opt_lio.is_in_slam_mode_ = false;

    lio_ = std::make_shared<LaserMapping>(opt_lio);
    if (!lio_->Init(yaml_path)) {
        LOG(ERROR) << "failed to init lio";
        return false;
    }

    /// 激光定位
    LidarLoc::Options lidar_loc_options;
    lidar_loc_options.update_dynamic_cloud_ = yaml.GetValue<bool>("lidar_loc", "update_dynamic_cloud");
    lidar_loc_options.force_2d_ = yaml.GetValue<bool>("lidar_loc", "force_2d");
    lidar_loc_options.map_option_.enable_dynamic_polygon_ = false;
    lidar_loc_options.map_option_.map_path_ = global_map_path;
    lidar_loc_ = std::make_shared<LidarLoc>(lidar_loc_options);

    if (options_.with_ui_) {
        ui_ = std::make_shared<ui::PangolinWindow>();
        ui_->SetCurrentScanSize(10);
        ui_->Init();

        lidar_loc_->SetUI(ui_);

        // lio_->SetUI(ui_);
    }

    lidar_loc_->Init(yaml_path);

    /// pose graph
    pgo_ = std::make_shared<PGO>();
    pgo_->SetDebug(false);

    ///  各模块的异步调用
    options_.enable_lidar_loc_skip_ = yaml.GetValue<bool>("system", "enable_lidar_loc_skip");
    options_.enable_lidar_loc_rviz_ = yaml.GetValue<bool>("system", "enable_lidar_loc_rviz");
    options_.lidar_loc_skip_num_ = yaml.GetValue<int>("system", "lidar_loc_skip_num");
    options_.enable_lidar_odom_skip_ = yaml.GetValue<bool>("system", "enable_lidar_odom_skip");
    options_.lidar_odom_skip_num_ = yaml.GetValue<int>("system", "lidar_odom_skip_num");
    options_.loc_on_kf_ = yaml.GetValue<bool>("lidar_loc", "loc_on_kf");

    lidar_odom_proc_cloud_.SetMaxSize(1);
    lidar_loc_proc_cloud_.SetMaxSize(1);

    lidar_odom_proc_cloud_.SetName("激光里程计");
    lidar_loc_proc_cloud_.SetName("激光定位");

    // 允许跳帧
    lidar_loc_proc_cloud_.SetSkipParam(options_.enable_lidar_loc_skip_, options_.lidar_loc_skip_num_);
    lidar_odom_proc_cloud_.SetSkipParam(options_.enable_lidar_odom_skip_, options_.lidar_odom_skip_num_);

    lidar_odom_proc_cloud_.SetProcFunc([this](const QueuedCloud& cloud) { LidarOdomProcCloud(cloud); });
    lidar_loc_proc_cloud_.SetProcFunc([this](const QueuedCloud& cloud) { LidarLocProcCloud(cloud); });

    if (options_.online_mode_) {
        lidar_odom_proc_cloud_.Start();
        lidar_loc_proc_cloud_.Start();
    }

    {
        std::lock_guard<std::mutex> lock(pending_external_pose_mutex_);
        pending_external_pose_ = PendingExternalPose();
        next_cloud_seq_ = 0;
    }

    /// TODO: 发布
    pgo_->SetHighFrequencyGlobalOutputHandleFunction([this](const LocalizationResult& res) {
        // if (loc_result_.timestamp_ > 0) {
        //             double loc_fps = 1.0 / (res.timestamp_ - loc_result_.timestamp_);
        //             // LOG_EVERY_N(INFO, 10) << "loc fps: " << loc_fps;
        //         }

        loc_result_ = res;

        if (HasPendingExternalPose()) {
            return;
        }

        if (global_loc_callback_ && loc_result_.valid_) {
            global_loc_callback_(loc_result_);
        }

        if (ui_) {
            ui_->UpdateNavState(loc_result_.ToNavState());
            ui_->UpdateRecentPose(loc_result_.pose_);
        }
    });

    /// 预处理器
    preprocess_.reset(new PointCloudPreprocess());
    preprocess_->Blind() = yaml.GetValue<double>("fasterlio", "blind");
    preprocess_->TimeScale() = yaml.GetValue<double>("fasterlio", "time_scale");
    int lidar_type = yaml.GetValue<int>("fasterlio", "lidar_type");
    preprocess_->NumScans() = yaml.GetValue<int>("fasterlio", "scan_line");
    preprocess_->PointFilterNum() = yaml.GetValue<int>("fasterlio", "point_filter_num");

    LOG(INFO) << "lidar_type " << lidar_type;
    if (lidar_type == 1) {
        preprocess_->SetLidarType(LidarType::AVIA);
        LOG(INFO) << "Using AVIA Lidar";
    } else if (lidar_type == 2) {
        preprocess_->SetLidarType(LidarType::VELO32);
        LOG(INFO) << "Using Velodyne 32 Lidar";
    } else if (lidar_type == 3) {
        preprocess_->SetLidarType(LidarType::OUST64);
        LOG(INFO) << "Using OUST 64 Lidar";
    } else {
        LOG(WARNING) << "unknown lidar_type";
    }

    return true;
}

void Localization::ProcessLidarMsg(const sensor_msgs::msg::PointCloud2::SharedPtr cloud) {
    UL lock(global_mutex_);
    if (lidar_loc_ == nullptr || lio_ == nullptr || pgo_ == nullptr) {
        return;
    }

    // 串行模式
    CloudPtr laser_cloud(new PointCloudType);
    preprocess_->Process(cloud, laser_cloud);
    laser_cloud->header.stamp = cloud->header.stamp.sec * 1e9 + cloud->header.stamp.nanosec;
    QueuedCloud queued_cloud;
    queued_cloud.cloud = laser_cloud;
    queued_cloud.seq = next_cloud_seq_++;

    if (options_.online_mode_) {
        lidar_odom_proc_cloud_.AddMessage(queued_cloud);
    } else {
        LidarOdomProcCloud(queued_cloud);
    }
}

void Localization::ProcessLivoxLidarMsg(const livox_ros_driver2::msg::CustomMsg::SharedPtr cloud) {
    UL lock(global_mutex_);
    if (lidar_loc_ == nullptr || lio_ == nullptr || pgo_ == nullptr) {
        return;
    }

    // 串行模式
    CloudPtr laser_cloud(new PointCloudType);
    preprocess_->Process(cloud, laser_cloud);
    laser_cloud->header.stamp = cloud->header.stamp.sec * 1e9 + cloud->header.stamp.nanosec;
    QueuedCloud queued_cloud;
    queued_cloud.cloud = laser_cloud;
    queued_cloud.seq = next_cloud_seq_++;

    if (options_.online_mode_) {
        lidar_odom_proc_cloud_.AddMessage(queued_cloud);
    } else {
        LidarOdomProcCloud(queued_cloud);
    }
}

void Localization::LidarOdomProcCloud(QueuedCloud queued_cloud) {
    if (lio_ == nullptr || queued_cloud.cloud == nullptr) {
        return;
    }

    /// NOTE: 在NCLT这种数据集中，lio内部是有缓存的，它拿到的点云不一定是最新时刻的点云
    lio_->ProcessPointCloud2(queued_cloud.cloud);
    if (!lio_->Run()) {
        return;
    }

    auto lo_state = lio_->GetState();
    auto continuous_local_odom_state = lio_->GetIMUState();

    if (continuous_local_odom_callback_ && continuous_local_odom_state.pose_is_ok_) {
        continuous_local_odom_callback_(continuous_local_odom_state);
    }

    lidar_loc_->ProcessLO(lo_state);
    pgo_->ProcessLidarOdom(lo_state);

    // LOG(INFO) << "LO pose: " << std::setprecision(12) << lo_state.timestamp_ << " "
    //           << lo_state.GetPose().translation().transpose();

    /// 获得lio的关键帧
    if (options_.loc_on_kf_) {
        auto kf = lio_->GetKeyframe();
        if (kf == lio_kf_) {
            /// 关键帧未更新，那就只更新IMU状态

            // auto dr_state = lio_->GetState();
            // lidar_loc_->ProcessDR(dr_state);
            // pgo_->ProcessDR(dr_state);
            return;
        }

        lio_kf_ = kf;

        auto scan = lio_->GetScanUndist();
        QueuedCloud loc_cloud;
        loc_cloud.cloud = scan;
        loc_cloud.seq = queued_cloud.seq;
        loc_cloud.lo_state = lo_state;
        loc_cloud.has_lo_state = true;

        if (options_.online_mode_) {
            lidar_loc_proc_cloud_.AddMessage(loc_cloud);
        } else {
            LidarLocProcCloud(loc_cloud);
        }
    } else {
        auto scan = lio_->GetScanUndist();
        QueuedCloud loc_cloud;
        loc_cloud.cloud = scan;
        loc_cloud.seq = queued_cloud.seq;
        loc_cloud.lo_state = lo_state;
        loc_cloud.has_lo_state = true;

        if (options_.online_mode_) {
            lidar_loc_proc_cloud_.AddMessage(loc_cloud);
        } else {
            LidarLocProcCloud(loc_cloud);
        }
    }
}

void Localization::LidarLocProcCloud(QueuedCloud queued_cloud) {
    if (lidar_loc_ == nullptr || pgo_ == nullptr || queued_cloud.cloud == nullptr) {
        return;
    }

    SE3 external_pose_to_apply;
    bool applied_external_pose = false;
    switch (GetExternalPoseActionForScan(queued_cloud.seq, &external_pose_to_apply)) {
        case ExternalPoseAction::kDropScan:
            LOG(INFO) << "drop stale loc scan before external reloc, seq: " << queued_cloud.seq;
            return;
        case ExternalPoseAction::kApplyAndProcessScan:
            LOG(INFO) << "external reloc start, seq: " << queued_cloud.seq
                      << ", seed: " << external_pose_to_apply.translation().transpose();
            pgo_->Reset();
            loc_result_ = LocalizationResult();
            lidar_loc_->SetInitialPose(external_pose_to_apply);
            applied_external_pose = true;
            break;
        case ExternalPoseAction::kProcessScan:
            break;
    }

    lidar_loc_->ProcessCloud(queued_cloud.cloud);

    if (ShouldDropLocResult(queued_cloud.seq)) {
        LOG(INFO) << "discard loc result due to newer external reloc request, seq: " << queued_cloud.seq;
        return;
    }

    auto res = lidar_loc_->GetLocalizationResult();
    if (applied_external_pose && queued_cloud.has_lo_state) {
        res.rel_pose_set_ = true;
        res.rel_pose_ = queued_cloud.lo_state.GetPose();
        res.vel_b_ = queued_cloud.lo_state.GetRot().matrix().transpose() * queued_cloud.lo_state.GetVel();
    }
    pgo_->ProcessLidarLoc(res);

    if (ui_) {
        // Twi with Til, here pose means Twl, thus Til=I
        ui_->UpdateScan(queued_cloud.cloud, res.pose_);
    }

    if (loc_state_callback_) {
        auto loc_state = std::make_shared<std_msgs::msg::Int32>();
        loc_state->data = static_cast<int>(res.status_);
        LOG(INFO) << "loc_state: " << loc_state->data;
        loc_state_callback_(*loc_state);
    }
}

void Localization::ProcessIMUMsg(IMUPtr imu) {
    UL lock(global_mutex_);

    if (lidar_loc_ == nullptr || lio_ == nullptr || pgo_ == nullptr) {
        return;
    }

    double this_imu_time = imu->timestamp;
    if (last_imu_time_ > 0 && this_imu_time < last_imu_time_) {
        LOG(WARNING) << "IMU 时间异常：" << this_imu_time << ", last: " << last_imu_time_;
    }
    last_imu_time_ = this_imu_time;

    /// 里程计处理IMU
    lio_->ProcessIMU(imu);

    /// 这里需要 IMU predict，否则没法process DR了
    auto dr_state = lio_->GetIMUState();

    if (!dr_state.pose_is_ok_) {
        return;
    }

    if (continuous_local_odom_callback_) {
        continuous_local_odom_callback_(dr_state);
    }

    // /// 停车判定
    // constexpr auto kThVbrbStill = 0.05;  // 0.08;
    // constexpr auto kThOmegaStill = 0.05;

    // if (dr_state.GetVel().norm() < kThVbrbStill && imu->angular_velocity.norm() < kThOmegaStill) {
    //     dr_state.is_parking_ = true;
    //     dr_state.SetVel(Vec3d::Zero());
    // }

    /// 如果没有odm, 用lio替代DR

    // LOG(INFO) << "dr state: " << std::setprecision(12) << dr_state.timestamp_ << " "
    //           << dr_state.GetPose().translation().transpose()
    //           << ", q=" << dr_state.GetPose().unit_quaternion().coeffs().transpose();

    lidar_loc_->ProcessDR(dr_state);
    pgo_->ProcessDR(dr_state);
}

// void Localization::ProcessOdomMsg(const nav_msgs::msg::Odometry::SharedPtr odom_msg) {
//     UL lock(global_mutex_);
//
//     if (lidar_loc_ == nullptr || lio_ == nullptr || pgo_ == nullptr) {
//         return;
//     }
//     double this_odom_time = ToSec(odom_msg->header.stamp);
//     if (last_odom_time_ > 0 && this_odom_time < last_odom_time_) {
//         LOG(WARNING) << "Odom Time Abnormal:" << this_odom_time << ", last: " << last_odom_time_;
//     }
//     last_odom_time_ = this_odom_time;
//
//     lio_->ProcessOdometry(odom_msg);
//
//     if (!lio_->GetbOdomHF()) {
//         return;
//     }
//
//     auto dr_state = lio_->GetStateHF(mapping::FasterLioMapping::kHFStateOdomFiltered);
//
//     constexpr auto kThVbrbStill = 0.03;  // 0.08;
//     constexpr auto kThOmegaStill = 0.03;
//     if (dr_state.Getvwi().norm() < kThVbrbStill && dr_state.Getwii().norm() < kThOmegaStill) {
//         dr_state.is_parking_ = true;
//         dr_state.Setvwi(Vec3d::Zero());
//         dr_state.Setwii(Vec3d::Zero());
//     }
//
//     lidar_loc_->ProcessDR(dr_state);
//     pgo_->ProcessDR(dr_state);
// }

void Localization::Finish() {
    lidar_loc_->Finish();
    if (ui_) {
        ui_->Quit();
    }

    lidar_loc_proc_cloud_.Quit();
    lidar_odom_proc_cloud_.Quit();
}

void Localization::SetExternalPose(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) {
    UL lock(global_mutex_);

    if (lidar_loc_ == nullptr || lio_ == nullptr || pgo_ == nullptr) {
        return;
    }

    {
        std::lock_guard<std::mutex> external_lock(pending_external_pose_mutex_);
        pending_external_pose_.active = true;
        pending_external_pose_.pose = SE3(q, t);
        pending_external_pose_.min_seq = next_cloud_seq_;
        LOG(INFO) << "queue external reloc request, min scan seq: " << pending_external_pose_.min_seq
                  << ", seed: " << t.transpose();
    }

    lidar_loc_proc_cloud_.ClearPending();
    loc_result_ = LocalizationResult();
}

Localization::ExternalPoseAction Localization::GetExternalPoseActionForScan(uint64_t scan_seq, SE3* pose_to_apply) {
    std::lock_guard<std::mutex> lock(pending_external_pose_mutex_);
    if (!pending_external_pose_.active) {
        return ExternalPoseAction::kProcessScan;
    }

    if (scan_seq < pending_external_pose_.min_seq) {
        return ExternalPoseAction::kDropScan;
    }

    if (pose_to_apply != nullptr) {
        *pose_to_apply = pending_external_pose_.pose;
    }
    pending_external_pose_.active = false;
    return ExternalPoseAction::kApplyAndProcessScan;
}

bool Localization::ShouldDropLocResult(uint64_t scan_seq) {
    std::lock_guard<std::mutex> lock(pending_external_pose_mutex_);
    return pending_external_pose_.active && scan_seq < pending_external_pose_.min_seq;
}

bool Localization::HasPendingExternalPose() {
    std::lock_guard<std::mutex> lock(pending_external_pose_mutex_);
    return pending_external_pose_.active;
}

void Localization::SetGlobalLocCallback(Localization::GlobalLocCallback&& callback) {
    global_loc_callback_ = std::move(callback);
}

void Localization::SetContinuousLocalOdomCallback(Localization::ContinuousLocalOdomCallback&& callback) {
    continuous_local_odom_callback_ = std::move(callback);
}

void Localization::SetLocStateCallback(Localization::LocStateCallback&& callback) {
    loc_state_callback_ = std::move(callback);
}

}  // namespace lightning::loc
