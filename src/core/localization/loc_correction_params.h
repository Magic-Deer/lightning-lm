#pragma once

namespace lightning::loc {

struct LocCorrectionParams {
    bool correction_suspended = false;
    double reject_threshold_xy = 0.0;
    double reject_threshold_yaw = 0.0;
};

}  // namespace lightning::loc
