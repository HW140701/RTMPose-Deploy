# RTMPose-Deploy

[中文说明](./README_CN.md)

RTMPose-Deploy is a C ++ code example that does not use MMDEPLOY for RTMPose localized deployment.

At present, we use the onnxruntime CPU SDK on the Windows system to localize the RTMDetnano and RTMPose, and build a simple frame-skipping RTMPoseTracker class for real-time estimates pose on the CPU.

Subsequent will consider adding the use of C ++ Tensorrt SDK on the Windows system to deploy RTMDetnano and RTMPose.