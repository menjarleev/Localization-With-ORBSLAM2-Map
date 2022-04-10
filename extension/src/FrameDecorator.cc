#include "FrameDecorator.hpp"
namespace PF
{
    FrameDecorator::FrameDecorator(Frame &frame) : frame(frame)
    {
        keyPoints.reserve(frame.mvKeysUn.size());
        for (int i = 0; i < frame.mvKeysUn.size(); i++)
        {
            cv::Mat featPoseInCameraFrame_i = frame.GetPoseInCameraFrame(i, frame.mvDepth[i]);
            // create covariance based on scale
            double sigma2 = frame.mvLevelSigma2[frame.mvKeysUn[i].octave];
            Eigen::Matrix2d cov;
            cov << sigma2, 0, 0, sigma2;
            const cv::Mat &descriptor = frame.mDescriptors.row(i);
            keyPoints.emplace_back(descriptor, featPoseInCameraFrame_i, frame.mvKeysUn[i], cov);
        }
    }

    FrameDecorator::FrameDecorator(Frame frame) : frame(frame)
    {
        keyPoints.reserve(frame.mvKeysUn.size());
        for (int i = 0; i < frame.mvKeysUn.size(); i++)
        {
            cv::Mat featPoseInCameraFrame_i = frame.GetPoseInCameraFrame(i, frame.mvDepth[i]);
            // create covariance based on scale
            double sigma2 = frame.mvLevelSigma2[frame.mvKeysUn[i].octave];
            const cv::Mat &descriptor = frame.mDescriptors.row(i);
            keyPoints.emplace_back(descriptor, featPoseInCameraFrame_i, frame.mvKeysUn[i], sigma2);
        }
    }

    void FrameDecorator::assignWeightForKeyPoints()
    {
        double totalWeight = .0f;
        for (const auto &keyPoint : keyPoints)
        {
            if (keyPoint.numMatch > 0)
            {
                totalWeight += exp(keyPoint.numMatch);
            }
        }
        for (auto &keyPoint : keyPoints)
        {
            if (keyPoint.numMatch > 0)
            {
                keyPoint.SetWeight(exp(keyPoint.numMatch) / totalWeight);
            }
        }
    }
}