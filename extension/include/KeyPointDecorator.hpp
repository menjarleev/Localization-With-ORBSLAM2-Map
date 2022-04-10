#ifndef KEY_POINT_DECORATOR_H
#define KEY_POINT_DECORATOR_H
#include <vector>
#include <MapPoint.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

using namespace ORB_SLAM2;
extern int NParticle;
namespace PF
{
    class KeyPointDecorator
    {
    public:
        int numMatch;
        const cv::Mat &descriptor;
        const cv::Mat x3Dc;
        const cv::KeyPoint &data;
        const double sigma;
        std::vector<bool> matched;
        std::vector<MapPoint *> matchedMapPoints;
        std::vector<double> reprojectionErrors;
        double weight;
        double maxReprojectionError;
        KeyPointDecorator(const cv::Mat &descriptor, cv::Mat x3Dc, cv::KeyPoint &data, double cov);
        void SetWeight(double w);
    };

}
#endif