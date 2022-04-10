#include "KeyPointDecorator.hpp"

namespace PF
{
    /**
     * @brief Construct a new Key Point Decorator:: Key Point Decorator object
     *
     * @param[in] N number of particles
     * @param[in] descriptor descriptor of keypoint
     * @param[in] x3Dc 3D pose in camera frame
     * @param[in] data (u, v) coordinate in observation
     */
    KeyPointDecorator::KeyPointDecorator(const cv::Mat &descriptor, cv::Mat x3Dc, cv::KeyPoint &data, double cov) :
    descriptor(descriptor), x3Dc(x3Dc), data(data), maxReprojectionError(0), sigma(sqrt(cov))
    {
        // initalize as all particles do not find the keypoint
        numMatch = 0;
        matched = std::vector<bool>(NParticle, false);
        matchedMapPoints = std::vector<MapPoint *>(NParticle, nullptr);
        reprojectionErrors = std::vector<double>(NParticle, -1);
    }

    /**
     * @brief assign weight for this keypoint in the reprojection error
     *
     * @param[in] w
     */
    void KeyPointDecorator::SetWeight(double w)
    {
        this->weight = w;
    }
}