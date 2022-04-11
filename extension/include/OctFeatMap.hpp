#ifndef OCTFEATMAP_H
#define OCTFEATMAP_H

#include "KeyPointDecorator.hpp"
#include "MapPoint.h"
#include "nanoflann.hpp"
#include "ParticleFilter.hpp"
#include "ORBmatcher.h"
#include <set>
#include <vector>
using namespace ORB_SLAM2;
namespace PF
{
    const int TH_DIST_HIGH = 100;
    const int TH_DIST_LOW = 50;
    const double TH_RATIO = 0.6;
    struct PointCloudAdaptor
    {
        const std::vector<MapPoint *> &mapPoints;
        PointCloudAdaptor() = delete;
        PointCloudAdaptor(std::vector<MapPoint *> &mapPoints) : mapPoints(mapPoints) {}
        inline size_t kdtree_get_point_count() const
        {
            return mapPoints.size();
        }
        inline double kdtree_get_pt(const size_t idx, const size_t dim) const
        {
            if (dim == 0)
                return mapPoints[idx]->GetWorldPos().at<double>(0);
            else if (dim == 1)
                return mapPoints[idx]->GetWorldPos().at<double>(1);
            else
                return mapPoints[idx]->GetWorldPos().at<double>(2);
        }
        template <class BBOX>
        bool kdtree_get_bbox(BBOX &bb) const
        {
            return false;
        }
    };

    // oct tree feature map
    using kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloudAdaptor>, PointCloudAdaptor, 3>;
    class OctFeatMap
    {
    private:
        std::vector<MapPoint *> mapPoints;
        PointCloudAdaptor octree;
        kd_tree_t index;
        int DescriptorDistance(const cv::Mat &a, const cv::Mat &b);

    public:
        OctFeatMap(std::vector<MapPoint *> sMapPoints);
        bool FindMatch(KeyPointDecorator &kp, Particle &particle, const double radius);
    };

}
#endif