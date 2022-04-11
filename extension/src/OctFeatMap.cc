#include "OctFeatMap.hpp"
#include <vector>
#include "Converter.h"
namespace PF
{
    OctFeatMap::OctFeatMap(std::vector<MapPoint *> sMapPoints) : mapPoints(sMapPoints), octree(mapPoints), index(3, octree, {10})
    {
    }

    bool OctFeatMap::FindMatch(KeyPointDecorator &kp, Particle &particle, double const searchRange)
    {
        // find keypoints in the map that is within radius
        const cv::Mat pose = kp.x3Dc;
        // get the rotation and transformation matrix for current particle hypothesis
        cv::Mat Tcw_hat = ORB_SLAM2::Converter::toCvMat(particle.pose);
        cv::Mat R = Tcw_hat.rowRange(0, 3).colRange(0, 3);
        cv::Mat t = Tcw_hat.rowRange(0, 3).col(3);
        cv::Mat T_feat_hat = R * pose + t;
        double queryPoint[3] = {T_feat_hat.at<float>(0), T_feat_hat.at<float>(1), T_feat_hat.at<float>(2)};
        int num_results = (int)searchRange;
        std::vector<uint32_t> ret_index(num_results);
        std::vector<double> out_dist_sqr(num_results);
        // nanoflann::RadiusResultSet<double, size_t> resultSet(radius, indiciesDists);
        // nanoflann::KNNResultSet<double> resultSet(int(searchRange));
        num_results = index.knnSearch(&queryPoint[0], num_results, &ret_index[0], &out_dist_sqr[0]);
        ret_index.resize(num_results);
        out_dist_sqr.resize(num_results);

        int minDist1 = (TH_DIST_LOW + TH_DIST_HIGH) / 2;
        int bestIdx = -1;
        int minDist2 = (TH_DIST_LOW + TH_DIST_HIGH) / 2;
        for (int i = 0; i < num_results; i++)
        {
            int idx = ret_index[i];
            MapPoint *reference = mapPoints[idx];
            int tmpDist = DescriptorDistance(reference->mDescriptor, kp.descriptor);
            if (tmpDist < minDist1)
            {
                minDist2 = minDist1;
                minDist1 = tmpDist;
                bestIdx = i;
            }
            else if (tmpDist < minDist2)
            {
                minDist2 = tmpDist;
            }
        }
        if (minDist1 == (TH_DIST_LOW + TH_DIST_HIGH) / 2)
        {
            return false;
        }
        else
        {
            double ratio = ((double)minDist1) / minDist2;
            if (ratio < TH_RATIO)
            {
                int particleIdx = particle.idx;
                kp.matched[particleIdx] = bestIdx;
                kp.numMatch++;
                kp.matchedMapPoints[particleIdx] = mapPoints[bestIdx];
                return true;
            }
            else
            {
                return false;
            }
        }
    }

    int OctFeatMap::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
    {
        const int *pa = a.ptr<int32_t>();
        const int *pb = b.ptr<int32_t>();

        int dist = 0;

        // 8*32=256bit

        for (int i = 0; i < 8; i++, pa++, pb++)
        {
            unsigned int v = *pa ^ *pb;
            v = v - ((v >> 1) & 0x55555555);
            v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
            dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
        }

        return dist;
    }
}