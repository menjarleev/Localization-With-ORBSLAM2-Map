#include "OctFeatMap.hpp"
#include <vector>
namespace PF
{
    OctFeatMap::OctFeatMap(std::set<MapPoint *> sMapPoints) : mapPoints(sMapPoints.begin(), sMapPoints.end()), octree(mapPoints), index(3, octree, {10})
    {
    }

    bool OctFeatMap::FindMatch(KeyPointDecorator &kp, Particle &particle, double const radius)
    {
        // find keypoints in the map that is within radius
        const cv::Mat pose = kp.x3Dc;
        double queryPoint[3] = {pose.at<double>(0), pose.at<double>(1), pose.at<double>(2)};
        std::vector<std::pair<size_t, double>> indiciesDists;
        nanoflann::RadiusResultSet<double, size_t> resultSet(radius, indiciesDists);
        index.findNeighbors(resultSet, queryPoint, nanoflann::SearchParams());

        if (indiciesDists.empty())
        {
            return false;
        }
        int minDist1 = TH_DIST;
        int bestIdx = -1;
        int minDist2 = TH_DIST;
        for (int i = 0; i < indiciesDists.size(); i++)
        {
            int idx = indiciesDists[i].first;
            MapPoint *reference = mapPoints[idx];
            int tmpDist = DescriptorDistance(reference->GetDescriptor(), kp.descriptor);
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
        if (minDist1 == TH_DIST || minDist2 == TH_DIST)
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