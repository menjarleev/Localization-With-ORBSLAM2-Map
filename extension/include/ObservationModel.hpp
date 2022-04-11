#ifndef OBSERVATION_MODEL_H
#define OBSERVATION_MODEL_H
#include <ParticleFilter.hpp>
#include "OctFeatMap.hpp"
#include <opencv2/core/core.hpp>
#include "Tracking.h"
#include "FrameDecorator.hpp"

namespace PF
{
    // visitor class of tracking
    class ObservationModel
    {
    private:
        std::shared_ptr<FrameDecorator> currentFrameDec;
        OctFeatMap featMap;

    public:
        vector<string> strImageRight;
        vector<string> strImageLeft;
        vector<float> timestamps;

        ObservationModel(vector<MapPoint *> mapPoints);
        void sampleObservation(vector<Particle> &particles, Tracking &tracker, int observationIdx);
        void GetObservation(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, Tracking &tracker);
        void ImportanceMeasurement(vector<Particle> &particles);
        void LoadImageSequence(const string &strPathToSequence);
    };
}

#endif