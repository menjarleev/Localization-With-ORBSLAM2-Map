#ifndef OBSERVATION_MODEL_H
#define OBSERVATION_MODEL_H
#include <ParticleFilter.hpp>
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

    public:
        vector<string> strImageRight;
        vector<string> strImageLeft;

        ObservationModel();
        bool finish();
        static int observationIdx;
        vector<float> timestamps;
        void sampleObservation(vector<Particle> &particles, Tracking& tracker);
        void GetObservation(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, Tracking &tracker);
        void ImportanceMeasurement(vector<Particle> &particles);
        void LoadImageSequence(const string &strPathToSequence);
    };
}

#endif