#include <iostream>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <unistd.h>
#include <Eigen/StdVector>
#include <sophus/se3.hpp>

#include "System.h"
#include "MapPoint.h"
#include "ParticleFilter.hpp"
#include "MotionModel.hpp"
#include "ObservationModel.hpp"

using namespace std;
using namespace ORB_SLAM2;
using namespace PF;

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        cerr << endl
             << "Usage: ./localization path_to_vocabulary path_to_settings path_to_sequence path_to_motion" << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::STEREO, false);

    ObservationModel obModel;
    obModel.LoadImageSequence(argv[3]);
    MotionModel moModel;
    moModel.LoadMotions(argv[4]);
    Matrix4d initPose = moModel.GetInitPose();
    Matrix6d initPoseCov;
    initPoseCov.diagonal() << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
    ParticleFilter pf = ParticleFilter(initPose, initPoseCov, false);
    while (!obModel.finish() || !moModel.finish())
    {
        if (!moModel.finish())
        {
            double start = obModel.timestamps[moModel.motionIdx - 1];
            double end = obModel.timestamps[moModel.motionIdx];
            moModel.SampleMotion(pf.particles, end - start);
        }
        if (!obModel.finish())
        {
            obModel.sampleObservation(pf.particles, *SLAM.mpTracker);
        }
    }
    // Stop all threads
    SLAM.Shutdown();
}