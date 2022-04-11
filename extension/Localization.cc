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

extern vector<double> alpha;
extern ArrayXd poseCov;

void SavePose(const string &filename, ParticleFilter &pf);

int main(int argc, char **argv)
{
    poseCov.setZero();
    // ! modify alpha here
    poseCov << 0.1, 0.1, 0.1, 0.1, 0.1, 0.1;
    if (argc != 7)
    {
        cerr << endl
             << "Usage: ./localization path_to_vocabulary path_to_settings path_to_sequence map_file path_to_motion pose_save_path" << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::STEREO, false, false, argv[4]);

    ObservationModel obModel(SLAM.mpMap->GetAllMapPoints());
    obModel.LoadImageSequence(argv[3]);
    MotionModel moModel(alpha);
    moModel.LoadMotions(argv[5]);
    Matrix4d initPose = moModel.GetInitPose();
    Matrix6d pose_cov;
    pose_cov.diagonal() << poseCov;
    ParticleFilter pf = ParticleFilter(initPose, pose_cov, true);
    size_t NFrame = obModel.strImageLeft.size();
    // get initial mean and covariance
    pf.getMeanAndCovariance();
    for (int i = 1; i < NFrame; i++)
    {
        double start = obModel.timestamps[i];
        double end = obModel.timestamps[i - 1];
        moModel.SampleMotion(pf.particles, end - start, i);
        obModel.sampleObservation(pf.particles, *SLAM.mpTracker, i);
        double sqrtSum = .0f;
        for (int i = 0; i < pf.particles.size(); i++)
        {
            sqrtSum += pf.particles[i].weight * pf.particles[i].weight;
        }
        int Neff = 1 / sqrtSum;
        if (Neff < NParticle / 5)
        {
            pf.Resample();
        }
        pf.getMeanAndCovariance();
    }

    SavePose(argv[6], pf);
    // Stop all threads
    SLAM.Shutdown();
}

void SavePose(const string &filename, ParticleFilter &pf)
{
    cout << endl
         << "Saving camera pose to" << filename << "..." << endl;
    const auto &poses = pf.meanPose;
    ofstream f;
    f.open(filename.c_str());
    for (const auto &pose : poses)
    {
        f << setprecision(6) << pose(0, 0) << " " << pose(0, 1) << " " << pose(0, 2) << " " << pose(0, 3) << " "
          << pose(1, 0) << " " << pose(1, 1) << " " << pose(1, 2) << " " << pose(1, 3) << " "
          << pose(2, 0) << " " << pose(2, 1) << " " << pose(2, 2) << " " << pose(2, 3) << endl;
    }
    f.close();
    cout << endl
         << "pose saved" << endl;
}