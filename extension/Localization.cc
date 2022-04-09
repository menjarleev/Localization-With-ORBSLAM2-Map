#include <iostream>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <unistd.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>
#include <unistd.h>
#include <sophus/se3.hpp>

#include "System.h"
#include "MapPoint.h"
#include "ParticleFilter.h"

using namespace std;
using namespace ORB_SLAM2;

// 基本操作类似 stereo_euroc.cc

void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

void LoadMotion(const string &strPathToSequence,
                vector<Matrix4d> &vTcl);

int main(int argc, char **argv)
{
    if (argc != 5)
    {
        cerr << endl
             << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence path_to_motion" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    vector<Matrix4d> vTcl;
    vector<pair<cv::KeyPoint, double>> observations;
    vector<MapPoint *> mapPoints;
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestamps);
    LoadMotion(string(argv[4]), vTcl);

    const int nImages = vstrImageLeft.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::STEREO, false);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);
    ParticleFilter pf = ParticleFilter();

    cout << endl
         << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl
         << endl;

    // NOTE 由于Kitti数据集的图像已经经过双目矫正的处理，所以这里就不需要再进行矫正的操作了

    // Main loop
    cv::Mat imLeft, imRight;
    for (int ni = 1; ni < nImages; ni++)
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni], CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if (imLeft.empty())
        {
            cerr << endl
                 << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

#ifdef COMPILEDWITHC17
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif

        pf.SampleMotion(vTcl[ni], vTimestamps[ni] - vTimestamps[ni - 1]);
        // Pass the images to the SLAM system
        SLAM.GetObservations(imLeft, imRight, tframe, observations, mapPoints);

#ifdef COMPILEDWITHC17
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

        vTimesTrack[ni] = ttrack;

        // Wait to load the next frame
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if (ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(), vTimesTrack.end());
    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++)
    {
        totaltime += vTimesTrack[ni];
    }
    cout << "-------" << endl
         << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;
    return 0;
}

// 类似 mono_kitti.cc， 不过是生成了双目的图像路径
void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while (!fTimes.eof())
    {
        string s;
        getline(fTimes, s);
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);

    for (int i = 0; i < nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}

void LoadMotion(const string &strPathToSequence,
                vector<Matrix4d> &vTcl)
{
    ifstream fSE3;
    fSE3.open(strPathToSequence.c_str());
    while (!fSE3.eof())
    {
        string s;
        getline(fSE3, s);
        if (!s.empty())
        {
            Eigen::Matrix4d m;
            stringstream ss(s);
            vector<double> R_T(12, 0);
            for (int i = 0; i < 12; i++)
            {
                ss >> R_T[i];
            }
            m << R_T[0], R_T[1], R_T[2], R_T[3], R_T[4], R_T[5], R_T[6], R_T[7], R_T[8], R_T[9], R_T[10], R_T[11], 0, 0, 0, 1;
            // cout << SE3_R.matrix() << endl;
            vTcl.emplace_back(m);
        }
    }
    fSE3.close();
}