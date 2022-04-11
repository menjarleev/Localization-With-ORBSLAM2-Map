#include "MotionModel.hpp"
#include <iostream>
#include <string>
#include <fstream>
#include <sophus/se3.hpp>
#include "Utils.hpp"

extern int NParticle;

using namespace Sophus;
using namespace std;
namespace PF
{
    MotionModel::MotionModel(std::vector<double> alpha)
    {
        for (int i = 0; i < 6; i++)
        {
            this->alpha(i) = alpha[i];
        }
    }
    void MotionModel::LoadMotions(const std::string &strPathToSequence)
    {
        ifstream fSE3;
        fSE3.open(strPathToSequence.c_str());
        bool uninitialized = true;
        Eigen::Matrix4d Twl; // last to world
        while (!fSE3.eof())
        {
            string s;
            getline(fSE3, s);
            if (!s.empty())
            {
                Eigen::Matrix4d Tcw;
                stringstream ss(s);
                vector<double> R_T(12, 0);
                for (int i = 0; i < 12; i++)
                {
                    ss >> R_T[i];
                }
                Tcw << R_T[0], R_T[1], R_T[2], R_T[3], R_T[4], R_T[5], R_T[6], R_T[7], R_T[8], R_T[9], R_T[10], R_T[11], 0, 0, 0, 1;
                if (uninitialized)
                {
                    Tcl.emplace_back(Tcw);
                    uninitialized = false;
                }
                else
                {
                    Tcl.emplace_back(Tcw * Twl);
                }
                Twl.setZero();
                Eigen::Matrix3d R = Tcw.block(0, 0, 3, 3);
                Eigen::Vector3d t = Tcw.block(0, 3, 3, 1);
                Twl.block(0, 0, 3, 3) = R.transpose();
                Twl.block(0, 3, 3, 1) = -R.transpose() * t;
                Twl(3, 3) = 1;
            }
        }
        fSE3.close();
    }
    void MotionModel::SampleMotion(std::vector<Particle> &particles, double timeElapse, int motionIdx)
    {
        // cholesky decomposition of motion noise
        SE3d SE3_Tcl = SE3d(Tcl[motionIdx]);
        Vector6d delta_se3 = SE3_Tcl.log() / timeElapse;

        Array<double, 6, 1> diag = this->alpha * delta_se3.array().square();
        Matrix6d Q = Matrix6d::Zero();
        Q.diagonal() << diag;
        LLT<Matrix6d> lltofQ(Q);
        Matrix6d LQ = lltofQ.matrixL();
        for (int i = 0; i < NParticle; i++)
        {
            Vector6d noise = SampleNoise(LQ).cast<double>();
            // TODO: check wrap to pi
            Matrix4d Tcl_with_noise = SE3d::exp(SE3_Tcl.log() + noise).matrix();
            particles[i].pose = Tcl_with_noise * particles[i].pose;
        }
        motionIdx++;
    }

    Matrix4d MotionModel::GetInitPose()
    {
        return Tcl[0];
    }
}