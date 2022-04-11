#ifndef MOTION_MODEL_H
#define MOTION_MODEL_H
#include <string>
#include <vector>
#include "ParticleFilter.hpp"

namespace PF
{
    class MotionModel
    {
    private:
        Array<double, 6, 1> alpha;
        EigenMatrix4dVector Tcl;

    public:
        Matrix4d GetInitPose();
        MotionModel(std::vector<double> alpha = std::vector<double>(6, 1));
        void LoadMotions(const std::string &strPathToSequence);
        void SampleMotion(std::vector<Particle> &particles, double timeElapse, int motionIdx);
    };
}
#endif