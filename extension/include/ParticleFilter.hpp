#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <sophus/se3.hpp>
#include <vector>
#include <cmath>
#include <random>

int NParticle = 1000;
using namespace cv;
using namespace std;
using namespace Eigen;
using namespace Sophus;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> EigenMatrix4dVector;
namespace PF
{
    struct Particle
    {
        static int GUID;
        Matrix4d pose;
        double weight;
        int idx;
        Particle(Matrix4d pose, double weight) : pose(pose), weight(weight), idx(GUID++){};
    };

    class ParticleFilter
    {
    public:
        std::vector<Particle> particles;
        ParticleFilter(Matrix4d init_pose = Matrix4d().Zero(), Matrix6d pose_cov = Matrix6d().Zero(), bool has_init_pose = false);
        void Resample();
    };

}

#endif