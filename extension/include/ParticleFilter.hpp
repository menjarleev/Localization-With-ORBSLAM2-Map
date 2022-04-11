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

using namespace cv;
using namespace std;
using namespace Eigen;
using namespace Sophus;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> EigenMatrix4dVector;
namespace PF
{
    class Particle
    {
    public:
        static int GUID;
        Matrix4d pose;
        double weight;
        int idx;
        Particle(Matrix4d &pose, double weight, int idx) : pose(pose), weight(weight), idx(idx){};
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    class ParticleFilter
    {
    public:
        vector<Matrix4d> meanPose;
        vector<Matrix6d> sigma;
        std::vector<Particle> particles;
        ParticleFilter(const Matrix4d &init_pose = Matrix4d().Identity(), const Matrix6d &pose_cov = Matrix6d(), bool has_init_pose = false);
        void Resample();
        void getMeanAndCovariance();
    };

}

#endif