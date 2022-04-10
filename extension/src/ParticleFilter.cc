#include "ParticleFilter.hpp"
#include <Eigen/StdVector>
#include "Utils.hpp"
using namespace Eigen;
namespace PF
{
    // GUID starts from 0
    int Particle::GUID = 0;

    ParticleFilter::ParticleFilter(Matrix4d init_pose, Matrix6d pose_cov, bool has_init_pose)
    {
        // if no initial pose provided, initialize pose with large covariance
        if (!has_init_pose)
        {
            init_pose.diagonal() << 1, 1, 1, 1;
            pose_cov.setZero();
            pose_cov.diagonal() << 10000, 10000, 10000, 10000, 10000, 10000;
        }
        // cholesky decomposition
        LLT<MatrixXd> lltofPose_cov(pose_cov);
        MatrixXd L = lltofPose_cov.matrixL();

        // uniform weights
        double weight = ((double)1) / NParticle;
        SE3d init_pose_SE3d(init_pose);

        // initialize particles
        particles.reserve(NParticle);
        for (int i = 0; i < NParticle; i++)
        {
            // need to verify correctness
            Vector6d noise = SampleNoise(L);
            Vector6d randPose_se3 = init_pose_SE3d.log() + noise;
            Matrix4d randPose = SE3d::exp(randPose_se3).matrix();
            particles.emplace_back(randPose, weight);
        }
    }

    void ParticleFilter::Resample()
    {
        // calculate cumulative sum
        vector<double> cumSum(NParticle, 0);
        for (int i = 0; i < NParticle; i++)
        {
            cumSum[i] = (i == 0 ? 0 : cumSum[i - 1]) + particles[i].weight;
        }
        // generate r [0, 1]
        double r = ((double)rand()) / RAND_MAX;
        int j = 1;
        for (int i = 0; i < NParticle; i++)
        {
            double u = r + i / NParticle;
            while (u > cumSum[j])
            {
                j++;
            }
            particles[i] = particles[j];
            particles[i].weight = ((double)1) / NParticle;
        }
    }
}