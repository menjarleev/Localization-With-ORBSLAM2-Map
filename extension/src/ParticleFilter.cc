#include "ParticleFilter.hpp"
#include <Eigen/StdVector>
#include "Utils.hpp"
using namespace Eigen;

namespace PF
{
    // GUID starts from 0
    int Particle::GUID = 0;

    ParticleFilter::ParticleFilter(const Matrix4d &init_pose, const Matrix6d &pose_cov_, bool has_init_pose)
    {
        Matrix6d pose_cov = pose_cov_;
        // if no initial pose provided, initialize pose with large covariance
        if (!has_init_pose)
        {
            pose_cov.diagonal() << 100, 100, 100, 100, 100, 100;
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
            particles.emplace_back(randPose, weight, Particle::GUID++);
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

    void ParticleFilter::getMeanAndCovariance()
    {
        // convert pose into se3
        Vector6d mean;
        vector<Vector6d> poses_se3;
        poses_se3.reserve(particles.size());
        mean.setZero();
        MatrixXd WMatrix(particles.size(), particles.size());
        for (int i = 0; i < particles.size(); i++)
        {
            Vector6d pose_se3 = SE3d(particles[i].pose).log();
            poses_se3[i] = pose_se3;
            mean += particles[i].weight * pose_se3;
            WMatrix(i, i) = particles[i].weight;
        }
        MatrixXd zeroMean(6, particles.size());
        for (int i = 0; i < particles.size(); i++)
        {
            auto diff = poses_se3[i] - mean;
            for (int j = 0; j < 6; j++)
            {
                zeroMean(j, i) = diff(j);
            }
        }
        this->sigma.push_back(zeroMean * WMatrix * zeroMean.transpose());
        this->meanPose.push_back(SE3d::exp(mean).matrix());
    }
}