#include "ParticleFilter.h"
using namespace Eigen;

typedef Eigen::Matrix<double, 6, 1> Vector6d;
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
// normal distribution generator
std::default_random_engine ParticleFilter::generator;
std::normal_distribution<double> ParticleFilter::dist(0, 1);

Vector6d ParticleFilter::SampleNoise(Matrix6d L)
{
    Vector6d seed;
    for (int i = 0; i < 6; i++)
    {
        seed(i) = ParticleFilter::dist(ParticleFilter::generator);
    }
    return L * seed;
}

ParticleFilter::ParticleFilter(int n_sample, Matrix4d init_pose, Matrix6d pose_cov, bool has_init_pose, vector<double> alpha) : N(n_sample)
{
    // if no initial pose provided, initialize pose with large covariance
    if (!has_init_pose)
    {
        init_pose.diagonal() << 1, 1, 1, 1;
        pose_cov.setZero();
        pose_cov.diagonal() << 10000, 10000, 10000, 10000, 10000, 10000;
    }
    for (int i = 0; i < 6; i++)
    {
        this->alpha(i) = alpha[i];
    }
    // cholesky decomposition
    LLT<MatrixXd> lltofPose_cov(pose_cov);
    MatrixXd L = lltofPose_cov.matrixL();
    // uniform weights
    double weight = ((double)1) / N;
    // initialize particles
    particleWeights.reserve(N);
    particlePoses.reserve(N);
    SE3d init_pose_SE3d(init_pose);
    for (int i = 0; i < N; i++)
    {
        Vector6d noise = SampleNoise(L);
        // TODO: can we assume the noise to be small?
        Vector6d randPose_se3 = init_pose_SE3d.log() + noise;
        Matrix4d randPose = SE3d::exp(randPose_se3).matrix();
        particleWeights.emplace_back(weight);
        particlePoses.emplace_back(randPose);
    }
}

void ParticleFilter::SampleMotion(Matrix4d Tcl, double timeElapse)
{
    // cholesky decomposition of motion noise
    SE3d SE3_Tcl = SE3d(Tcl);
    Vector6d delta_se3 = SE3_Tcl.log() / timeElapse;

    Array<double, 6, 1> diag = this->alpha * delta_se3.array().square();
    Matrix6d Q = Matrix6d::Zero();
    Q.diagonal() << diag;
    LLT<Matrix6d> lltofQ(Q);
    Matrix6d LQ = lltofQ.matrixL();
    for (int i = 0; i < N; i++)
    {
        Vector6d noise = SampleNoise(LQ);
        // TODO: check wrap to pi
        Matrix4d Tlc_with_noise = SE3d::exp(SE3_Tcl.log() + noise).matrix();
        particlePoses[i] = Tlc_with_noise * particlePoses[i];
    }
}

void ParticleFilter::ImportanceMeasurement()
{
    // TODO
    return;
}

void ParticleFilter::Resample()
{
    // calculate cumulative sum
    vector<double> cumSum(N);
    partial_sum(particleWeights.begin(), particleWeights.end(), cumSum.begin());
    // generate r [0, 1]
    double r = ((double)rand()) / RAND_MAX;
    int j = 1;
    for (int i = 0; i < N; i++)
    {
        double u = r + i / N;
        while (u > cumSum[j])
        {
            j++;
        }
        particlePoses[i] = particlePoses[j];
        particleWeights[i] = ((double)1) / N;
    }
}
