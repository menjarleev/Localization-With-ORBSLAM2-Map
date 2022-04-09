#include <opencv2/core/core.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
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

// using MotionModelPtr = void (*const)(std::vector<Particle>&, MatrixXd Tlc, VectorXd noise);
// TODO fill up the measurement model
using MeasurementModelPtr = void (*const)();
class ParticleFilter
{
private:
    std::vector<Matrix4d> particlePoses;
    std::vector<double> particleWeights;
    // hyper-parameter for motion model
    Array<double, 6, 1> alpha;
    int N;

    static std::default_random_engine generator;
    static std::normal_distribution<double> dist;

    Vector6d SampleNoise(Matrix6d L);

public:
    ParticleFilter(int n_sample = 1000, Matrix4d init_pose = Matrix4d().Zero(), Matrix6d pose_cov = Matrix6d().Zero(), bool has_init_pose = false, vector<double> alpha = vector<double>(6, 0.001));

    void Resample();
    void SampleMotion(Matrix4d Tlc, double timeElapse);
    void ImportanceMeasurement();
};