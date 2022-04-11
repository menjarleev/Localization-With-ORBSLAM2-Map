#ifndef UTILS_H
#define UTILS_H

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;
namespace PF
{

    VectorXd SampleNoise(const MatrixXd& L);
    double SampleNoise(double sigma);
    double GaussianPDF(double x, double mean, double sigma);

}

#endif