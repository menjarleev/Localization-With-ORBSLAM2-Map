#include "Utils.hpp"
#include <cmath>
#include <random>

namespace PF
{
    // normal distribution generator
    static std::default_random_engine generator;
    static std::normal_distribution<double> dist(0, 1);

    VectorXd SampleNoise(const MatrixXd& L)
    {
        VectorXd seed(L.rows());
        for (int i = 0; i < L.rows(); i++)
        {
            seed(i) = dist(generator);
        }
        return L * seed;
    }

    double SampleNoise(double sigma)
    {
        return sigma * dist(generator);
    }

    double GaussianPDF(double x, double mean, double sigma)
    {
        static const double inv_sqrt_2pi = 0.3989422804014327;
        double a = (x - mean) / sigma;

        return inv_sqrt_2pi / sigma * std::exp(-0.5f * a * a);
    }
}