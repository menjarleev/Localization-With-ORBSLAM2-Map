#include <string>
#include <vector>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int TH_DIST_HIGH = 100;
int TH_DIST_LOW = 50;
double TH_RATIO = 0.6;
int NParticle = 100;
bool LOAD_MAP = true;
vector<double> alpha = {0.01, 0.01, 0.01, 0.01, 0.01, 0.01};
ArrayXd poseCov(6);
int RADIUS = 100;
