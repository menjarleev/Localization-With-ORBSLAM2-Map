#include <iostream>
#include <fstream>
#include <unistd.h>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

using namespace std;
using namespace Sophus;

void DrawTrajectory(const vector<Matrix4d> &gt, const vector<Matrix4d> &est);
void DrawLines(const vector<Matrix4d> &pose, vector<float> color);
vector<Matrix4d> ReadTrajectory(const string &path);

int main(int argc, char **argv)
{
    auto gt = ReadTrajectory(argv[1]);
    auto est = ReadTrajectory(argv[2]);
    assert(!gt.empty() && gt.size() == est.size());
    // double rmse = 0;
    // for (size_t i = 0; i < est.size(); i++)
    // {
    //     Matrix4d p1 = est[i], p2 = gt[i];
    //     double error = SE3d(p2.inverse() * p1).log().norm();
    //     rmse += error * error;
    // }
    // rmse = rmse / double(est.size());
    // rmse = sqrt(rmse);
    // cout << "RMSE = " << rmse << endl;
    DrawTrajectory(gt, est);
}

using namespace pangolin;

void DrawLines(const vector<Matrix4d> &pose, vector<float> color)
{
    for (size_t i = 0; i < pose.size() - 1; i++)
    {
        glColor3f(color[0], color[1], color[2]);
        glBegin(GL_LINES);
        Matrix4d p1 = pose[i], p2 = pose[i + 1];
        glVertex3d(p1(0, 3), p1(1, 3), p1(2, 3));
        glVertex3d(p2(0, 3), p2(1, 3), p2(2, 3));
        glEnd();
    }
}

void DrawTrajectory(const vector<Matrix4d> &gt, const vector<Matrix4d> &est)
{
    CreateWindowAndBind("Trajectory Viewer", 1024, 768);
    glEnable(GL_DEPTH);

    OpenGlRenderState s_cam(
        ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
        ModelViewLookAt(-2, 0, -2, 0, 0, 0, AxisY));

    View &d_cam = CreateDisplay().SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f).SetHandler(new Handler3D(s_cam));

    while (!ShouldQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glLineWidth(2);
        DrawLines(gt, {0, 0, 1});
        DrawLines(est, {1, 0, 0});
        FinishFrame();
        usleep(5000);
    }
}

vector<Matrix4d> ReadTrajectory(const string &path)
{
    ifstream fin(path);
    vector<Matrix4d> trajectory;
    if (!fin)
    {
        cerr << "trajectory " << path << " not found" << endl;
        return trajectory;
    }
    while (!fin.eof())
    {
        Eigen::Matrix4d Twc = Eigen::Matrix4d::Zero();
        Twc(3, 3) = 1;
        int idx = 0;
        while (idx != 12)
        {
            float tmp;
            fin >> tmp;
            Twc(idx / 4, idx % 4) = tmp;
            idx++;
        }
        trajectory.push_back(Twc);
    }
    return trajectory;
}