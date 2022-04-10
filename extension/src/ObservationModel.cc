#include "ObservationModel.hpp"
#include "Converter.h"
#include "Utils.hpp"

namespace PF
{
    int ObservationModel::observationIdx = 0;

    ObservationModel::ObservationModel() : currentFrameDec(shared_ptr<FrameDecorator>(nullptr)) {}

    void ObservationModel::sampleObservation(vector<Particle>& particles, Tracking& tracker)
    {
        // Read left and right images from file
        cv::Mat imageRectLeft = cv::imread(strImageLeft[observationIdx], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat imageRectRight = cv::imread(strImageRight[observationIdx], CV_LOAD_IMAGE_UNCHANGED);
        double tframe = timestamps[observationIdx];

        if (imageRectLeft.empty())
        {
            cerr << endl
                 << "Failed to load image at: "
                 << string(strImageLeft[observationIdx]) << endl;
            return;
        }
        GetObservation(imageRectLeft, imageRectRight, tframe, tracker);
        ImportanceMeasurement(particles);
        observationIdx++;
    }

    void ObservationModel::GetObservation(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp, Tracking &tracker)
    {
        currentFrameDec = shared_ptr<FrameDecorator>(new FrameDecorator(tracker.GetFrameForObservation(imRectLeft, imRectRight, timestamp)));
    }

    void ObservationModel::ImportanceMeasurement(vector<Particle> &particles)
    {
        // assign weight for different keypoint
        currentFrameDec->assignWeightForKeyPoints();
        for (int i = 0; i < currentFrameDec->keyPoints.size(); i++)
        {
            KeyPointDecorator &kp = currentFrameDec->keyPoints[i];
            // there is at least one match
            if (kp.numMatch > 0)
            {
                for (int j = 0; j < particles.size(); j++)
                {
                    // if there is a match for particle j for key point i
                    if (kp.matched[i])
                    {
                        // get the rotation and transformation matrix for current particle hypothesis
                        cv::Mat Tcw_hat = ORB_SLAM2::Converter::toCvMat(particles[j].pose);
                        cv::Mat R = Tcw_hat.rowRange(0, 3).colRange(0, 3);
                        cv::Mat t = Tcw_hat.rowRange(0, 3).col(3);
                        cv::Mat T_feat_hat = R * kp.x3Dc + t;

                        // calculate the reporjection error
                        double invZ1 = 1 / T_feat_hat.at<float>(2);
                        double u_hat = currentFrameDec->frame.fx * T_feat_hat.at<float>(0) * invZ1 + currentFrameDec->frame.cx;
                        double v_hat = currentFrameDec->frame.fy * T_feat_hat.at<float>(1) * invZ1 + currentFrameDec->frame.cy;
                        double sqError = (u_hat - kp.data.pt.x) * (u_hat - kp.data.pt.x) + (v_hat - kp.data.pt.y) * (v_hat - kp.data.pt.y);
                        kp.reprojectionErrors[j] = sqError;

                        // update max reprojection error for later use
                        if (sqError > kp.maxReprojectionError)
                        {
                            kp.maxReprojectionError = sqError;
                        }
                    }
                }
            }
        }

        for (int i = 0; i < particles.size(); i++)
        {
            double error = 0; // reprojection error
            double R = 0.f;   // measurement noise
            for (int j = 0; j < currentFrameDec->keyPoints.size(); j++)
            {
                KeyPointDecorator &kp = currentFrameDec->keyPoints[j];
                if (kp.numMatch > 0)
                {
                    // no match for particle i at keypoint j
                    if (kp.matched[i] == -1)
                    {
                        error += kp.weight * kp.maxReprojectionError; // add noise caused by scale
                    }
                    // else get reprojection error
                    else
                    {
                        error += kp.weight * kp.reprojectionErrors[i];
                    }
                    R += kp.weight * kp.sigma * kp.weight; // R = WSW^T
                }
            }
            // assign weigh for particles
            particles[i].weight = GaussianPDF(error, 0, R);
        }
    }

    void ObservationModel::LoadImageSequence(const string &strPathToSequence)
    {
        ifstream fTimes;
        string strPathTimeFile = strPathToSequence + "/times.txt";
        fTimes.open(strPathTimeFile.c_str());
        while (!fTimes.eof())
        {
            string s;
            getline(fTimes, s);
            if (!s.empty())
            {
                stringstream ss;
                ss << s;
                double t;
                ss >> t;
                timestamps.push_back(t);
            }
        }

        string strPrefixLeft = strPathToSequence + "/image_0/";
        string strPrefixRight = strPathToSequence + "/image_1/";

        const int nTimes = timestamps.size();
        strImageLeft.resize(nTimes);
        strImageRight.resize(nTimes);

        for (int i = 0; i < nTimes; i++)
        {
            stringstream ss;
            ss << setfill('0') << setw(6) << i;
            strImageLeft[i] = strPrefixLeft + ss.str() + ".png";
            strImageRight[i] = strPrefixRight + ss.str() + ".png";
        }
    }

    bool ObservationModel::finish()
    {
        return this->strImageLeft.size() == observationIdx;
    }
}
