#ifndef FRAME_DECORATOR_H
#define FRMAE_DECORATOR_H

#include "Frame.h"
#include "KeyPointDecorator.hpp"
#include <vector>

using namespace ORB_SLAM2;

namespace PF
{
    class FrameDecorator
    {
    public:
        const Frame &frame;
        std::vector<KeyPointDecorator> keyPoints;

        FrameDecorator(Frame &frame, int height, int width);
        FrameDecorator(Frame frame, int height, int width);
        FrameDecorator &operator=(const FrameDecorator &fd);
        void assignWeightForKeyPoints();
    };
}

#endif