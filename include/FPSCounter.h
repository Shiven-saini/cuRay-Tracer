#ifndef FPSCOUNTER_H
#define FPSCOUNTER_H

#include <chrono>
#include <string>

class FPSCounter {
public:
    FPSCounter();
    ~FPSCounter() = default;

    void update();
    float getFPS() const { return fps; }
    float getDeltaTime() const { return deltaTime; }
    std::string getFPSString() const;

private:
    std::chrono::high_resolution_clock::time_point lastTime;
    std::chrono::high_resolution_clock::time_point currentTime;
    float fps;
    float deltaTime;
    int frameCount;
    float fpsUpdateTimer;
};

#endif