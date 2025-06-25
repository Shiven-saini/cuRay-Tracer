#include "FPSCounter.h"
#include <sstream>
#include <iomanip>

FPSCounter::FPSCounter() 
    : fps(0.0f), deltaTime(0.0f), frameCount(0), fpsUpdateTimer(0.0f) {
    lastTime = std::chrono::high_resolution_clock::now();
    currentTime = lastTime;
}

void FPSCounter::update() {
    currentTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(currentTime - lastTime);
    deltaTime = duration.count() / 1000000.0f;
    lastTime = currentTime;
    
    frameCount++;
    fpsUpdateTimer += deltaTime;
    
    if (fpsUpdateTimer >= 1.0f) {
        fps = frameCount / fpsUpdateTimer;
        frameCount = 0;
        fpsUpdateTimer = 0.0f;
    }
}

std::string FPSCounter::getFPSString() const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << fps << " FPS";
    return oss.str();
}