#ifndef SCENE_H
#define SCENE_H

#include "primitives/Sphere.h"
#include "primitives/Plane.h"
#include "math/Vec3.h"
#include <vector>

#define MAX_SPHERES 32
#define MAX_PLANES 32

struct SceneData {
    Sphere spheres[MAX_SPHERES];
    Plane planes[MAX_PLANES];
    int numSpheres;
    int numPlanes;
    Vec3 lightPosition;
    Vec3 lightColor;
};

class Scene {
public:
    Scene();
    ~Scene() = default;

    void setupRoomScene();
    const SceneData& getSceneData() const { return sceneData; }
    
    // New methods for dynamic light control
    void setLightPosition(const Vec3& pos);
    Vec3 getLightPosition() const { return sceneData.lightPosition; }
    void updateLight(float deltaTime, bool moveUp, bool moveDown, bool moveLeft, bool moveRight);

private:
    SceneData sceneData;
    void addSphere(const Vec3& center, float radius, const Material& material);
    void addPlane(const Vec3& point, const Vec3& normal, const Material& material);
};

#endif