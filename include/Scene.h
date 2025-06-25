#ifndef SCENE_H
#define SCENE_H

#include "math/Vec3.h"
#include "primitives/Sphere.h"
#include "primitives/Plane.h"
#include <vector>

#define MAX_SPHERES 50
#define MAX_PLANES 10

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

private:
    SceneData sceneData;
    void addSphere(const Sphere& sphere);
    void addPlane(const Plane& plane);
};

#endif