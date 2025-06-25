#include "Scene.h"

Scene::Scene() {
    sceneData.numSpheres = 0;
    sceneData.numPlanes = 0;
    sceneData.lightPosition = Vec3(2.0f, 4.0f, 2.0f);
    sceneData.lightColor = Vec3(1.0f, 1.0f, 1.0f);
}

void Scene::setupRoomScene() {
    // Room walls (planes)
    // Floor
    addPlane(Plane(Vec3(0, -2, 0), Vec3(0, 1, 0), 
                   Material(Vec3(0.8f, 0.8f, 0.8f), DIFFUSE, 0.8f)));
    
    // Ceiling
    addPlane(Plane(Vec3(0, 6, 0), Vec3(0, -1, 0), 
                   Material(Vec3(0.9f, 0.9f, 0.9f), DIFFUSE, 0.9f)));
    
    // Back wall
    addPlane(Plane(Vec3(0, 0, -5), Vec3(0, 0, 1), 
                   Material(Vec3(0.7f, 0.7f, 0.9f), DIFFUSE, 0.7f)));
    
    // Left wall
    addPlane(Plane(Vec3(-5, 0, 0), Vec3(1, 0, 0), 
                   Material(Vec3(0.9f, 0.7f, 0.7f), DIFFUSE, 0.8f)));
    
    // Right wall
    addPlane(Plane(Vec3(5, 0, 0), Vec3(-1, 0, 0), 
                   Material(Vec3(0.7f, 0.9f, 0.7f), DIFFUSE, 0.8f)));

    // Spheres with different materials
    // Large reflective sphere
    addSphere(Sphere(Vec3(-1.5f, 0.0f, -1.0f), 1.0f, 
                     Material(Vec3(0.9f, 0.9f, 0.9f), REFLECTIVE, 0.1f, 1.0f, 0.9f)));
    
    // Medium refractive sphere (glass)
    addSphere(Sphere(Vec3(1.5f, -0.5f, 0.0f), 0.8f, 
                     Material(Vec3(0.9f, 0.9f, 1.0f), REFRACTIVE, 0.0f, 1.5f, 0.1f)));
    
    // Small diffuse spheres
    addSphere(Sphere(Vec3(0.0f, -1.0f, 1.5f), 0.5f, 
                     Material(Vec3(0.8f, 0.3f, 0.3f), DIFFUSE, 0.9f)));
    
    addSphere(Sphere(Vec3(-2.5f, -1.0f, 0.5f), 0.6f, 
                     Material(Vec3(0.3f, 0.8f, 0.3f), DIFFUSE, 0.8f)));
    
    addSphere(Sphere(Vec3(2.5f, -1.2f, -2.0f), 0.4f, 
                     Material(Vec3(0.3f, 0.3f, 0.8f), DIFFUSE, 0.7f)));
    
    // Smaller reflective spheres
    addSphere(Sphere(Vec3(0.8f, 0.5f, -2.5f), 0.3f, 
                     Material(Vec3(1.0f, 0.8f, 0.3f), REFLECTIVE, 0.2f, 1.0f, 0.7f)));
    
    addSphere(Sphere(Vec3(-0.8f, 1.2f, 0.8f), 0.25f, 
                     Material(Vec3(0.8f, 0.3f, 0.8f), REFLECTIVE, 0.3f, 1.0f, 0.6f)));
}

void Scene::addSphere(const Sphere& sphere) {
    if (sceneData.numSpheres < MAX_SPHERES) {
        sceneData.spheres[sceneData.numSpheres++] = sphere;
    }
}

void Scene::addPlane(const Plane& plane) {
    if (sceneData.numPlanes < MAX_PLANES) {
        sceneData.planes[sceneData.numPlanes++] = plane;
    }
}