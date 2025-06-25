#include "Scene.h"
#include <iostream>

Scene::Scene() {
    sceneData.numSpheres = 0;
    sceneData.numPlanes = 0;
    sceneData.lightPosition = Vec3(2.0f, 4.0f, 2.0f);
    sceneData.lightColor = Vec3(1.0f, 0.95f, 0.8f); // Warm white light
}

void Scene::setupRoomScene() {
    // Clear existing objects
    sceneData.numSpheres = 0;
    sceneData.numPlanes = 0;

    // Room walls (planes)
    
    // Floor (gray)
    Material floorMaterial;
    floorMaterial.type = DIFFUSE;
    floorMaterial.color = Vec3(0.7f, 0.7f, 0.7f);
    floorMaterial.roughness = 0.8f;
    addPlane(Vec3(0, 0, 0), Vec3(0, 1, 0), floorMaterial);

    // Ceiling (white)
    Material ceilingMaterial;
    ceilingMaterial.type = DIFFUSE;
    ceilingMaterial.color = Vec3(0.9f, 0.9f, 0.9f);
    ceilingMaterial.roughness = 0.7f;
    addPlane(Vec3(0, 8, 0), Vec3(0, -1, 0), ceilingMaterial);

    // Back wall (light blue)
    Material backWallMaterial;
    backWallMaterial.type = DIFFUSE;
    backWallMaterial.color = Vec3(0.6f, 0.8f, 1.0f);
    backWallMaterial.roughness = 0.6f;
    addPlane(Vec3(0, 0, -5), Vec3(0, 0, 1), backWallMaterial);

    // Left wall (red)
    Material leftWallMaterial;
    leftWallMaterial.type = DIFFUSE;
    leftWallMaterial.color = Vec3(1.0f, 0.3f, 0.3f);
    leftWallMaterial.roughness = 0.6f;
    addPlane(Vec3(-5, 0, 0), Vec3(1, 0, 0), leftWallMaterial);

    // Right wall (green)
    Material rightWallMaterial;
    rightWallMaterial.type = DIFFUSE;
    rightWallMaterial.color = Vec3(0.3f, 1.0f, 0.3f);
    rightWallMaterial.roughness = 0.6f;
    addPlane(Vec3(5, 0, 0), Vec3(-1, 0, 0), rightWallMaterial);

    // Spheres with different materials

    // Large mirror sphere (center)
    Material mirrorMaterial;
    mirrorMaterial.type = REFLECTIVE;
    mirrorMaterial.color = Vec3(0.9f, 0.9f, 1.0f);
    mirrorMaterial.reflectivity = 0.9f;
    mirrorMaterial.roughness = 0.0f;
    addSphere(Vec3(0, 1.5f, -2), 1.5f, mirrorMaterial);

    // Glass sphere (right)
    Material glassMaterial;
    glassMaterial.type = REFRACTIVE;
    glassMaterial.color = Vec3(0.9f, 1.0f, 0.9f);
    glassMaterial.reflectivity = 0.1f;
    glassMaterial.refractiveIndex = 1.5f;
    glassMaterial.roughness = 0.0f;
    addSphere(Vec3(2.5f, 1.0f, 0), 1.0f, glassMaterial);

    // Red diffuse sphere (left)
    Material redMaterial;
    redMaterial.type = DIFFUSE;
    redMaterial.color = Vec3(1.0f, 0.2f, 0.2f);
    redMaterial.roughness = 0.8f;
    addSphere(Vec3(-2.5f, 1.0f, 0), 1.0f, redMaterial);

    // Blue metallic sphere (front left)
    Material blueMaterial;
    blueMaterial.type = REFLECTIVE;
    blueMaterial.color = Vec3(0.2f, 0.4f, 1.0f);
    blueMaterial.reflectivity = 0.7f;
    blueMaterial.roughness = 0.1f;
    addSphere(Vec3(-1.5f, 0.7f, 2), 0.7f, blueMaterial);

    // Yellow sphere (front right)
    Material yellowMaterial;
    yellowMaterial.type = DIFFUSE;
    yellowMaterial.color = Vec3(1.0f, 1.0f, 0.2f);
    yellowMaterial.roughness = 0.6f;
    addSphere(Vec3(1.5f, 0.7f, 2), 0.7f, yellowMaterial);

    // Small purple sphere (high up)
    Material purpleMaterial;
    purpleMaterial.type = DIFFUSE;
    purpleMaterial.color = Vec3(0.8f, 0.2f, 1.0f);
    purpleMaterial.roughness = 0.7f;
    addSphere(Vec3(0, 3.5f, 1), 0.5f, purpleMaterial);

    // Orange sphere (back corner)
    Material orangeMaterial;
    orangeMaterial.type = DIFFUSE;
    orangeMaterial.color = Vec3(1.0f, 0.6f, 0.1f);
    orangeMaterial.roughness = 0.5f;
    addSphere(Vec3(3, 0.8f, -3), 0.8f, orangeMaterial);

    std::cout << "Room scene created with " << sceneData.numSpheres << " spheres and " 
              << sceneData.numPlanes << " planes" << std::endl;
    std::cout << "Initial light position: (" << sceneData.lightPosition.x << ", " 
              << sceneData.lightPosition.y << ", " << sceneData.lightPosition.z << ")" << std::endl;
}

void Scene::setLightPosition(const Vec3& pos) {
    sceneData.lightPosition = pos;
}

void Scene::updateLight(float deltaTime, bool moveUp, bool moveDown, bool moveLeft, bool moveRight) {
    float lightSpeed = 5.0f; // units per second
    float movement = lightSpeed * deltaTime;
    
    if (moveUp) {    // W key
        sceneData.lightPosition.z -= movement;
    }
    if (moveDown) {  // S key
        sceneData.lightPosition.z += movement;
    }
    if (moveLeft) {  // A key
        sceneData.lightPosition.x -= movement;
    }
    if (moveRight) { // D key
        sceneData.lightPosition.x += movement;
    }
    
    // Keep light above ground
    if (sceneData.lightPosition.y < 1.0f) {
        sceneData.lightPosition.y = 1.0f;
    }
    
    // Constrain light to room bounds
    sceneData.lightPosition.x = fmaxf(-4.5f, fminf(4.5f, sceneData.lightPosition.x));
    sceneData.lightPosition.z = fmaxf(-4.5f, fminf(4.5f, sceneData.lightPosition.z));
}

void Scene::addSphere(const Vec3& center, float radius, const Material& material) {
    if (sceneData.numSpheres < MAX_SPHERES) {
        sceneData.spheres[sceneData.numSpheres] = Sphere(center, radius, material);
        sceneData.numSpheres++;
    }
}

void Scene::addPlane(const Vec3& point, const Vec3& normal, const Material& material) {
    if (sceneData.numPlanes < MAX_PLANES) {
        sceneData.planes[sceneData.numPlanes] = Plane(point, normal, material);
        sceneData.numPlanes++;
    }
}