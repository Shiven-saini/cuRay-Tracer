#include <GL/glew.h>  // MUST be included FIRST
#include "RayTracer.h"
#include "cuda/raytracing_kernel.h"
#include "cuda/cuda_utils.h"
#include <iostream>
#include <vector>

RayTracer::RayTracer(int width, int height)
    : width(width), height(height), textureID(0), pbo(0), shaderProgram(0),
      VAO(0), VBO(0), EBO(0), cudaPboResource(nullptr), d_output(nullptr), d_sceneData(nullptr) {}

RayTracer::~RayTracer() {
    cleanup();
}

bool RayTracer::initialize() {
    std::cout << "Initializing RayTracer..." << std::endl;
    
    if (!initializeGL()) {
        std::cerr << "Failed to initialize OpenGL resources" << std::endl;
        return false;
    }
    std::cout << "OpenGL resources initialized successfully" << std::endl;
    
    if (!initializeCUDA()) {
        std::cerr << "Failed to initialize CUDA resources" << std::endl;
        return false;
    }
    std::cout << "CUDA resources initialized successfully" << std::endl;
    
    return true;
}

bool RayTracer::initializeGL() {
    const char* version = (const char*)glGetString(GL_VERSION);
    const char* vendor = (const char*)glGetString(GL_VENDOR);
    
    std::cout << "OpenGL Version: " << (version ? version : "Unknown") << std::endl;
    std::cout << "OpenGL Vendor: " << (vendor ? vendor : "Unknown") << std::endl;
    std::cout << "Display Server: " << (getenv("WAYLAND_DISPLAY") ? "Wayland" : "X11") << std::endl;
    
    // Create modern OpenGL resources
    if (!createShaderProgram()) {
        return false;
    }
    
    if (!createQuadGeometry()) {
        return false;
    }
    
    // Create texture
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);
    
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    // Create initial test pattern
    std::vector<unsigned char> testData(width * height * 3);
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 3;
            // Create a simple test pattern
            if (x < width/2 && y < height/2) {
                testData[idx + 0] = 255; testData[idx + 1] = 0; testData[idx + 2] = 0; // Red
            } else if (x >= width/2 && y < height/2) {
                testData[idx + 0] = 0; testData[idx + 1] = 255; testData[idx + 2] = 0; // Green
            } else if (x < width/2 && y >= height/2) {
                testData[idx + 0] = 0; testData[idx + 1] = 0; testData[idx + 2] = 255; // Blue
            } else {
                testData[idx + 0] = 255; testData[idx + 1] = 255; testData[idx + 2] = 0; // Yellow
            }
        }
    }
    
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, testData.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    
    std::cout << "Test texture created successfully" << std::endl;
    
    return glGetError() == GL_NO_ERROR;
}

bool RayTracer::createShaderProgram() {
    // Vertex shader source
    const char* vertexShaderSource = R"(
        #version 330 core
        layout (location = 0) in vec2 aPos;
        layout (location = 1) in vec2 aTexCoord;
        
        out vec2 TexCoord;
        
        void main() {
            gl_Position = vec4(aPos, 0.0, 1.0);
            TexCoord = aTexCoord;
        }
    )";
    
    // Fragment shader source
    const char* fragmentShaderSource = R"(
        #version 330 core
        out vec4 FragColor;
        
        in vec2 TexCoord;
        uniform sampler2D ourTexture;
        
        void main() {
            FragColor = texture(ourTexture, TexCoord);
        }
    )";
    
    // Compile vertex shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);
    
    GLint success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "Vertex shader compilation failed: " << infoLog << std::endl;
        return false;
    }
    
    // Compile fragment shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);
    
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "Fragment shader compilation failed: " << infoLog << std::endl;
        return false;
    }
    
    // Create shader program
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "Shader program linking failed: " << infoLog << std::endl;
        return false;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    std::cout << "Shader program created successfully" << std::endl;
    return true;
}

bool RayTracer::createQuadGeometry() {
    // Define quad vertices (position + texture coordinates)
    float vertices[] = {
        // positions   // texture coords
        -1.0f, -1.0f,  0.0f, 1.0f, // bottom left
         1.0f, -1.0f,  1.0f, 1.0f, // bottom right
         1.0f,  1.0f,  1.0f, 0.0f, // top right
        -1.0f,  1.0f,  0.0f, 0.0f  // top left
    };
    
    unsigned int indices[] = {
        0, 1, 2,
        2, 3, 0
    };
    
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);
    
    glBindVertexArray(VAO);
    
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);
    
    // Position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    
    // Texture coordinate attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    
    glBindVertexArray(0);
    
    std::cout << "Quad geometry created successfully" << std::endl;
    return true;
}

bool RayTracer::initializeCUDA() {
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_output, width * height * 3 * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_sceneData, sizeof(SceneData)));
    
    std::cout << "CUDA memory allocated successfully" << std::endl;
    return true;
}

void RayTracer::render(const Camera& camera, const Scene& scene) {
    static int frameCount = 0;
    static std::vector<unsigned char> hostOutput(width * height * 3);
    frameCount++;
    
    if (frameCount > 60) {
        // After 60 frames, switch to ray tracing
        if (frameCount == 61) {
            std::cout << "Switching to CUDA ray tracing..." << std::endl;
        }
        
        // Copy scene data to device
        CUDA_CHECK(cudaMemcpy(d_sceneData, &scene.getSceneData(), sizeof(SceneData), cudaMemcpyHostToDevice));
        
        // Get camera parameters
        Vec3 cameraPos = camera.getPosition();
        Vec3 cameraDir = camera.getDirection();
        Vec3 cameraUp(0, 1, 0);
        Vec3 cameraRight = cameraDir.cross(cameraUp).normalize();
        cameraUp = cameraRight.cross(cameraDir).normalize();
        
        if (frameCount % 120 == 0) {
            std::cout << "Frame " << frameCount << " - Camera: (" 
                      << cameraPos.x << ", " << cameraPos.y << ", " << cameraPos.z << ")" << std::endl;
        }
        
        // Launch CUDA kernel
        launchRayTracingKernel((unsigned char*)d_output, width, height, d_sceneData,
                              cameraPos, cameraDir, cameraUp, cameraRight, 45.0f);
        
        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << std::endl;
            return;
        }
        
        // Copy result back to host
        CUDA_CHECK(cudaMemcpy(hostOutput.data(), d_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));
        
        // Debug output
        if (frameCount == 61) {
            int nonZeroPixels = 0;
            for (int i = 0; i < width * height * 3; i++) {
                if (hostOutput[i] > 0) nonZeroPixels++;
            }
            std::cout << "Non-zero pixels in output: " << nonZeroPixels << " / " << (width * height * 3) << std::endl;
            
            std::cout << "First 10 RGB values: ";
            for (int i = 0; i < 30 && i < hostOutput.size(); i += 3) {
                std::cout << "(" << (int)hostOutput[i] << "," << (int)hostOutput[i+1] << "," << (int)hostOutput[i+2] << ") ";
            }
            std::cout << std::endl;
        }
        
        // Update texture
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, hostOutput.data());
        glBindTexture(GL_TEXTURE_2D, 0);
    }
}

void RayTracer::renderQuad() {
    // Use shader program
    glUseProgram(shaderProgram);
    
    // Bind texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureID);
    glUniform1i(glGetUniformLocation(shaderProgram, "ourTexture"), 0);
    
    // Render quad
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);
    
    // Unbind
    glUseProgram(0);
}

void RayTracer::resize(int newWidth, int newHeight) {
    width = newWidth;
    height = newHeight;
    cleanup();
    initialize();
}

void RayTracer::cleanup() {
    if (d_output) {
        cudaFree(d_output);
        d_output = nullptr;
    }
    if (d_sceneData) {
        cudaFree(d_sceneData);
        d_sceneData = nullptr;
    }
    if (textureID) {
        glDeleteTextures(1, &textureID);
        textureID = 0;
    }
    if (shaderProgram) {
        glDeleteProgram(shaderProgram);
        shaderProgram = 0;
    }
    if (VAO) {
        glDeleteVertexArrays(1, &VAO);
        VAO = 0;
    }
    if (VBO) {
        glDeleteBuffers(1, &VBO);
        VBO = 0;
    }
    if (EBO) {
        glDeleteBuffers(1, &EBO);
        EBO = 0;
    }
}