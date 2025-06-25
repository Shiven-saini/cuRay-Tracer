#ifndef WINDOW_H
#define WINDOW_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>

class Window {
public:
    Window(int width, int height, const std::string& title);
    ~Window();

    bool initialize();
    bool shouldClose() const;
    void swapBuffers();
    void pollEvents();
    
    GLFWwindow* getGLFWWindow() const { return window; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }

    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
    static void errorCallback(int error, const char* description);

private:
    GLFWwindow* window;
    int width, height;
    std::string title;
};

#endif