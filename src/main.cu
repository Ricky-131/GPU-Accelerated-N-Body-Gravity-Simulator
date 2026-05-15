#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cmath>

#include "../include/imgui.h"
#include "../include/imgui_impl_glfw.h"
#include "../include/imgui_impl_opengl3.h"

#define SOFTENING 1e-2f

int n = 4096;
float dt = 0.0005f;
GLuint vbo;
float3 *d_v;
cudaGraphicsResource *res;

const char* vertexShaderSource = "#version 130\n"
    "in vec4 position;\n"
    "out vec3 vColor;\n"
    "void main() {\n"
    "    gl_Position = vec4(position.xyz, 1.0);\n"
    "    float speed = position.w * 15.0;\n" // Use w as speed
    "    vColor = mix(vec3(0.1, 0.4, 1.0), vec3(1.0, 0.8, 0.2), clamp(speed, 0.0, 1.0));\n"
    "}\0";

const char* fragmentShaderSource = "#version 130\n"
    "in vec3 vColor;\n"
    "out vec4 fragColor;\n"
    "void main() {\n"
    "    fragColor = vec4(vColor, 1.0);\n"
    "}\0";

__global__ void interactBodies(float4 *p, float3 *v, float dt, int n) {
    extern __shared__ float4 shPos[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float3 accel = {0.0f, 0.0f, 0.0f};
    float4 iPos = (i < n) ? p[i] : make_float4(0, 0, 0, 0);

    for (int tile = 0; tile < gridDim.x; tile++) {
        int idx = tile * blockDim.x + threadIdx.x;
        shPos[threadIdx.x] = (idx < n) ? p[idx] : make_float4(0, 0, 0, 0);
        __syncthreads();
        for (int j = 0; j < blockDim.x; j++) {
            float4 jPos = shPos[j];
            float3 r = {jPos.x - iPos.x, jPos.y - iPos.y, jPos.z - iPos.z};
            float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float s = 0.1f * (invDist * invDist * invDist); // Using fixed mass for math
            accel.x += r.x * s; accel.y += r.y * s; accel.z += r.z * s;
        }
        __syncthreads();
    }

    if (i < n) {
        v[i].x += accel.x * dt; v[i].y += accel.y * dt; v[i].z += accel.z * dt;
        p[i].x += v[i].x * dt; p[i].y += v[i].y * dt; p[i].z += v[i].z * dt;
        // Store magnitude in W for the shader to use as color index
        p[i].w = sqrtf(v[i].x * v[i].x + v[i].y * v[i].y + v[i].z * v[i].z);
    }
}

GLuint compileShader() {
    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vertexShaderSource, NULL);
    glCompileShader(vs);
    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fragmentShaderSource, NULL);
    glCompileShader(fs);
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs); glAttachShader(prog, fs);
    glLinkProgram(prog);
    return prog;
}

void initSimulation() {
    float4* h_p = new float4[n];
    float3* h_v_init = new float3[n];
    for (int i = 0; i < n; i++) {
        float angle = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
        float r = ((float)rand() / RAND_MAX) * 0.5f + 0.1f;
        h_p[i] = { r * cosf(angle), r * sinf(angle), 0.0f, 0.0f };
        h_v_init[i] = { -sinf(angle) * 0.07f, cosf(angle) * 0.07f, 0.0f };
    }
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, n * sizeof(float4), h_p, GL_DYNAMIC_DRAW);
    cudaMemcpy(d_v, h_v_init, n * sizeof(float3), cudaMemcpyHostToDevice);
    delete[] h_p; delete[] h_v_init;
}

int main() {
    if (!glfwInit()) return -1;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    GLFWwindow* window = glfwCreateWindow(1200, 800, "CUDA Galaxy Engine", NULL, NULL);
    glfwMakeContextCurrent(window);
    glewInit();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 130");

    GLuint shaderProg = compileShader();
    glGenBuffers(1, &vbo);
    cudaMalloc(&d_v, n * sizeof(float3));
    initSimulation();
    cudaGraphicsGLRegisterBuffer(&res, vbo, cudaGraphicsRegisterFlagsNone);

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Engine Controls");
        if (ImGui::Button("Reset Galaxy")) initSimulation();
        ImGui::SliderFloat("Time Step", &dt, 0.0001f, 0.005f);
        ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
        ImGui::End();

        float4 *d_p; size_t size;
        cudaGraphicsMapResources(1, &res);
        cudaGraphicsResourceGetMappedPointer((void**)&d_p, &size, res);
        interactBodies<<<(n+255)/256, 256, 256*sizeof(float4)>>>(d_p, d_v, dt, n);
        cudaDeviceSynchronize();
        cudaGraphicsUnmapResources(1, &res);

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClearColor(0.01f, 0.01f, 0.02f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(shaderProg);
        glPointSize(1.5f);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        GLuint posAttrib = glGetAttribLocation(shaderProg, "position");
        glEnableVertexAttribArray(posAttrib);
        glVertexAttribPointer(posAttrib, 4, GL_FLOAT, GL_FALSE, 0, 0);
        glDrawArrays(GL_POINTS, 0, n);
        glUseProgram(0);

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    cudaFree(d_v);
    glfwTerminate();
    return 0;
}