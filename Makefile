NVCC = nvcc
CFLAGS = -I./include -I./external
LIBS = -lGL -lGLEW -lglfw
OFFLOAD = __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia

SRCS = src/main.cu \
       external/imgui.cpp \
       external/imgui_draw.cpp \
       external/imgui_widgets.cpp \
       external/imgui_tables.cpp \
       external/imgui_impl_glfw.cpp \
       external/imgui_impl_opengl3.cpp

all:
	mkdir -p build
	$(NVCC) $(CFLAGS) $(SRCS) -o build/nbody_viz $(LIBS)

run:
	$(OFFLOAD) ./build/nbody_viz

clean:
	rm -rf build/