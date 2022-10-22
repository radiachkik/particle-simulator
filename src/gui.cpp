#include <GL/glut.h>
#include "../include/gui.hpp"
#include <chrono>
#include <thread>

using namespace std;

namespace gui {
    namespace {
        const int window_width = 800;
        const int window_height = 800;
        const int window_position_x = 200;
        const int window_position_y = 200;
        const char* window_title = "Life Simulator";
        const unsigned short frames_per_second = 60;

        simulation::simulationConfig *config;

        const int colors[4][3] = {
                {0, 255, 0},
                {255, 0, 255},
                {255, 255, 255},
                {255, 255, 255}
        };


        void initGL() {
            // Set "clearing" or background color
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black and opaque
        }

        [[noreturn]] void display2d() {
            chrono::milliseconds delay_between_frames = chrono::milliseconds(1000 / frames_per_second);
            auto timestamp = chrono::system_clock::now();
            while (true) {
                glClear(GL_COLOR_BUFFER_BIT);   // Clear the color buffer with current clearing color
                glPointSize(2.0);
                glBegin(GL_POINTS);

                float *coordinates = simulation::get_coordinates();
                for (auto pc_index = 0; pc_index < config->num_point_clouds; pc_index++) {
                    glColor3ub(colors[pc_index][0], colors[pc_index][1], colors[pc_index][2]);
                    float *pc_coordinates = coordinates + pc_index * config->points_per_cloud * 2;
                    for (auto p_index = 0; p_index < config->points_per_cloud; p_index++) {
                        glVertex2fv(pc_coordinates + p_index * 2);
                    }
                }


                glEnd();
                auto passed_time = chrono::system_clock::now() - timestamp;

                if (passed_time < delay_between_frames) {
                    this_thread::sleep_for(delay_between_frames - passed_time);
                }
                glFlush();  // Render now
                simulation::simulate_next_frame();
            }
        }
    }


    void setup_window(int argc, char** argv) {
        glutInit(&argc, argv);                 // Initialize GLUT
        glutInitWindowSize(window_width, window_height);   // Set the window's initial width & height
        glutInitWindowPosition(window_position_x, window_position_y); // Position the window's initial top-left corner
        glutCreateWindow(window_title); // Create a window with the given title
    }

    void run_animation(simulation::simulationConfig *sim_config) {
        config = sim_config;
        glutDisplayFunc(display2d);
        initGL();
        glutMainLoop();
    }
}
