#include "../include/device_management.hpp"
#include "../include/simulation.hpp"
#include "../include/gui.hpp"

simulation::simulationConfig config = {
        15000,
        4,
        new float[16]{
                4.0, 7.50, 0.00, 9.00,
                -28.0, 19.0, -17.0, -57.5,
                -10, 0, -1.0, 0,
                -28.0, 16.0, -40.0, -6.5
        },
        new float[16]{
                61.45, 55.325, 126, 59,
                261, 56.0, 183.0, 62.5,
                79, 77, 63, 154,
                144, 72, 154, 50
        },
        true
};

int main(int argc, char *argv[]) {
    set_cuda_device(0);
    initialize_simulation(&config);
    gui::setup_window(argc, argv);
    gui::run_animation(&config);
    simulation::suspend_simulation();
    return 0;
}