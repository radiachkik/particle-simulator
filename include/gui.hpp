#ifndef LIFE_SIMULATOR_GUI_HPP
#define LIFE_SIMULATOR_GUI_HPP

#include "simulation.hpp"

namespace gui {
    void setup_window(int argc, char** argv);
    void run_animation(simulation::simulationConfig *sim_config);
}

#endif //LIFE_SIMULATOR_GUI_HPP
