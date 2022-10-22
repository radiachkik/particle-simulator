#ifndef LIFE_SIMULATOR_SIMULATION_HPP
#define LIFE_SIMULATOR_SIMULATION_HPP

namespace simulation {
    typedef struct {
        const unsigned long points_per_cloud;
        const unsigned short num_point_clouds;
        const float *gravities;
        const float *distance_thresholds;
        bool border_enabled;
    } simulationConfig;

    void initialize_simulation(simulationConfig *simulation_config);
    void suspend_simulation();
    void simulate_next_frame();
    float *get_coordinates();
}

#endif //LIFE_SIMULATOR_SIMULATION_HPP
