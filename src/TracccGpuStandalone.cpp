#include "TracccGpuStandalone.hpp"

void TracccGpuStandalone::initialize()
{
    // HACK: hard code location of detector and digitization file
    m_detector_opts.detector_file = "../itk_configuration/ITk_DetectorBuilder_geometry.json";
    m_detector_opts.digitization_file = "../itk_configuration/ITk_digitization_config_with_strips.json";
    m_detector_opts.grid_file = "../itk_configuration/ITk_DetectorBuilder_surface_grids.json";
    m_detector_opts.material_file = "";

    // Read the detector description
    traccc::io::read_detector_description(
        m_det_descr, m_detector_opts.detector_file,
        m_detector_opts.digitization_file, traccc::data_format::json);
    traccc::silicon_detector_description::data m_det_descr_data{
        vecmem::get_data(m_det_descr)};
    m_device_det_descr = traccc::silicon_detector_description::buffer(
            static_cast<traccc::silicon_detector_description::buffer::size_type>(
                m_det_descr.size()),
            m_device_mr);
    m_copy.setup(m_device_det_descr)->wait();
    m_copy(m_det_descr_data, m_device_det_descr)->wait();

    // Create the detector and read the configuration file
    m_detector = std::make_unique<host_detector_type>(m_host_mr);
    traccc::io::read_detector(
        *m_detector, m_host_mr, m_detector_opts.detector_file,
        m_detector_opts.material_file, m_detector_opts.grid_file);
    
    // copy it to the device - dereference the unique_ptr to get the actual object
    m_device_detector = detray::get_buffer(*m_detector, m_device_mr, m_copy);
    m_stream.synchronize();
    m_device_detector_view = detray::get_data(m_device_detector);

    return;
}

// input has to be _clusters_ now
// TODO: could create spacepoints from measurements here
// TODO: measurement sorting using same strategy: first measurements to spacepoints
traccc::track_state_container_types::host TracccGpuStandalone::run(
    traccc::edm::spacepoint_collection::host spacepoints_per_event,
    traccc::measurement_collection_types::host measurements_per_event)
{   
    // copy spacepoints and measurements to device
    traccc::edm::spacepoint_collection::buffer spacepoints(
        static_cast<unsigned int>(spacepoints_per_event.size()), *m_cached_device_mr);
    m_copy(vecmem::get_data(spacepoints_per_event), spacepoints)->wait();

    traccc::measurement_collection_types::buffer measurements(
        static_cast<unsigned int>(measurements_per_event.size()), *m_cached_device_mr);
    m_copy(vecmem::get_data(measurements_per_event), measurements)->wait();

    // Seeding and parameter estimation
    auto seeds = m_seeding(spacepoints);
    m_stream.synchronize();

    const traccc::cuda::track_params_estimation::output_type track_params =
        m_track_parameter_estimation(measurements, spacepoints,
            seeds, m_field_vec);
    m_stream.synchronize();

    // track finding                        
    const finding_algorithm::output_type track_candidates = m_finding(
        m_device_detector_view, m_field, measurements, track_params);

    m_stream.synchronize();
    std::cout << "Track finding complete" << std::endl;

    // Run the track fitting
    const fitting_algorithm::output_type track_states = 
        m_fitting(m_device_detector_view, m_field, track_candidates);


    // copy track states to host
    traccc::track_state_container_types::host track_states_host = m_copy_track_states(track_states);

    return track_states_host;
}


int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        std::cout << "Not enough arguments, minimum requirement two of the form: " << std::endl;
        std::cout << argv[0] << " <event_file> " << "<deviceID>" << std::endl;
        return -1;
    }

    std::string event_file = std::string(argv[1]);
    int deviceID = std::stoi(argv[2]);

    std::cout << "Using device ID: " << deviceID << std::endl;
    std::cout << "Running " << argv[0] << " on " << event_file << std::endl;

    TracccGpuStandalone traccc_gpu(deviceID);

    // do conversion and run chain...

    return 0;
}