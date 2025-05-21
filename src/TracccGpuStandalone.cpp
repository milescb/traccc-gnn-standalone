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
    auto track_states_host = m_copy_track_states(track_states);
    // run ambiguity resolution
    // TODO: make this optional? 
    traccc::track_state_container_types::host resolved_track_states_cuda =
        m_resolution_alg(track_states_host);

    return resolved_track_states_cuda;
}

traccc::edm::spacepoint_collection::host TracccGpuStandalone::read_spacepoints(
    std::vector<clusterInfo>& detray_clusters, bool do_strip)
{
    traccc::edm::spacepoint_collection::host spacepoints{m_host_mr};
    traccc::measurement_collection_types::host measurements =
        read_measurements(detray_clusters, do_strip);
  
    std::map<traccc::geometry_id, unsigned int> m;
    for(std::vector<clusterInfo>::size_type i = 0; i < detray_clusters.size();i++){
        clusterInfo cluster = detray_clusters[i];
        if(do_strip && cluster.pixel){continue;}

        // Construct the local 3D(2D) position of the measurement.
        // traccc::measurement meas;
        // meas = measurements[i];

        spacepoints.push_back({static_cast<unsigned int>(i), {
            static_cast<float>(cluster.globalPosition[0]),
            static_cast<float>(cluster.globalPosition[1]),
            static_cast<float>(cluster.globalPosition[2])},
            0.f, 0.f});
    }

    // // Verify values of spacepoints for the first spacepoint
    // if (spacepoints.size() > 0) {
    //     const auto& first_spacepoint = spacepoints[0];
    //     std::cout << "First spacepoint: "
    //             << "global[0]: " << first_spacepoint.global[0]
    //             << ", global[1]: " << first_spacepoint.global[1]
    //             << ", global[2]: " << first_spacepoint.global[2]
    //             << std::endl;
    // }

    return spacepoints;
}

traccc::measurement_collection_types::host TracccGpuStandalone::read_measurements(
    std::vector<clusterInfo>& detray_clusters, bool do_strip)
{
    traccc::measurement_collection_types::host measurements;

    //! two more pointless lines?
    std::map<traccc::geometry_id, unsigned int> m;
    std::multimap<uint64_t, detray::geometry::barcode> sf_seen;

    for(std::vector<clusterInfo>::size_type i = 0; i < detray_clusters.size();i++)
    {

        clusterInfo cluster = detray_clusters[i];
        if(do_strip && cluster.pixel) {continue;}

        uint64_t geometry_id = cluster.detray_id;
        const auto& sf = detray::geometry::barcode{geometry_id};
        const detray::tracking_surface surface{*m_detector, sf}; //! unused variable?
        //! pointless lines?
        cluster.localPosition[0] = cluster.localPosition.x();
        cluster.localPosition[1] = cluster.localPosition.y();

        // Construct the measurement object.
        traccc::measurement meas;
        std::array<detray::dsize_type<traccc::default_algebra>, 2u> indices{0u, 0u};
        meas.meas_dim = 0u;
        for (unsigned int ipar = 0; ipar < 2u; ++ipar) 
        {
            if (((cluster.local_key) & (1 << (ipar + 1))) != 0) 
            {
                switch (ipar) {
                    case 0: {
                        meas.local[0] = cluster.localPosition.x();
                        meas.variance[0] = 0.0025;
                        indices[meas.meas_dim++] = ipar;
                    }; break;
                    case 1: {
                        meas.local[1] = cluster.localPosition.y();
                        meas.variance[1] = 0.0025;
                        indices[meas.meas_dim++] = ipar;
                    }; break;
                }
            }
        }

        meas.subs.set_indices(indices);
        meas.surface_link = detray::geometry::barcode{geometry_id};

        // Keeps measurement_id for ambiguity resolution
        meas.measurement_id = i;
        measurements.push_back(meas);
    }

    // verify values of measurements for first meas
    if (measurements.size() > 0) {
        std::cout << "First measurement: "
                  << "local[0]: " << measurements[0].local[0]
                  << ", local[1]: " << measurements[0].local[1]
                  << ", variance[0]: " << measurements[0].variance[0]
                  << ", variance[1]: " << measurements[0].variance[1]
                  << std::endl;
    }

    return measurements;
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

    // run rest of chain...

    return 0;
}