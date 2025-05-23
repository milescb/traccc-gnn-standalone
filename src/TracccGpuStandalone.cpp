#include "TracccGpuStandalone.hpp"

void TracccGpuStandalone::initialize()
{
    // HACK: hard code location of detector and digitization file
    m_detector_opts.detector_file = "/global/cfs/projectdirs/m3443/data/GNN4ITK-traccc/ITk_data/ATLAS-P2-RUN4-03-00-00/ITk_DetectorBuilder_geometry.json";
    m_detector_opts.digitization_file = "/global/cfs/projectdirs/m3443/data/GNN4ITK-traccc/ITk_data/ATLAS-P2-RUN4-03-00-00/ITk_digitization_config_with_strips.json";
    m_detector_opts.grid_file = "/global/cfs/projectdirs/m3443/data/GNN4ITK-traccc/ITk_data/ATLAS-P2-RUN4-03-00-00/ITk_DetectorBuilder_surface_grids.json";
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
traccc::track_state_container_types::host TracccGpuStandalone::fitFromGnnOutput(
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
    // TODO: instead use first three measurments and create seeds from this
    // Then, pass to the parameter estimation algo
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

void TracccGpuStandalone::inputDataToTracccMeasurements(
    InputData gnnTracks,
    traccc::edm::spacepoint_collection::host& spacepoints,
    traccc::measurement_collection_types::host& measurements) 
{
    // Create measurements collection
    measurements = traccc::measurement_collection_types::host(&m_host_mr);
    measurements.reserve(gnnTracks.cl_x.size());

    // Create spacepoints collection
    spacepoints = traccc::edm::spacepoint_collection::host(m_host_mr);
    spacepoints.reserve(gnnTracks.cl_x.size());

    // First create all measurements since we need them for spacepoint linking
    for (size_t i = 0; i < gnnTracks.cl_x.size(); i++) {
        traccc::measurement meas;
        // Set local coordinates (eta, phi)
        meas.local = {gnnTracks.cl_loc_eta[i], gnnTracks.cl_loc_phi[i]};
        // Set variance
        meas.variance = {gnnTracks.cl_cov_00[i], gnnTracks.cl_cov_11[i]};
        // Set measurement ID (needed for linking)
        meas.measurement_id = i;
        // Set measurement dimension
        meas.meas_dim = 2;
        // // Set geometry ID
        // meas.surface_link = 0; // TODO: need some meaningful number
        
        measurements.push_back(meas);
    }

    // Now create spacepoints with measurement links
    for (size_t i = 0; i < gnnTracks.sp_x.size(); i++) {
        // Get measurement indices for this spacepoint
        unsigned int meas_idx1 = gnnTracks.sp_cl1_index.at(i);
        unsigned int meas_idx2 = gnnTracks.sp_cl2_index[i] >= 0 ? 
            gnnTracks.sp_cl2_index[i] : 
            traccc::edm::spacepoint_collection::host::INVALID_MEASUREMENT_INDEX;

        // Create spacepoint using same format as read_spacepoints
        spacepoints.push_back({
            meas_idx1,                              // First measurement index
            meas_idx2,                              // Second measurement index (or INVALID)
            {gnnTracks.sp_x[i],                     // Global position
             gnnTracks.sp_y[i], 
             gnnTracks.sp_z[i]},
            0.f,                                    // Radius
            0.f                                     // Phi
        });
    }

    // print number of spacepoints / measurements
    std::cout << "Number of measurements created: " << measurements.size() << std::endl;
    std::cout << "Number of spacepoints created: " << spacepoints.size() << std::endl;
}


int main(int argc, char *argv[])
{

    TracccGpuStandalone traccc_gpu(0);
    InputData gnnTracks;
    traccc_gpu.execute(gnnTracks);

    return 0;
}