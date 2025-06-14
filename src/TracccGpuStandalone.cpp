#include "TracccGpuStandalone.hpp"
#include "TracccEdmConversion.hpp"

void TracccGpuStandalone::initialize()
{
    // Set geometry files
    m_detector_opts.detector_file = m_geoDir + "ITk_DetectorBuilder_geometry.json";
    m_detector_opts.digitization_file = m_geoDir + "ITk_digitization_config_with_strips.json";
    m_detector_opts.grid_file = m_geoDir + "ITk_DetectorBuilder_surface_grids.json";
    m_detector_opts.material_file = ""; 

    // read geometry and create source to barcode map for use in the edm
    auto geometry_pair = traccc::io::read_geometry(
        m_detector_opts.detector_file, traccc::data_format::json);
    m_surface_transforms = std::move(geometry_pair.first);
    m_barcode_map = std::move(geometry_pair.second);  

    // Load Athena-to-ACTS mapping
    std::string athenaTransformsPath = m_geoDir + "athena_transforms.csv";
    m_athena_to_acts_map = read_athena_to_acts_mapping(athenaTransformsPath);

    // Read the detector description
    traccc::io::read_detector_description(
        m_det_descr, m_detector_opts.detector_file,
        m_detector_opts.digitization_file, traccc::data_format::json);
    traccc::silicon_detector_description::data m_det_descr_data{
        vecmem::get_data(m_det_descr)};
    m_device_det_descr = traccc::silicon_detector_description::buffer(
            static_cast<traccc::silicon_detector_description::buffer::size_type>(
                m_det_descr.size()),
            *m_device_mr);
    m_copy.setup(m_device_det_descr)->wait();
    m_copy(m_det_descr_data, m_device_det_descr)->wait();

    // Create the detector and read the configuration file
    m_detector = std::make_unique<host_detector_type>(*m_host_mr);
    traccc::io::read_detector(
        *m_detector, *m_host_mr, m_detector_opts.detector_file,
        m_detector_opts.material_file, m_detector_opts.grid_file);
    
    // copy it to the device - dereference the unique_ptr to get the actual object
    m_device_detector = detray::get_buffer(*m_detector, *m_device_mr, m_copy);
    m_stream.synchronize();
    m_device_detector_view = detray::get_data(m_device_detector);

    return;
}

traccc::track_state_container_types::host TracccGpuStandalone::fitFromGnnOutput(
    traccc::edm::spacepoint_collection::host spacepoints_per_event,
    traccc::measurement_collection_types::host measurements_per_event,
    std::vector<int> gnnOutputLabels)
{   
    // MARK: Sort spacepoints and gnnOutputLabels by radius
    std::vector<size_t> original_indices(spacepoints_per_event.size());
    std::iota(original_indices.begin(), original_indices.end(), 0);

    // Sort the indices based on the radius of the spacepoints
    std::sort(original_indices.begin(), original_indices.end(),
              [&](size_t a, size_t b) {
                  return spacepoints_per_event.at(a).radius() < spacepoints_per_event.at(b).radius();
              });
    
    traccc::edm::spacepoint_collection::host sorted_spacepoints(*m_host_mr);
    std::vector<int> sorted_gnnOutputLabels;
    sorted_spacepoints.reserve(spacepoints_per_event.size());
    sorted_gnnOutputLabels.reserve(gnnOutputLabels.size());

    for (size_t new_idx = 0; new_idx < original_indices.size(); ++new_idx) {
        size_t old_idx = original_indices.at(new_idx);
        sorted_spacepoints.push_back(spacepoints_per_event.at(old_idx));
        sorted_gnnOutputLabels.push_back(gnnOutputLabels.at(old_idx));
    }

    // MARK: unwind the GNN output labels to get the main particle and secondary particles
    auto nCandidates = *std::max_element(sorted_gnnOutputLabels.begin(), sorted_gnnOutputLabels.end());
    std::vector<std::vector<size_t>> spacepoints_per_particle(nCandidates+1);
    for (int i = 0; i < sorted_gnnOutputLabels.size(); ++i)
    {
        spacepoints_per_particle.at(sorted_gnnOutputLabels.at(i)).push_back(original_indices.at(i));
    }

    std::cout << "Number of candidates: " << nCandidates + 1 << std::endl;
    std::cout << "Number of spacepoints per particle: " << std::endl;
    for (size_t i = 0; i < spacepoints_per_particle.size(); ++i)
    {
        std::cout << "  Particle " << i << ": " << spacepoints_per_particle.at(i).size() << " spacepoints" << std::endl;
    }

    // MARK: Create seeds as first three spacepoint indices
    //? Could this be device friendly?
    traccc::edm::seed_collection::host seeds(*m_host_mr);
    std::vector<std::vector<size_t>> seed_indices;
    for (auto &particle : spacepoints_per_particle) {
        // need at least three spacepoints to form a seed
        if (particle.size() < 3) {
            continue;
        }

        // because already sorted, can push back first three
        typename decltype(seeds)::object_type new_seed_object(
            static_cast<unsigned int>(particle.at(0)), // bottom_index
            static_cast<unsigned int>(particle.at(1)), // middle_index
            static_cast<unsigned int>(particle.at(2))  // top_index
        );
        seeds.push_back(new_seed_object);

        // save the indices of all spacepoints for later use
        seed_indices.push_back(particle);
    }

    // copy spacepoints, measurements, seeds to device
    traccc::edm::spacepoint_collection::buffer spacepoints(
        static_cast<unsigned int>(spacepoints_per_event.size()), *m_cached_device_mr);
    m_copy(vecmem::get_data(spacepoints_per_event), spacepoints)->wait();

    traccc::measurement_collection_types::buffer measurements(
        static_cast<unsigned int>(measurements_per_event.size()), *m_cached_device_mr);
    m_copy(vecmem::get_data(measurements_per_event), measurements)->wait();

    traccc::edm::seed_collection::buffer seeds_device(
        static_cast<unsigned int>(seeds.size()), *m_cached_device_mr);
    m_copy(vecmem::get_data(seeds), seeds_device)->wait();

    // run parameter estimation (on device)
    const traccc::cuda::track_params_estimation::output_type track_params =
        m_track_parameter_estimation(measurements, spacepoints,
            seeds_device, m_field_vec);

    m_stream.synchronize();
    
    // print number of track params
    size_t nTrackParams = track_params.size();
    std::cout << "Number of track parameters: " << nTrackParams << std::endl;

    //! copy track params to host
    traccc::host::track_params_estimation::output_type track_params_host(
        static_cast<unsigned int>(nTrackParams), m_host_mr);
    m_copy(vecmem::get_data(track_params), vecmem::get_data(track_params_host))->wait();

    // Create and populate track_candidates container directly
    traccc::track_candidate_container_types::host track_candidates_host(m_host_mr);

    // track candidates are composed of container_types<finding_result, track_candidate>
    // where finding_results is a struct with seed_params and trk_quality
    // and track_candidate is just the measurements corresponding to the measurements
    for (size_t i = 0; i < nTrackParams; ++i) {
        // Create finding_result
        traccc::finding_result finding_result;
        finding_result.seed_params = track_params_host.at(i);
        finding_result.trk_quality = traccc::track_quality{}; // Default initialize

        // Create measurements vector for this track
        traccc::track_candidate_collection_types::host measurements_for_track(m_host_mr);

        // push back all measurements for the selected track
        for (const auto& sp_idx : seed_indices[i]) {

            const auto& spacepoint = spacepoints_per_event.at(sp_idx);
            
            // Add the first measurement
            measurements_for_track.push_back(measurements_per_event.at(spacepoint.measurement_index_1()));
            // Add the second measurement if it exists
            if (spacepoint.measurement_index_2() != traccc::edm::spacepoint_collection::host::INVALID_MEASUREMENT_INDEX) {
                measurements_for_track.push_back(measurements_per_event.at(spacepoint.measurement_index_2()));
            }
        }
        
        track_candidates_host.push_back(finding_result, measurements_for_track);
    }
    
    //! Copy to device
    traccc::track_candidate_container_types::buffer track_candidates = m_copy_track_candidates(
        traccc::get_data(track_candidates_host));

    // Run the track fitting
    const fitting_algorithm::output_type track_states = 
        m_fitting(m_device_detector_view, m_field, track_candidates);

    // copy track states to host
    traccc::track_state_container_types::host track_states_host = m_copy_track_states(track_states);
    std::cout << "Number of fitted tracks: " << track_states_host.size() << std::endl;

    return track_states_host;
}


int main(int argc, char *argv[])
{
    int deviceId = 0;
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::device_memory_resource device_mr(deviceId);

    // Optional: Parse command line argument for geometry directory
    std::string geoDir = "/global/cfs/projectdirs/m3443/data/GNN4ITK-traccc/ITk_data/ATLAS-P2-RUN4-03-00-00/";
    if (argc > 1) {
        geoDir = std::string(argv[1]);
        if (geoDir.back() != '/') {
            geoDir += "/";
        }
    }
    
    TracccGpuStandalone traccc_gpu(&host_mr, &device_mr, deviceId, geoDir);
   
    traccc::edm::spacepoint_collection::host spacepoints(host_mr);
    traccc::measurement_collection_types::host measurements(&host_mr);

    InputData input_data{};
    std::string athenaTransformsPath = geoDir + "athena_transforms.csv"; 
    inputDataToTracccMeasurements(
        input_data, spacepoints, measurements,
        traccc_gpu.getActsToBarcodeMap(), traccc_gpu.getAthenaToActsMap());

    std::vector<int> dummyGNNOutput(input_data.sp_x.size());
    auto main_particle = input_data.cl_particle_id.at(input_data.sp_cl1_index.at(0));
    for(auto i = 0ul; i< input_data.sp_x.size(); ++i) 
    {
        if( input_data.cl_particle_id.at(input_data.sp_cl1_index.at(i)) == main_particle) {
            int cl2_idx_val = input_data.sp_cl2_index.at(i);
            if (cl2_idx_val == -1) {
                dummyGNNOutput.at(i) = 0; // main particle
            } else if (input_data.cl_particle_id.at(cl2_idx_val) == main_particle) {
                dummyGNNOutput.at(i) = 0; // main particle
            } else {
                dummyGNNOutput.at(i) = 1; // secondary particle
            }
        } else {
            dummyGNNOutput.at(i) = 1; // secondary particle
        }
    }

    // print out dummy GNN output
    std::cout << "Dummy GNN Output: ";
    for (const auto &label : dummyGNNOutput) {
        std::cout << label << " ";
    }
    std::cout << std::endl;

    std::cout << "Length of dummy GNN output: " << dummyGNNOutput.size() << std::endl;

    auto tracks = traccc_gpu.fitFromGnnOutput(
      spacepoints, measurements, dummyGNNOutput);
    
    return 0;
}
