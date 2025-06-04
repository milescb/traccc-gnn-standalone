#include "TracccGpuStandalone.hpp"
#include "TracccEdmConversion.hpp"

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
    // MARK: sort by radius

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
        size_t old_idx = original_indices[new_idx];
        sorted_spacepoints.push_back(spacepoints_per_event[old_idx]);
        sorted_gnnOutputLabels.push_back(gnnOutputLabels[old_idx]);
    }

    // TODO: later, may need to deal with measurement ordering

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
        std::cout << "Particle " << i << ": " << spacepoints_per_particle.at(i).size() << " spacepoints" << std::endl;
    }

    // Seeds are the first three spacepoints
    traccc::edm::seed_collection::host seeds(*m_host_mr);
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

    // run parameter estimation
    const traccc::cuda::track_params_estimation::output_type track_params =
        m_track_parameter_estimation(measurements, spacepoints,
            seeds_device, m_field_vec);
    
    // print number of track params
    std::cout << "Number of track parameters: " << track_params.size() << std::endl;
    
    m_stream.synchronize();

    // Run the track fitting
    traccc::track_candidate_container_types::const_view *dummyCandidates{};
    const fitting_algorithm::output_type track_states = 
        m_fitting(m_device_detector_view, m_field, *dummyCandidates);


    // copy track states to host
    traccc::track_state_container_types::host track_states_host = m_copy_track_states(track_states);
    return track_states_host;
}


int main(int argc, char *argv[])
{
    int deviceId = 0;
    vecmem::host_memory_resource host_mr;
    vecmem::cuda::device_memory_resource device_mr(deviceId);
    
    TracccGpuStandalone traccc_gpu(&host_mr, &device_mr, deviceId);
   
    traccc::edm::spacepoint_collection::host spacepoints(host_mr);
    traccc::measurement_collection_types::host measurements(&host_mr);

    InputData input_data{};
    inputDataToTracccMeasurements(input_data, spacepoints, measurements);

    std::cout << input_data.sp_x.size() << " spacepoints " << std::endl;

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
      spacepoints, measurements, dummyGNNOutput
    );
    return 0;
}
