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

    // copy spacepoints and measurements to device
    traccc::edm::spacepoint_collection::buffer spacepoints(
        static_cast<unsigned int>(spacepoints_per_event.size()), *m_cached_device_mr);
    m_copy(vecmem::get_data(spacepoints_per_event), spacepoints)->wait();

    traccc::measurement_collection_types::buffer measurements(
        static_cast<unsigned int>(measurements_per_event.size()), *m_cached_device_mr);
    m_copy(vecmem::get_data(measurements_per_event), measurements)->wait();

    // Seeding and parameter estimation
    // TODO: instead use first three measurments and create seeds from this
    // TODO But first: sort by radius to then get the first three!
    // NOTE: this needs to be consistent between measurements and spacepoints
    // Then, pass to the parameter estimation algo
    traccc::edm::seed_collection::buffer dummySeeds{};
    m_stream.synchronize();

    const traccc::cuda::track_params_estimation::output_type track_params =
        m_track_parameter_estimation(measurements, spacepoints,
            dummySeeds, m_field_vec);
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

    std::vector<int> dummyGNNOutput(input_data.sp_x.size());
    auto main_particle = input_data.cl_particle_id.at(input_data.sp_cl1_index.at(0));
    for(auto i = 0ul; i< input_data.sp_x.size(); ++i) {
      if( input_data.cl_particle_id.at(input_data.sp_cl1_index.at(i)) == main_particle ) {
        dummyGNNOutput.push_back(0);
      } else {
        dummyGNNOutput.push_back(1);
      }
    }

    auto tracks = traccc_gpu.fitFromGnnOutput(
      spacepoints, measurements, dummyGNNOutput
    );
    return 0;
}
