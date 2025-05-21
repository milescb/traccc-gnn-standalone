#ifndef TRACCC_GPU_STANDALONE_HPP
#define TRACCC_GPU_STANDALONE_HPP

#include <iostream>
#include <memory>

// CUDA include(s).
#include <cuda_runtime.h>

// #include "DataHandler.hpp"

// Project include(s).
#include "traccc/clusterization/clustering_config.hpp"
#include "traccc/cuda/clusterization/clusterization_algorithm.hpp"
#include "traccc/cuda/clusterization/measurement_sorting_algorithm.hpp"
#include "traccc/cuda/finding/finding_algorithm.hpp"
#include "traccc/cuda/fitting/fitting_algorithm.hpp"
#include "traccc/cuda/seeding/seeding_algorithm.hpp"
#include "traccc/cuda/seeding/spacepoint_formation_algorithm.hpp"
#include "traccc/cuda/seeding/track_params_estimation.hpp"
#include "traccc/cuda/utils/stream.hpp"
#include "traccc/edm/silicon_cell_collection.hpp"
#include "traccc/edm/track_state.hpp"
#include "traccc/fitting/kalman_filter/kalman_fitter.hpp"
#include "traccc/utils/algorithm.hpp"
#include "traccc/device/container_d2h_copy_alg.hpp"
#include "traccc/ambiguity_resolution/greedy_ambiguity_resolution_algorithm.hpp"

#include "traccc/io/read_measurements.hpp"
#include "traccc/seeding/silicon_pixel_spacepoint_formation_algorithm.hpp"

// Detray include(s).
#include "detray/core/detector.hpp"
#include "detray/detectors/bfield.hpp"
#include "detray/io/frontend/detector_reader.hpp"
#include "detray/navigation/navigator.hpp"
#include "detray/propagator/propagator.hpp"
#include "detray/propagator/rk_stepper.hpp"

// VecMem include(s).
#include <vecmem/containers/vector.hpp>
#include <vecmem/containers/jagged_vector.hpp>
#include <vecmem/containers/jagged_device_vector.hpp>
#include <vecmem/memory/binary_page_memory_resource.hpp>
#include <vecmem/memory/cuda/device_memory_resource.hpp>
#include <vecmem/memory/memory_resource.hpp>
#include <vecmem/utils/cuda/async_copy.hpp>

// io
#include "traccc/io/read_cells.hpp"
#include "traccc/io/read_digitization_config.hpp"
#include "traccc/io/read_geometry.hpp"
#include "traccc/io/utils.hpp"
#include "traccc/io/csv/make_cell_reader.hpp"
#include "traccc/io/read_detector.hpp"
#include "traccc/io/read_detector_description.hpp"

// algorithm options
#include "traccc/options/accelerator.hpp"
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/performance.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_resolution.hpp"
#include "traccc/options/track_seeding.hpp"

// Command line option include(s).
#include "traccc/options/clusterization.hpp"
#include "traccc/options/detector.hpp"
#include "traccc/options/input_data.hpp"
#include "traccc/options/program_options.hpp"
#include "traccc/options/threading.hpp"
#include "traccc/options/throughput.hpp"
#include "traccc/options/track_finding.hpp"
#include "traccc/options/track_propagation.hpp"
#include "traccc/options/track_seeding.hpp"

struct clusterInfo {
    std::uint64_t detray_id; 
    unsigned int local_key;
    Eigen::Vector3d globalPosition;
    Eigen::Vector2d localPosition;
    bool pixel;
};

// function to set the CUDA device and get the stream
static traccc::cuda::stream setCudaDeviceAndGetStream(int deviceID)
{
    cudaError_t err = cudaSetDevice(deviceID);
    if (err != cudaSuccess)
    {
        throw std::runtime_error("Failed to set CUDA device: \
                " + std::string(cudaGetErrorString(err)));
    }
    return traccc::cuda::stream(deviceID);
}

/// Helper macro for checking the return value of CUDA function calls
#define CUDA_ERROR_CHECK(EXP)                                                  \
    do {                                                                       \
        const cudaError_t errorCode = EXP;                                     \
        if (errorCode != cudaSuccess) {                                        \
            throw std::runtime_error(std::string("Failed to run " #EXP " (") + \
                                     cudaGetErrorString(errorCode) + ")");     \
        }                                                                      \
    } while (false)

// Type definitions
using host_detector_type = traccc::default_detector::host;
using device_detector_type = traccc::default_detector::device;
using scalar_type = device_detector_type::scalar_type;

using stepper_type =
    detray::rk_stepper<detray::bfield::const_field_t<scalar_type>::view_t,
                   device_detector_type::algebra_type,
                   detray::constrained_step<scalar_type>>;
using navigator_type = detray::navigator<const device_detector_type>;
using device_navigator_type = detray::navigator<const device_detector_type>;

using spacepoint_formation_algorithm =
    traccc::cuda::spacepoint_formation_algorithm<
        traccc::default_detector::device>;
using clustering_algorithm = traccc::cuda::clusterization_algorithm;
using finding_algorithm =
    traccc::cuda::finding_algorithm<stepper_type, navigator_type>;
using fitting_algorithm = traccc::cuda::fitting_algorithm<
    traccc::kalman_fitter<stepper_type, navigator_type>>;

class TracccGpuStandalone
{
private:
    /// Device ID to use
    int m_device_id;

    /// Logger 
    std::unique_ptr<const traccc::Logger> logger;
    /// Host memory resource
    vecmem::host_memory_resource m_host_mr;
    /// CUDA stream to use
    traccc::cuda::stream m_stream;
    /// Device memory resource
    vecmem::cuda::device_memory_resource m_device_mr;
    /// Device caching memory resource
    std::unique_ptr<vecmem::binary_page_memory_resource> m_cached_device_mr;
    /// (Asynchronous) memory copy object
    mutable vecmem::cuda::async_copy m_copy;
    /// Memory resource for the host memory
    traccc::memory_resource m_mr;

    /// data configuration
    traccc::geometry m_surface_transforms;
    /// digitization configuration
    std::unique_ptr<traccc::digitization_config> m_digi_cfg;
    /// barcode map
    std::unique_ptr<std::map<std::uint64_t, detray::geometry::barcode>> m_barcode_map;

    // program configuration 
    /// detector options
    traccc::opts::detector m_detector_opts;
    /// propagation options
    traccc::opts::track_propagation m_propagation_opts;
    /// clusterization options
    detray::propagation::config m_propagation_config;
    /// Configuration for clustering
    traccc::clustering_config m_clustering_config;
    /// Configuration for the seed finding
    traccc::seedfinder_config m_finder_config;
    /// Configuration for the spacepoint grid formation
    traccc::spacepoint_grid_config m_grid_config;
    /// Configuration for the seed filtering
    traccc::seedfilter_config m_filter_config;

    /// further configuration
    /// Configuration for the track finding
    finding_algorithm::config_type m_finding_config;
    /// Configuration for the track fitting
    fitting_algorithm::config_type m_fitting_config;

    /// Constant B field for the (seed) track parameter estimation
    traccc::vector3 m_field_vec;
    /// Constant B field for the track finding and fitting
    detray::bfield::const_field_t<traccc::scalar> m_field;

    /// Detector description
    traccc::silicon_detector_description::host m_det_descr;
    /// Detector description buffer
    traccc::silicon_detector_description::buffer m_device_det_descr;
    /// Host detector
    std::unique_ptr<host_detector_type> m_detector;
    /// Buffer holding the detector's payload on the device
    host_detector_type::buffer_type m_device_detector;
    /// View of the detector's payload on the device
    host_detector_type::view_type m_device_detector_view;

    /// Sub-algorithms used by this full-chain algorithm
    /// Clusterization algorithm
    clustering_algorithm m_clusterization;
    /// Measurement sorting algorithm
    traccc::cuda::measurement_sorting_algorithm m_measurement_sorting;
    /// Spacepoint formation algorithm
    spacepoint_formation_algorithm m_spacepoint_formation;
    /// Seeding algorithm
    traccc::cuda::seeding_algorithm m_seeding;
    /// Track parameter estimation algorithm
    traccc::cuda::track_params_estimation m_track_parameter_estimation;

    /// Track finding algorithm
    finding_algorithm m_finding;
    /// Track fitting algorithm
    fitting_algorithm m_fitting;
    
    // copy back!
    traccc::device::container_d2h_copy_alg<traccc::track_state_container_types>
        m_copy_track_states;

    // ambiguity resolution
    traccc::greedy_ambiguity_resolution_algorithm::config_t m_resolution_config;
    traccc::greedy_ambiguity_resolution_algorithm m_resolution_alg;

    // Helper function to create and setup seedfinder_config
    static traccc::seedfinder_config create_and_setup_finder_config() {
        traccc::seedfinder_config cfg;
        // Set desired values
        cfg.zMin = -3000.f * traccc::unit<float>::mm;
        cfg.zMax = 3000.f * traccc::unit<float>::mm;
        cfg.rMax = 320.f * traccc::unit<float>::mm;
        cfg.rMin = 33.f * traccc::unit<float>::mm;
        cfg.collisionRegionMin = -200.f * traccc::unit<float>::mm;
        cfg.collisionRegionMax = 200.f * traccc::unit<float>::mm;
        cfg.minPt = 500.f * traccc::unit<float>::MeV; // Used by setup()
        cfg.cotThetaMax = 27.2899f;
        cfg.deltaRMin = 20.f * traccc::unit<float>::mm;
        cfg.deltaRMax = 280.f * traccc::unit<float>::mm;
        cfg.impactMax = 2.f * traccc::unit<float>::mm;
        cfg.sigmaScattering = 2.0f;
        cfg.maxPtScattering = 10.f * traccc::unit<float>::GeV;
        cfg.maxSeedsPerSpM = 3;
        // cfg.bFieldInZ uses its default (1.99724f T) unless set here
        // cfg.radLengthPerSeed uses its default (0.05f) unless set here

        cfg.setup(); // Call setup() again with the new values
        return cfg;
    }

public:
    TracccGpuStandalone(int deviceID = 0) :
        m_device_id(deviceID), 
        logger(traccc::getDefaultLogger("TracccGpuStandalone", traccc::Logging::Level::INFO)),
        m_host_mr(),
        m_stream(setCudaDeviceAndGetStream(deviceID)),
        m_device_mr(deviceID),
        m_cached_device_mr(
            std::make_unique<vecmem::binary_page_memory_resource>(m_device_mr)),
        m_copy(m_stream.cudaStream()),
        m_mr{*m_cached_device_mr, &m_host_mr},
        m_propagation_config(m_propagation_opts),
        m_clustering_config{256, 16, 8, 256},
        m_finder_config(create_and_setup_finder_config()), // Initialize m_finder_config using the helper
        m_grid_config(m_finder_config), 
        m_filter_config(), 
        // Initialize m_finding_config with members in declaration order
        m_finding_config{
            .max_num_branches_per_seed = 3,                
            .max_num_branches_per_surface = 5, 
            .min_track_candidates_per_track = 3,
            .max_track_candidates_per_track = 20,
            .max_num_skipping_per_cand = 3,
            .min_step_length_for_next_surface = 0.5f * detray::unit<float>::mm,
            .max_step_counts_for_next_surface = 100,
            .chi2_max = 10.f,
            .propagation = { 
                .navigation = { 
                    .overstep_tolerance = -300.f * traccc::unit<float>::um
                },
                .stepping = {
                    .min_stepsize = 1e-4f * traccc::unit<float>::mm,
                    .rk_error_tol = 1e-4f * traccc::unit<float>::mm,
                    .step_constraint = std::numeric_limits<float>::max(),
                    .path_limit = 5.f * traccc::unit<float>::m,
                    .max_rk_updates = 10000u,
                    .use_mean_loss = true,
                    .use_eloss_gradient = false,
                    .use_field_gradient = false,
                    .do_covariance_transport = true
                }
            }
            // .ptc_hypothesis and .initial_links_per_seed will use their defaults
        }, 
        m_fitting_config{
            .propagation = { 
                .navigation = {
                    .min_mask_tolerance = 1e-5f * traccc::unit<float>::mm,
                    .max_mask_tolerance = 3.f * traccc::unit<float>::mm,
                    .overstep_tolerance = -300.f * traccc::unit<float>::um,
                    .search_window = {0u, 0u}
                }
            }
        }, 
        m_field_vec{0.f, 0.f, m_finder_config.bFieldInZ},
        m_field(detray::bfield::create_const_field<host_detector_type::scalar_type>(m_field_vec)),
        m_det_descr{m_host_mr},
        m_clusterization(m_mr, m_copy, m_stream, m_clustering_config),
        m_measurement_sorting(m_mr, m_copy, m_stream, 
            logger->cloneWithSuffix("MeasSortingAlg")),
        m_spacepoint_formation(m_mr, m_copy, m_stream,
            logger->cloneWithSuffix("SpFormationAlg")),
        m_seeding(m_finder_config, m_grid_config, m_filter_config, 
                    m_mr, m_copy, m_stream,
                    logger->cloneWithSuffix("SeedingAlg")),
        m_track_parameter_estimation(m_mr, m_copy, m_stream,
            logger->cloneWithSuffix("TrackParEstAlg")),
        m_finding(m_finding_config, m_mr, m_copy, m_stream, 
            logger->cloneWithSuffix("TrackFindingAlg")),
        m_fitting(m_fitting_config, m_mr, m_copy, m_stream, 
            logger->cloneWithSuffix("TrackFittingAlg")),
        m_copy_track_states(m_mr, m_copy, logger->cloneWithSuffix("TrackStateD2HCopyAlg")),
        m_resolution_config(),
        m_resolution_alg(m_resolution_config)
    {
        // Tell the user what device is being used.
        int device = 0;
        CUDA_ERROR_CHECK(cudaGetDevice(&device));
        cudaDeviceProp props;
        CUDA_ERROR_CHECK(cudaGetDeviceProperties(&props, device));
        std::cout << "Using CUDA device: " << props.name << " [id: " << device
                << ", bus: " << props.pciBusID
                << ", device: " << props.pciDeviceID << "]" << std::endl;

        initialize();
    }

    // default destructor
    ~TracccGpuStandalone() = default;

    traccc::edm::spacepoint_collection::host read_spacepoints(
        std::vector<clusterInfo>& detray_clusters, bool do_strip);

    traccc::measurement_collection_types::host read_measurements(
        std::vector<clusterInfo>& detray_clusters, bool do_strip);

    void initialize();
    
    traccc::track_state_container_types::host run(
        traccc::edm::spacepoint_collection::host spacepoints_per_event,
        traccc::measurement_collection_types::host measurements_per_event);
};

#endif 