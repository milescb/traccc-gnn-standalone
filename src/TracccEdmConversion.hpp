#pragma once 

#include "TestEvent.hpp"

#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/edm/measurement.hpp"
#include "traccc/definitions/common.hpp"

#include <dfe/dfe_io_dsv.hpp>
#include <dfe/dfe_namedtuple.hpp>

#include <iostream>

struct InputData {
    // Spacepoint data
    std::vector<float> sp_x;
    std::vector<float> sp_y;
    std::vector<float> sp_z;
    std::vector<int> sp_cl1_index;
    std::vector<int> sp_cl2_index;

    // Cluster data
    std::vector<float> cl_x;
    std::vector<float> cl_y;
    std::vector<float> cl_z;
    std::vector<float> cl_loc_eta;
    std::vector<float> cl_loc_phi;
    std::vector<float> cl_cov_00;
    std::vector<float> cl_cov_11;

    std::vector<std::uint64_t> cl_particle_id;
    std::vector<std::uint64_t> cl_module_id;

    // Constructor to initialize with TestEvent data
    InputData() {
        // Initialize spacepoint vectors
        sp_x = SPx;
        sp_y = SPy;
        sp_z = SPz;
        sp_cl1_index = SPCL1_index;
        sp_cl2_index = SPCL2_index;

        // Initialize cluster vectors
        cl_x = CLx;
        cl_y = CLy;
        cl_z = CLz;
        cl_loc_eta = CLloc_eta;
        cl_loc_phi = CLloc_phi;
        cl_cov_00 = cov_00;
        cl_cov_11 = cov_11;

        cl_particle_id = particle_id;
        cl_module_id = CLmoduleID;
    }
};

struct athena_acts_mapping {
    uint64_t acts_geoid; // long long int
    std::string ath_geoid;  // hex string

    DFE_NAMEDTUPLE(athena_acts_mapping, acts_geoid, ath_geoid);
};

/// Create csv reader
inline dfe::NamedTupleCsvReader<athena_acts_mapping> make_athena_acts_reader(
    std::string_view filename) {
    return {filename.data(), {"acts_geoid", "ath_geoid"}};
}

/// Read Athena-to-ACTS geometry ID mapping from CSV file
inline std::map<uint64_t, uint64_t> read_athena_to_acts_mapping(
    std::string_view filename) {

    auto reader = make_athena_acts_reader(filename);
    std::map<uint64_t, uint64_t> result;
    athena_acts_mapping mapping;
    
    while (reader.read(mapping)) {
        // Convert hex string to uint64 (remove "0x" prefix)
        std::string hex_str = mapping.ath_geoid;
        if (hex_str.substr(0, 2) == "0x") {
            hex_str = hex_str.substr(2);
        }
        uint64_t athena_id = std::stoull(hex_str, nullptr, 16);
        result.insert({athena_id, mapping.acts_geoid});
    }
    
    return result;
}

inline void inputDataToTracccMeasurements(
    InputData gnnTracks,
    traccc::edm::spacepoint_collection::host& spacepoints,
    traccc::measurement_collection_types::host& measurements,
    const std::map<std::uint64_t, detray::geometry::barcode>& acts_id_to_barcode_map,
    const std::map<uint64_t, uint64_t>& athena_to_acts_map) 
{
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
        meas.meas_dim = gnnTracks.sp_cl2_index[i] >= 0 ? 2 : 1;
        // Set geometry ID
        // Output from GNN is Athena ID, first need to convert to ACTS ID
        auto it_athena_to_acts = athena_to_acts_map.find(gnnTracks.cl_module_id[i]);
        uint64_t acts_id = it_athena_to_acts->second;
        // Traccc needs detray::geometry::barcode, so we convert with final map
        auto it_acts_to_barcode = acts_id_to_barcode_map.find(acts_id);
        meas.surface_link = it_acts_to_barcode->second;
        
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
            0.f,                                    // Variance in z
            0.f                                     // Variance in radius
        });
    }

    // print number of spacepoints / measurements
    std::cout << "Number of measurements created: " << measurements.size() << std::endl;
    std::cout << "Number of spacepoints created: " << spacepoints.size() << std::endl;
}

