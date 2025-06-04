#pragma once 

#include "TestEvent.hpp"

#include "traccc/edm/spacepoint_collection.hpp"
#include "traccc/edm/measurement.hpp"

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
    }
};


inline void inputDataToTracccMeasurements(
    InputData gnnTracks,
    traccc::edm::spacepoint_collection::host& spacepoints,
    traccc::measurement_collection_types::host& measurements) 
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
            0.f,                                    // Variance in z
            0.f                                     // Variance in radius
        });
    }

    // print number of spacepoints / measurements
    std::cout << "Number of measurements created: " << measurements.size() << std::endl;
    std::cout << "Number of spacepoints created: " << spacepoints.size() << std::endl;
}

