// This file contains only Python bindings
#include "include/data_spec.hpp"

#include <torch/torch.h>


// scan
torch::Tensor inclusive_sum(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    bool normalize,
    bool backward);
torch::Tensor exclusive_sum(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    bool normalize,
    bool backward);
torch::Tensor inclusive_prod_forward(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs);
torch::Tensor inclusive_prod_backward(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    torch::Tensor outputs,
    torch::Tensor grad_outputs);
torch::Tensor exclusive_prod_forward(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs);
torch::Tensor exclusive_prod_backward(
    torch::Tensor chunk_starts,
    torch::Tensor chunk_cnts,
    torch::Tensor inputs,
    torch::Tensor outputs,
    torch::Tensor grad_outputs);

bool is_cub_available();
torch::Tensor inclusive_sum_cub(
    torch::Tensor ray_indices,
    torch::Tensor inputs,
    bool backward);
torch::Tensor exclusive_sum_cub(
    torch::Tensor indices,
    torch::Tensor inputs,
    bool backward);
torch::Tensor inclusive_prod_cub_forward(
    torch::Tensor indices,
    torch::Tensor inputs);
torch::Tensor inclusive_prod_cub_backward(
    torch::Tensor indices,
    torch::Tensor inputs,
    torch::Tensor outputs,
    torch::Tensor grad_outputs);
torch::Tensor exclusive_prod_cub_forward(
    torch::Tensor indices,
    torch::Tensor inputs);
torch::Tensor exclusive_prod_cub_backward(
    torch::Tensor indices,
    torch::Tensor inputs,
    torch::Tensor outputs,
    torch::Tensor grad_outputs);

// grid
std::vector<torch::Tensor> ray_aabb_intersect(
    const torch::Tensor rays_o, // [n_rays, 3]
    const torch::Tensor rays_d, // [n_rays, 3]
    const torch::Tensor aabbs,  // [n_aabbs, 6]
    const float near_plane,
    const float far_plane, 
    const float miss_value);
std::tuple<RaySegmentsSpec, RaySegmentsSpec, torch::Tensor> traverse_grids(
    // rays
    const torch::Tensor rays_o, // [n_rays, 3]
    const torch::Tensor rays_d, // [n_rays, 3]
    const torch::Tensor rays_mask,   // [n_rays]
    // grids
    const torch::Tensor binaries,  // [n_grids, resx, resy, resz]
    const torch::Tensor aabbs,     // [n_grids, 6]
    // intersections
    const torch::Tensor t_sorted,  // [n_rays, n_grids * 2]
    const torch::Tensor t_indices,  // [n_rays, n_grids * 2]
    const torch::Tensor hits,    // [n_rays, n_grids]
    // options
    const torch::Tensor near_planes,
    const torch::Tensor far_planes,
    const float step_size,
    const float cone_angle,
    const bool compute_intervals,
    const bool compute_samples,
    const bool compute_terminate_planes,
    const int32_t traverse_steps_limit, // <= 0 means no limit
    const bool over_allocate); // over allocate the memory for intervals and samples

// pdf
std::vector<RaySegmentsSpec> importance_sampling(
    RaySegmentsSpec ray_segments,
    torch::Tensor cdfs,                 
    torch::Tensor n_intervels_per_ray,  
    bool stratified);
std::vector<RaySegmentsSpec> importance_sampling(
    RaySegmentsSpec ray_segments,
    torch::Tensor cdfs,                  
    int64_t n_intervels_per_ray,
    bool stratified);
std::vector<torch::Tensor> searchsorted(
    RaySegmentsSpec query,
    RaySegmentsSpec key);

// cameras
torch::Tensor opencv_lens_undistortion(
    const torch::Tensor& uv,      // [..., 2]
    const torch::Tensor& params,  // [..., 6]
    const float eps,
    const int max_iterations);
torch::Tensor opencv_lens_undistortion_fisheye(
    const torch::Tensor& uv,      // [..., 2]
    const torch::Tensor& params,  // [..., 4]
    const float criteria_eps,
    const int criteria_iters);
