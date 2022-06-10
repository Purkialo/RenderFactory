import os
import json
import argparse

import torch
import mrcfile
import torchvision
import numpy as np
import open3d as o3d

from show_template import show_template

STAPLER_VOLUME_RENDERING = {
    'FOV': 30, 'RADIUS':3, 'SAMPLE_PER_RAY':36, 'RAY_RANGE': [0.8, 1.2],
    'SAMPLE_RANGE': [3.142, 0.524],
    'SAMPLE_CENTER': [1.571, 0.698],
    'SAMPLE_METHOD': 'uniform',
    'ARTI_CONFIG':{
        'THETA_AXIS': [2],
        'THETA_RANGE': [[-1.396, -0.174]],
        'THETA_CENTER': [[-0.3, 0.1, 0]]
    }
}
TRASHCAN_VOLUME_RENDERING = {
    'FOV': 30, 'RADIUS':3, 'SAMPLE_PER_RAY':36, 'RAY_RANGE': [0.8, 1.2],
    'SAMPLE_RANGE': [3.142, 0.524],
    'SAMPLE_CENTER': [1.571, 0.698],
    'SAMPLE_METHOD': 'uniform',
    'ARTI_CONFIG':{
        'THETA_AXIS': [2],
        'THETA_RANGE': [[-1.571, 0]],
        'THETA_CENTER': [[-0.15, 0.25, 0]]
    }
}

def build_template(model_dir_name, merge_list, rescale=1.0, cubic_size=3):
    parts_pcd = []
    all_union_pcd = o3d.geometry.PointCloud()
    for parts in model_dir_name:
        one_part_pcd = o3d.geometry.PointCloud()
        for obj_file in parts:
            mesh = o3d.io.read_triangle_mesh(obj_file)
            pcd = mesh.scale(rescale, center=[0, 0, 0]).sample_points_uniformly(5000)
            one_part_pcd += pcd
            all_union_pcd += pcd
        one_part_pcd = one_part_pcd.voxel_down_sample(0.01)
        parts_pcd.append(one_part_pcd)
        # o3d.visualization.draw_geometries([one_part_pcd])

    # merge parts
    merged_parts_pcd = []
    for merge_part in merge_list:
        one_part_pcd = o3d.geometry.PointCloud()
        for merge_part_id in merge_part:
            one_part_pcd += parts_pcd[merge_part_id]
        merged_parts_pcd.append(one_part_pcd)
    # for part_pcd in merged_parts_pcd:    
    #     o3d.visualization.draw_geometries([part_pcd])
    # o3d.visualization.draw_geometries([all_union_pcd])

    # build template
    voxel_array = torch.zeros((len(merged_parts_pcd), 32, 32, 32, 32))
    all_part_voxel_grid = []
    for idx, part_pcd in enumerate(merged_parts_pcd):
        part_voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
            part_pcd, voxel_size=cubic_size/32,
            min_bound=(-cubic_size/2, -cubic_size/2, -cubic_size/2),
            max_bound=(cubic_size/2, cubic_size/2, cubic_size/2)
        )
        indices = np.stack(list(vx.grid_index for vx in part_voxel_grid.get_voxels()))
        print((indices.min(axis=0)-16)/32)
        for indice in indices:
            voxel_array[idx, :, indice[0], indice[1], indice[2]] = torch.rand(32)
        all_part_voxel_grid.append(part_voxel_grid)
    # o3d.visualization.draw_geometries(all_part_voxel_grid)
    show_template(voxel_array, VOLUME_RENDERING)
    torch.save(voxel_array, 'template.pth')
    # output_array, _ = voxel_array.sum(dim=0).max(dim=0)
    # with mrcfile.new_mmap('outputs.mrc', overwrite=True, shape=output_array.shape, mrc_mode=2) as mrc:
    #     mrc.data[:] = output_array
    #     print('DONE')
    

def articulated_parser(model_path, mob_path, res_path):
    def find_all_children(origin_dict_list):
        parsed_list = []
        for origin_dict in origin_dict_list:
            if 'children' in origin_dict.keys():
                parsed_list += find_all_children(origin_dict['children'])
            else:
                parsed_list.append(origin_dict)
        return parsed_list
    with open(mob_path) as f:
        mobility_describe_list = json.load(f)
    with open(res_path) as f:
        mobility_parts_list = find_all_children(json.load(f))
    mobility_parts_mapping = {}
    for part in mobility_parts_list:
        mobility_parts_mapping[part['id']] = part
    results = []
    for mobility_parts in mobility_describe_list:
        parts_group = []
        for part in mobility_parts['parts']:
            parts_group += mobility_parts_mapping[part['id']]['objs']
        parts_path_group = []
        for part in parts_group:
            parts_path_group.append(os.path.join(model_path, 'textured_objs', part+'.obj'))
        results.append(parts_path_group)
    return results

if __name__ == '__main__':
    # TrashCan ../../SAPIEN_DATA/dataset/12483, [[1,2], [0]]
    # Stapler ../../SAPIEN_DATA/dataset/103111, [[2], [0, 1]], -s 5
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', '-m', required=True, type=str)
    parser.add_argument('--rescale', '-r', default=1.0, type=float)
    parser.add_argument('--cubic_size', '-s', default=3.0, type=float)
    args = parser.parse_args()
    config_dict = {
        'TrashCan': ['../../SAPIEN_DATA/dataset/12483', [[1,2], [0]]],
        'Stapler': ['../../SAPIEN_DATA/dataset/103111', [[2], [0, 1]]]
    }
    mobility_objs = articulated_parser(
        config_dict[args.model_name][0],
        os.path.join(config_dict[args.model_name][0], 'mobility_v2.json'),
        os.path.join(config_dict[args.model_name][0], 'result.json')
    )
    if args.model_name == 'Stapler':
        VOLUME_RENDERING = STAPLER_VOLUME_RENDERING
    if args.model_name == 'TrashCan':
        VOLUME_RENDERING = TRASHCAN_VOLUME_RENDERING
    build_template(
        mobility_objs, config_dict[args.model_name][1], rescale=args.rescale, cubic_size=args.cubic_size
    )
