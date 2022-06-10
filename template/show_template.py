import os
import sys
import yaml
import argparse

import math
import torch
import numpy as np
import torchvision
from torch.nn import functional as F

def show_template(template_tensor, render_config):
    batch_size = 4
    print(template_tensor.shape)
    # continue
    all_templates = template_tensor.unsqueeze(0).float().expand((batch_size, -1, -1, -1, -1, -1)).unbind(dim=1)
    # rotate template
    part_templates, encoded_angles = rotate_volume_tensor(
        all_templates[1:], arti_config=render_config['ARTI_CONFIG'],
        #encoded_artipose=torch.tensor([[0.0], [0.3], [0.7], [1.0]])
    )
    # render_config['SAMPLE_CENTER'] = [1.4, 0]
    # render_config['SAMPLE_RANGE'] = [0, 0]
    spatial_template = all_templates[0] + part_templates.sum(0) if part_templates is not None else all_templates[0]
    # sample camera
    camera_origins, pitch, yaw = sample_camera(
        batch_size, 
        sample_center=render_config['SAMPLE_CENTER'], sample_range=render_config['SAMPLE_RANGE'],
        mode=render_config['SAMPLE_METHOD'], radius=render_config['RADIUS']
    )
    ray_origins, ray_points, ray_points_z, ray_directions = sample_rays(
        camera_origins, resolution=(64, 64),
        points_per_ray=render_config['SAMPLE_PER_RAY'], fov=render_config['FOV'], depth_range=render_config['RAY_RANGE'],
        radius=render_config['RADIUS']
    )
    ray_points = ray_points.reshape(batch_size, 64 * 64 * render_config['SAMPLE_PER_RAY'], 3)
    print(ray_points[0].mean(dim=0))
    print(ray_points[0].min(dim=0))
    print(ray_points[0].max(dim=0))
    # assert 0 == 1
    ray_directions = torch.unsqueeze(ray_directions, -2).expand(-1, -1, render_config['SAMPLE_PER_RAY'], -1)
    ray_directions = ray_directions.reshape(batch_size, 64 * 64 * render_config['SAMPLE_PER_RAY'], 3)
    # show
    spatial_template = F.grid_sample(
        spatial_template, ray_points[:, None, None], padding_mode='zeros', align_corners=True
    ).squeeze(dim=3).squeeze(dim=2).permute(0, 2, 1)
    spatial_template = spatial_template.mean(dim=-1).reshape(batch_size, 64, 64, render_config['SAMPLE_PER_RAY']).mean(dim=-1)
    spatial_template = spatial_template - spatial_template.min()
    spatial_template = spatial_template / spatial_template.max()
    show_images = torchvision.transforms.functional.resize(spatial_template.float(), [256, 256], antialias=True)
    show_images = torchvision.utils.make_grid(
        show_images[:, None], nrow=show_images.shape[0],
        pad_value=1, padding=5
    ).cpu()
    torchvision.utils.save_image(show_images, 'template.jpg')
    print('DONE!')

def sample_camera(sample_num, sample_center, sample_range, radius=1, mode='uniform'):
    """
    Samples n random locations along a sphere of radius r. Uses the specified distribution.
    Theta is yaw in radians (-pi, pi)
    Phi is pitch in radians (0, pi)
    """
    eps = 1e-5
    if mode == 'uniform':
        theta = (torch.rand((sample_num, 1)) - 0.5) * 2 * sample_range[0] + sample_center[0]
        phi = (torch.rand((sample_num, 1)) - 0.5) * 2 * sample_range[1] + sample_center[1]
    elif mode == 'gaussian':
        theta = torch.randn((sample_num, 1)) * sample_range[0] + sample_center[0]
        phi = torch.randn((sample_num, 1)) * sample_range[1] + sample_center[1]
    else: # Just use the mean.
        raise NotImplementedError('Distribution mode {} not implemented!'.format(mode))
    phi = torch.clamp(phi, eps, math.pi - eps)
    output_points = torch.zeros((sample_num, 3))
    output_points[:, 0:1] = radius * torch.sin(phi) * torch.cos(theta)
    output_points[:, 2:3] = radius * torch.sin(phi) * torch.sin(theta)
    output_points[:, 1:2] = radius * torch.cos(phi)
    return output_points, phi, theta


def sample_rays(origins_cam, resolution, points_per_ray, fov, depth_range, radius=1):
    """Returns sample points, z_vals, and ray directions in camera space."""
    W, H = resolution
    ray_number = W * H
    camera_number = origins_cam.shape[0]
    device = origins_cam.device
    # Sample camera positions.
    x, y = torch.meshgrid(torch.linspace(-1, 1, W, device=device), torch.linspace(1, -1, H, device=device), indexing='xy')
    x = x.flatten()
    y = y.flatten()
    z = -torch.ones_like(x) / math.tan((fov * math.pi / 180)/2)
    rays_d_cam = torch.stack([x, y, z], dim=-1)
    rays_d_cam = F.normalize(rays_d_cam, dim=-1) # constrain to sphere
    z_vals = torch.linspace(depth_range[0] * radius, depth_range[1] * radius, points_per_ray, device=device).reshape(1, points_per_ray, 1).repeat(ray_number, 1, 1)
    points = rays_d_cam.unsqueeze(1).repeat(1, points_per_ray, 1) * z_vals
    points_cam = torch.stack(camera_number * [points])
    z_vals_cam = torch.stack(camera_number * [z_vals])
    rays_d_cam = torch.stack(camera_number * [rays_d_cam])
    points_cam, z_vals_cam = perturb_points(points_cam, z_vals_cam, rays_d_cam)
    # Map points in camera space to world space
    camera_forward_vector = F.normalize(-origins_cam, dim=-1) # constrain to sphere
    camera2world_matrix = create_cam2world_matrix(camera_forward_vector, origins_cam)
    points_cam_homogeneous = torch.ones((camera_number, ray_number, points_per_ray, 4), device=device)
    points_cam_homogeneous[:, :, :, :3] = points_cam
    origins_cam_homogeneous = torch.zeros((camera_number, 4, ray_number), device=device)
    origins_cam_homogeneous[:, 3, :] = 1
    transformed_points = torch.bmm(
            camera2world_matrix,
            points_cam_homogeneous.reshape(camera_number, -1, 4).permute(0,2,1)
        ).permute(0, 2, 1).reshape(camera_number, ray_number, points_per_ray, 4)
    transformed_ray_directions = torch.bmm(
            camera2world_matrix[..., :3, :3],
            rays_d_cam.reshape(camera_number, -1, 3).permute(0,2,1)
        ).permute(0, 2, 1).reshape(camera_number, ray_number, 3)
    transformed_ray_origins = torch.bmm(
            camera2world_matrix, 
            origins_cam_homogeneous
        ).permute(0, 2, 1).reshape(camera_number, ray_number, 4)[..., :3]
    transformed_points = transformed_points[..., :3]
    return transformed_ray_origins, transformed_points, z_vals_cam, transformed_ray_directions

def perturb_points(points, z_vals, ray_directions):
    distance_between_points = z_vals[:,:,1:2,:] - z_vals[:,:,0:1,:]
    offset = (torch.rand(z_vals.shape, device=z_vals.device) - 0.5) * distance_between_points
    z_vals = z_vals + offset
    points = points + offset * ray_directions.unsqueeze(2)
    return points, z_vals


def create_cam2world_matrix(forward_vector, origin):
    """Takes in the direction the camera is pointing and the camera origin and returns a cam2world matrix."""
    # build camera coordinate system
    up_vector = torch.tensor([0, 1, 0], dtype=torch.float, device=forward_vector.device).expand_as(forward_vector)
    left_vector = F.normalize(torch.cross(up_vector, forward_vector, dim=-1), dim=-1)
    up_vector = F.normalize(torch.cross(forward_vector, left_vector, dim=-1), dim=-1)
    # rot
    rotation_matrix = torch.eye(4, device=forward_vector.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    rotation_matrix[:, :3, :3] = torch.stack((-left_vector, up_vector, -forward_vector), axis=-1)
    # trans
    translation_matrix = torch.eye(4, device=forward_vector.device).unsqueeze(0).repeat(forward_vector.shape[0], 1, 1)
    translation_matrix[:, :3, 3] = origin
    cam2world = translation_matrix @ rotation_matrix
    return cam2world


def rotate_volume_tensor(volume_tensor_list, arti_config, encoded_artipose=None):
    if not len(volume_tensor_list):
        return None, None
    else:
        all_volume_tensors, all_encoded_angles = [], []
        for idx in range(len(volume_tensor_list)):
            volume_tensor = volume_tensor_list[idx]
            theta_range = arti_config['THETA_RANGE'][idx]
            if encoded_artipose is None:
                encoded_angles = torch.rand(volume_tensor.shape[0], device=volume_tensor.device)
            else:
                encoded_angles = encoded_artipose[:, idx]
            angles = encoded_angles * abs(theta_range[0]-theta_range[1]) + theta_range[0]
            angles_tensor = volume_tensor.new_zeros(volume_tensor.shape[0], 3)
            angles_tensor[:, arti_config['THETA_AXIS'][idx]] = angles
            rmatrix = angle_to_rmatrix(angles_tensor, center=arti_config['THETA_CENTER'][idx], device=volume_tensor.device)
            grid = F.affine_grid(rmatrix, volume_tensor.size(), align_corners=False)
            volume_tensor = F.grid_sample(volume_tensor, grid.detach(), padding_mode='zeros', align_corners=False)
            all_volume_tensors.append(volume_tensor)
            all_encoded_angles.append(encoded_angles.reshape(-1, 1))
        all_volume_tensors = torch.stack(all_volume_tensors)
        all_encoded_angles = torch.cat(all_encoded_angles, dim=1)
        return all_volume_tensors, all_encoded_angles


def angle_to_rmatrix(angles, center, device):
    # all in radians
    def rot_matrix_z(theta):
        mat = torch.eye(4, device=theta.device)
        mat[0, 0] = torch.cos(theta)
        mat[0, 1] = -torch.sin(theta)
        mat[1, 0] = torch.sin(theta)
        mat[1, 1] = torch.cos(theta)
        return mat

    def rot_matrix_x(theta):
        mat = torch.eye(4, device=theta.device)
        mat[1, 1] = torch.cos(theta)
        mat[1, 2] = -torch.sin(theta)
        mat[2, 1] = torch.sin(theta)
        mat[2, 2] = torch.cos(theta)
        return mat

    def rot_matrix_y(theta):
        mat = torch.eye(4, device=theta.device)
        mat[0, 0] = torch.cos(theta)
        mat[0, 2] = torch.sin(theta)
        mat[2, 0] = -torch.sin(theta)
        mat[2, 2] = torch.cos(theta)
        return mat
    assert angles.dim() == 2 and angles.shape[-1] == 3
    angles_x, angles_y, angles_z = angles.unbind(dim=-1)
    batch_size = len(angles)
    trans_before = torch.eye(4, 4, device=device)
    trans_before[:3, -1] = torch.tensor(center).type_as(trans_before)
    trans_after = torch.inverse(trans_before)
    theta = torch.eye(4, 4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    trans_after = torch.eye(4, 4, device=device)
    trans_after[:3, -1] = -torch.tensor(center).type_as(trans_before)
    for i in range(batch_size):
        theta[i] = torch.linalg.multi_dot(
            [trans_before, rot_matrix_z(angles_z[i]), rot_matrix_y(angles_y[i]), rot_matrix_x(angles_x[i]), trans_after]
        )
    theta = theta[:, :3]
    return theta
