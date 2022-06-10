#!/usr/bin/env python3 
# -*- coding:utf-8 -*- 
# Copyright (C) Xuangeng Chu <purkialo@gmail.com>
# If you have any questions about this repo, please contact @xuangeng_chu

import os
import math
import json
import random
import argparse

import bpy
import numpy as np

TARGET_CATEGORY = ['TrashCan']

class DataFactory:
    def __init__(self, arti_reverse):
        self.config = dict(
            light_setting={}, camera_setting={}, articulation_setting={}
        )
        # light
        self.config['light_setting']['number'] = 6
        self.config['light_setting']['azimuths'] = np.linspace(0, 360, self.config['light_setting']['number']+1)[:-1] + \
                                                   np.random.randint(0, 360//(self.config['light_setting']['number']*2))
        self.config['light_setting']['elevation'] = [30, 40]
        self.config['light_setting']['distance'] = [4.5, 6]
        self.config['light_setting']['energy'] = [800, 1000]
        # camera
        self.config['camera_setting']['distance'] = [5.5, 5.5]
        self.config['camera_setting']['elevation'] = [10, 70]
        self.config['camera_setting']['azimuths'] = [0, 360]
        # articulation
        self.config['articulation_setting']['overwrite'] = True
        l, h = -90, 0#-80, -10
        self.config['articulation_setting']['range'] = [l, h]
        if arti_reverse:
            self.config['articulation_setting']['range'] = [-h, -l]

    def set_model_path(self, model_path, mobility_describe_list, mobility_parts_mapping):
        self.model_path = model_path
        self.mobility_describe_list = mobility_describe_list
        self.mobility_parts_mapping = mobility_parts_mapping

    def run_generate(self, lineid):
        self._clear()
        bpy.context.scene.eevee.use_gtao = True
        bpy.context.scene.eevee.taa_samples = 8
        bpy.context.scene.eevee.taa_render_samples = 32
        self._import_all_obj()
        theta = self._articulation_deformation(self.mobility_describe_list, self.config['articulation_setting'])
        self._add_lights(self.config['light_setting'])
        cam_params = self._import_camera(self.config['camera_setting'])
        # self._add_world_voronoi_texture()
        result_path = './results/{}_{}.png'.format(lineid, cam_params+f'_{abs(theta)}')
        result_path = './results/{}/{}_{}.png'.format(lineid, lineid, cam_params+f'_{abs(theta)}')
        self._render_rgb_and_save(result_path)

    def _import_all_obj(self):
        bpy.ops.mesh.primitive_cube_add(size=0.05, location=(0, 0, 0))
        bpy.data.objects[0].name = 'Origin_Cube'
        bpy.data.objects[0].data.name = 'Origin_Cube'
        bpy.data.objects[0].hide_render = True
        random_color = [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]
        for mobility_parts in self.mobility_describe_list:
            parts_group = []
            for part in mobility_parts['parts']:
                parts_group += self.mobility_parts_mapping[part['id']]['objs']
            parts_path_group = []
            for part in parts_group:
                parts_path_group.append(os.path.join(data_path, 'textured_objs', part+'.obj'))
            self._import_objs(parts_path_group, '{}_{}'.format(mobility_parts['name'], mobility_parts['id']), random_color)
        # bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children['Collection']

    @staticmethod
    def _articulation_deformation(mobility_describe_list, articulation_setting):
        articulation_description = []
        for mobility_parts in mobility_describe_list:
            articulation_description.append({
                'name': '{}_{}'.format(mobility_parts['name'], mobility_parts['id']),
                'joint': mobility_parts['jointData']
            })
        theta = random.randint(
            articulation_setting['range'][0], articulation_setting['range'][1]
        )
        for part_description in articulation_description:
            part_name = part_description['name']
            part_joint = part_description['joint']
            if not len(part_joint.keys()):
                continue
            # norm theta
            axis_vector = part_joint['axis']['direction']
            assert axis_vector[1] * axis_vector[2] >= -1e-3
            theta_norm_x = math.atan(np.sqrt(axis_vector[0]**2+axis_vector[2]**2)/(np.abs(axis_vector[1]+1e-6)))
            theta_norm_y = math.atan(axis_vector[0]/(axis_vector[2]+1e-6))
            # set origin
            bpy.context.scene.cursor.location = [
                part_joint['axis']['origin'][0],
                -part_joint['axis']['origin'][2],
                part_joint['axis']['origin'][1]
            ]
            for obj in bpy.data.collections[part_name].objects:
                bpy.ops.object.select_all(action='DESELECT')
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
                bpy.context.scene.objects[obj.name].select_set(True)
                # bpy.ops.transform.rotate(value=1, orient_axis='X', orient_type='GLOBAL')
                # a = bpy.context.screen.areas
                ov=bpy.context.copy()
                ov['area']=[a for a in bpy.context.screen.areas if a.type=="VIEW_3D"][0]
                bpy.ops.transform.rotate(ov, value=theta_norm_x, orient_axis='X', orient_type='GLOBAL')
                bpy.ops.transform.rotate(ov, value=theta_norm_y, orient_axis='Y', orient_type='GLOBAL')
                bpy.ops.transform.rotate(ov, value=-np.radians(theta), orient_axis='Z', orient_type='GLOBAL')
                bpy.ops.transform.rotate(ov, value=-theta_norm_y, orient_axis='Y', orient_type='GLOBAL')
                bpy.ops.transform.rotate(ov, value=-theta_norm_x, orient_axis='X', orient_type='GLOBAL')
        bpy.ops.object.select_all(action='DESELECT')
        return theta

    @staticmethod
    def _add_lights(light_setting):
        # new lamps
        for idx in range(light_setting['number']):
            light_data = bpy.data.lights.new(name="light_{}".format(idx), type='POINT')
            light_data.energy = random.randint(light_setting['energy'][0], light_setting['energy'][1]) 
            light_object = bpy.data.objects.new(name="light_{}".format(idx), object_data=light_data)
            bpy.context.collection.objects.link(light_object)
            # bpy.context.view_layer.objects.active = light_object
            dist = random.uniform(light_setting['distance'][0], light_setting['distance'][1]) 
            elev = random.randint(
                light_setting['elevation'][0], light_setting['elevation'][1]
            ) / 180 * math.pi
            azim = light_setting['azimuths'][idx] / 180 * math.pi
            x = (dist * math.cos(azim) * math.cos(elev))
            y = (dist * math.sin(azim) * math.cos(elev))
            z = (dist * math.sin(elev))
            light_object.location = (x, y, z)

    @staticmethod
    def _import_camera(camera_config):
        # params
        dist = random.uniform(camera_config['distance'][0], camera_config['distance'][1])# * scale
        elev = random.randint(
            camera_config['elevation'][0], camera_config['elevation'][1]
        ) / 180 * math.pi
        azim = random.randint(
            camera_config['azimuths'][0], camera_config['azimuths'][1]
        ) / 180 * math.pi
        x = (dist * math.cos(azim) * math.cos(elev))
        y = (dist * math.sin(azim) * math.cos(elev))
        z = (dist * math.sin(elev))
        # new camera
        camera = bpy.data.cameras.new(name="DataFactoryCamera")
        cam_obj = bpy.data.objects.new("DataFactoryCamera", object_data=camera)
        cam_obj.location = (x, y, z)
        cam_obj.rotation_euler = (-math.cos(azim), math.cos(elev), math.sin(azim))
        # track to object
        # obj_names = bpy.data.objects.keys()
        # obj_names = [obj_name for obj_name in obj_names if obj_name not in ['DataFactoryCamera', 'light_0', 'light_1', 'light_2', 'light_3']]
        # assert len(obj_names) == 1, bpy.data.objects.keys()
        # obj_name = obj_names[0]
        cam_obj.constraints.new(type='TRACK_TO').target = bpy.data.objects['Origin_Cube']
        bpy.context.scene.camera = cam_obj
        bpy.context.collection.objects.link(cam_obj)
        # set resolution
        for scene in bpy.data.scenes:
            scene.render.resolution_x = 256
            scene.render.resolution_y = 256
        cam_params = 'EA_{}_{}'.format(int(elev/math.pi*180), int(azim/math.pi*180))
        return cam_params

    @staticmethod
    def _import_objs(model_path_list, collection_name, random_color):
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
        bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[collection_name]
        for model_path in model_path_list:
            exsist_material_names = bpy.data.materials.keys()
            exsist_material_names = [material_name for material_name in exsist_material_names if material_name not in ['Dots Stroke']]
            bpy.ops.import_scene.obj(filepath=model_path)
            obj_name = bpy.data.objects.keys()[0]
            bpy.ops.object.select_all(action = 'DESELECT')
            # change material
            material_names = bpy.data.materials.keys()
            material_names = [material_name for material_name in material_names if material_name not in ['Dots Stroke']]
            new_material_names = [material_name for material_name in material_names if material_name not in exsist_material_names]
            for material_name in new_material_names:
                bpy.data.materials[material_name].node_tree.nodes["Principled BSDF"].inputs[0].default_value = (random_color[0], random_color[1], random_color[2], 1)
                bpy.data.materials[material_name].node_tree.nodes["Principled BSDF"].inputs[7].default_value = 0.9

    @staticmethod
    def _clear():
        #Existing mesh, light, camera,Delete all
        # for item in bpy.data.objects:
        #     bpy.data.objects.remove(item)
        bpy.ops.wm.read_factory_settings(use_empty=True)

    @staticmethod
    def _add_world_voronoi_texture():
        bpy.context.scene.world.use_nodes = True
        # voronoitex=nodes.new("ShaderNodeTexVoronoi")
        bpy.context.scene.world.node_tree.nodes.new("ShaderNodeTexVoronoi")
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value = 2
        bpy.data.worlds["World"].node_tree.nodes["Voronoi Texture"].feature = 'SMOOTH_F1'
        bpy.data.worlds["World"].node_tree.nodes["Voronoi Texture"].voronoi_dimensions = '4D'
        bpy.data.worlds["World"].node_tree.nodes["Voronoi Texture"].inputs[2].default_value = 1.3
        bpy.data.worlds["World"].node_tree.nodes["Voronoi Texture"].inputs[3].default_value = 2
        bpy.context.scene.world.node_tree.links.new(
            bpy.context.scene.world.node_tree.nodes['Background'].inputs[0],
            bpy.context.scene.world.node_tree.nodes['Voronoi Texture'].outputs[1]
        )

    @staticmethod
    def _render_rgb_and_save(result_path):
        if bpy.context.scene.world is None:
            # create a new world
            new_world = bpy.data.worlds.new("New World")
            bpy.context.scene.world = new_world
        # bpy.context.scene.world.color = (1.0, 1.0, 1.0) # realy 128, 128, 128
        bpy.context.scene.world.color = (0.215861, 0.215861, 0.215861)
        bpy.context.scene.view_settings.view_transform = 'Standard'
        # rgb
        bpy.context.scene.render.filepath = result_path
        bpy.context.scene.render.image_settings.file_format = 'JPEG'
        bpy.ops.render.render(write_still=True)
    
    @staticmethod
    def _render_silhouettes_and_save(result_path):
        # silhouettes
        for item in bpy.data.objects:
            if item.type == 'LIGHT':
                bpy.data.objects.remove(item)
        material_names = bpy.data.materials.keys()
        material_names = [material_name for material_name in material_names if material_name not in ['Dots Stroke']]
        for material_name in material_names:
            bpy.data.materials[material_name].node_tree.nodes["Principled BSDF"].inputs[21].default_value = 0
        bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (1, 1, 1, 1)
        bpy.data.worlds["World"].node_tree.links.remove(
            bpy.context.scene.world.node_tree.nodes['Background'].inputs[0].links[0]
        )
        bpy.context.scene.render.filepath = result_path
        bpy.ops.render.render(write_still=True)

def run_generator(annotations_path, dataset_path):
    lines = load_gtformat_file(os.path.join(annotations_path, 'annotation.gtf'))
    lines = [line for line in lines if line['category'] in TARGET_CATEGORY]
    # enviroment
    for idx, line in enumerate(lines):
        mobility_describe_list, mobility_parts_list = articulated_parser(
            os.path.join(dataset_path, line['id'], 'mobility_v2.json'),
            os.path.join(dataset_path, line['id'], 'result.json')
        )

def articulated_parser(mob_path, res_path):
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
    return mobility_describe_list, mobility_parts_mapping

def load_gtformat_file(fpath):
    assert os.path.exists(fpath)
    with open(fpath,'r') as fid:
        lines = fid.readlines()
        lines = [json.loads(line) for line in lines]
    return lines

if __name__ == '__main__':
    delete = [
        '100731', '101378', '101380', '101384', '102153', 
        '100732', '102171', '102181', '102187', '10357',
        '11229', '11259', '11951', '12231', '12289',
        '12477', '102193', '102218', '102227', '102234', 
        '102244', '102252', '102996', '103008', '103007', 
        '103010', '103013', '4108', '102219', '11818',
        '12445', '12447', '102194', '103633', '103634', 
        '102163', '11124', '103639', '103647', '102165',
        '10584', '102173', '102209', '102257', '102992',
        '103635', '102160', '102182', '102186'
    ] #['103283']
    lines = load_gtformat_file('./annotations/annotation.gtf')
    lines = [line for line in lines if line['category'] in TARGET_CATEGORY]
    lines = [line for line in lines if line['id'] not in delete]
    NUMBER = 8000 // len(lines)
    done = [
        '102156', '102158', '102192', '103012', '102202', 
        '102210', '102229', '102256', '102259'
    ]
    lines = [line for line in lines if line['id'] not in done]
    # enviroment
    for idx, line in enumerate(lines):
        data_path = os.path.join('./dataset', line['id'])
        mobility_describe_list, mobility_parts_mapping = articulated_parser(
            os.path.join(data_path, 'mobility_v2.json'),
            os.path.join(data_path, 'result.json')
        )
        arti_reverse = ['102154', '102155', '102156', '102158', '102182', '102189', '102210', '102254']
        data_factory = DataFactory(arti_reverse=(line['id'] in arti_reverse))
        data_factory.set_model_path(data_path, mobility_describe_list, mobility_parts_mapping)
        for _ in range(NUMBER):
            data_factory.run_generate(line['id'])
            # break
        # break