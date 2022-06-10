#!/usr/bin/env python3 
# -*- coding:utf-8 -*- 
# Copyright (c) Tencent, Inc. and its affiliates. 
# You can get the update in the repo: https://git.woa.com/xuangengchu/arc_detection.git. 
# If you have any questions about this repo, please contact @xuangengchu

import os
import sys
import math
import json
import yaml
import random
import argparse

import bpy
import numpy as np

class DataFactory:
    def __init__(self, category, obj_id, model_path, output_path, articulation_name):
        self.category = category
        self.obj_id = obj_id
        self.model_path = model_path
        self.output_path = output_path
        self.articulation_name = articulation_name
        self.mobility_describe_list, self.mobility_parts_mapping = articulated_parser(
            os.path.join(model_path, 'mobility_v2.json'),
            os.path.join(model_path, 'result.json')
        )

    def run_merge(self, theta, debug=False):
        commander = sys.stdout
        f = open(os.devnull, 'w')
        sys.stdout = f
        self._clear()
        self._import_all_obj(debug)
        self._articulation_deformation(self.mobility_describe_list, self.articulation_name, theta)
        self._join_all_obj(debug)
        # blend_file_path = bpy.data.filepath
        # directory = os.path.dirname(blend_file_path)
        if debug:
            if not os.path.exists(f'{self.output_path}'):
                os.makedirs(f'{self.output_path}')
            target_file = os.path.join(f'{self.output_path}/{self.obj_id}_{abs(theta)}.obj')
        else:
            if not os.path.exists(f'{self.output_path}/{self.category}/{self.obj_id}'):
                os.makedirs(f'{self.output_path}/{self.category}/{self.obj_id}')
            target_file = os.path.join(f'{self.output_path}/{self.category}/{self.obj_id}/{self.obj_id}_{abs(theta)}.obj')
        bpy.ops.export_scene.obj(filepath=target_file)
        sys.stdout = commander

    def _join_all_obj(self, debug=False):
        obs = []
        for ob in bpy.context.scene.objects:
            if ob.type == 'MESH':
                obs.append(ob)
        ctx = bpy.context.copy()
        # one of the objects to join
        ctx['active_object'] = obs[0]
        ctx['selected_objects'] = obs
        bpy.ops.object.join(ctx)

    def _import_all_obj(self, debug=False):
        for mobility_parts in self.mobility_describe_list:
            parts_group = []
            for part in mobility_parts['parts']:
                parts_group += self.mobility_parts_mapping[part['id']]['objs']
            parts_path_group = []
            for part in parts_group:
                parts_path_group.append(os.path.join(self.model_path, 'textured_objs', part+'.obj'))
            self._import_objs(parts_path_group, '{}_{}'.format(mobility_parts['name'], mobility_parts['id']))
        # bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children['Collection']

    @staticmethod
    def _articulation_deformation(mobility_describe_list, articulation_name, theta):
        articulation_description = []
        for mobility_parts in mobility_describe_list:
            articulation_description.append({
                'name': '{}_{}'.format(mobility_parts['name'], mobility_parts['id']),
                'joint': mobility_parts['jointData']
            })
        for part_description in articulation_description:
            part_name = part_description['name']
            part_joint = part_description['joint']
            arti_flag = False
            for arti_name in articulation_name:
                arti_flag = arti_flag | (arti_name in part_name)
            if not arti_flag:
                continue
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
    def _import_objs(model_path_list, collection_name):
        collection = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(collection)
        bpy.context.view_layer.active_layer_collection = bpy.context.view_layer.layer_collection.children[collection_name]
        for model_path in model_path_list:
            bpy.ops.import_scene.obj(filepath=model_path)
            bpy.ops.object.select_all(action = 'DESELECT')

    @staticmethod
    def _clear():
        bpy.ops.wm.read_factory_settings(use_empty=True)

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

def load_all_file_path(dir_path):
    pair = os.walk(dir_path)
    result = []
    for path, dirs, files in pair:
        if len(files):
            for file_name in files:
                result.append(os.path.join(path, file_name))
    return result

class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    @fr_andres
    https://blender.stackexchange.com/questions/6817/how-to-pass-command-line-arguments-to-a-blender-python-script
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """
    def _get_argv_after_doubledash(self):
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx+1:] # the list after '--'
        except ValueError as e: # '--' not in the list:
            return []

    def parse_args(self):
        return super().parse_args(args=self._get_argv_after_doubledash())


if __name__ == '__main__':
    parser = ArgumentParserForBlender()
    parser.add_argument('--config', '-c', type=str, default='merge_config.yaml')
    parser.add_argument('--target_category', '-t', required=True, type=str)
    parser.add_argument('--output_path', '-o', default='./results', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--theta', type=int)
    args = parser.parse_args()
    # Loading
    with open(args.config) as f:
        config_dict = yaml.load(f, Loader=yaml.Loader)
    lines = load_gtformat_file(config_dict['SAPIEN_DATA']['ANNO_PATH'])
    assert args.target_category in config_dict.keys(), config_dict.keys()
    lines = [line for line in lines if line['category'] == args.target_category]
    lines = [line for line in lines if line['id'] not in config_dict[args.target_category]['DELETE']]
    # Merging
    print('Start Merging...')
    for idx, line in enumerate(lines):
        data_path = os.path.join(config_dict['SAPIEN_DATA']['DATA_PATH'], line['id'])
        data_factory = DataFactory(args.target_category, line['id'], data_path, args.output_path, config_dict[args.target_category]['ARTI_NAME'])
        if not args.debug:
            for theta in range(config_dict[args.target_category]['THETA_RANGE'][0], config_dict[args.target_category]['THETA_RANGE'][1]+1):
                theta = theta if line['id'] not in config_dict[args.target_category]['REVERSE'] else -theta
                data_factory.run_merge(theta)
                print('{}/[{}, {}], {}/{}...'.format(
                    abs(theta), config_dict[args.target_category]['THETA_RANGE'][0], config_dict[args.target_category]['THETA_RANGE'][1],
                    idx+1, len(lines)
                ))
        else:
            theta = args.theta if line['id'] not in config_dict[args.target_category]['REVERSE'] else -args.theta
            data_factory.run_merge(theta, debug=True)
