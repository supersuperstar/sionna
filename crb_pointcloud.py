import os
gpu_num = 0 # 使用 "" 来启用 CPU
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)
# Avoid warnings from TensorFlow
tf.get_logger().setLevel('ERROR')

tf.random.set_seed(1) # Set global random seed for reproducibility
import matplotlib.pyplot as plt
import numpy as np
import mitsuba as mi
import open3d as o3d

# Import Sionna RT components
from mysionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera
from mysionna.rt.scattering_pattern import *

scene = load_scene("Indoor/indoor.xml")

#################配置发端天线阵列#################
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="dipole",
                             polarization="H")

#################配置收端天线阵列#################
scene.rx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="dipole",
                             polarization="H")

################创建发射机#########################
tx = Transmitter(name="tx",
                 position=[0,0,2.95])
################ 将发射机加入到场景中##############
scene.add(tx)
#################创建接收机########################
rx = Receiver(name="rx",
              position=[0,0,2.95])
################ 将接收机加入到场景中##############
scene.add(rx)
tx.look_at([0,0,0])
rx.look_at([0,0,0])

scene.frequency = 2.14e9 # in Hz; implicitly updates RadioMaterials
scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)

scene.target_names = ["door"]
scene.target_velocities = [(1.,1.,0.)]

p1 = LambertianPattern()
p2 = DirectivePattern(50)
scene.get("itu_plywood").scattering_coefficient = 0.4
scene.get("itu_plywood").scattering_pattern = p2
scene.get("itu_concrete").scattering_coefficient = 0.7
scene.get("itu_concrete").scattering_pattern = p1
scene.get("itu_floorboard").scattering_coefficient = 0.3
scene.get("itu_floorboard").scattering_pattern = p2
scene.get("itu_ceiling_board").scattering_coefficient = 0.7
scene.get("itu_ceiling_board").scattering_pattern = p1

paths = scene.compute_paths(max_depth=3,diffraction=True,scattering=True,edge_diffraction=True,scat_keep_prob=0.01)

v,obj=scene.compute_target_velocities(paths, return_obj_names=True)

subcarrier_spacing = 15e3
num_time_steps = 14
paths.apply_doppler(sampling_frequency=subcarrier_spacing, num_time_steps=num_time_steps,target_velocities=v)

vertices = paths.vertices
vertices = tf.reshape(vertices,[-1,3])

print("")
# o3d.visualization.webrtc_server.enable_webrtc()
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(vertices)
o3d.visualization.draw_geometries([pcd])
