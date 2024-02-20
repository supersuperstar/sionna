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
import sionna
import tqdm
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel
# Import Sionna RT components
from mysionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Scene
from mysionna.rt.scattering_pattern import *


scenes = [["./Indoor/indoor.xml","./IndoorEnv/indoor-env.xml"]]
scene_info = {"indoor.xml":{
    "tgname":["human"],
    "tgv":[(0,0,0)],
    "map_center":[0,0,2.95],
    "map_size_x":6,
    "map_size_y":10,
    "cell_size":0.5,
    "look_at":[1.9,2.7,1.0],
}}

subcarrier_spacing = 15e3
subcarrier_num = 2048
num_time_steps = 1
ebno_db = 30


def CSI(scene:Scene,scene_name,cell_pos,return_tau=False):
    print("computing csi...")
    look_at = scene_info.get(scene_name).get("look_at")
    h = []
    tau_true = []
    for pos in tqdm.tqdm(cell_pos):
        # Set the transmitter and receiver
        tx = Transmitter(name='tx',position=pos)
        rx = Receiver(name='rx',position=pos)
        tx.look_at(look_at)
        tx.look_at(look_at)
        if scene.get("tx") is not None:
            scene.remove("tx")
        scene.add(tx)
        if scene.get("rx") is not None:
            scene.remove("rx")
        scene.add(rx)
        # Compute the channel impulse response
        paths = scene.compute_paths(max_depth=3,diffraction=True,scattering=True,edge_diffraction=True,num_samples=1000000)
        if return_tau:
            v,obj_name = scene.compute_target_velocities(paths, return_obj_names=True)
            paths.apply_doppler(sampling_frequency=subcarrier_spacing, num_time_steps=num_time_steps,target_velocities=v)
        else:
            paths.apply_doppler(sampling_frequency=subcarrier_spacing, num_time_steps=num_time_steps)
        a, tau = paths.cir()
        frequencies = subcarrier_frequencies(subcarrier_num, subcarrier_spacing)
        h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
        h.append(h_freq)
        # 记录真实tau
        if return_tau:
            tau_true.append(999999.0)
            tgname = scene_info.get(scene_name).get("tgname")
            for name in tgname:
                mask = tf.equal(obj_name, name)
                mask = tf.reduce_any(mask, axis=0)
                tau = tf.squeeze(tau)
                mask = tf.squeeze(mask)
                tau_obj = tf.gather(tau, tf.where(mask))
                tau_true[-1] = min(tau_true[-1],tf.reduce_min(tau_obj))
        
    if return_tau:
        return h,tau_true
    return h


def music(h_freq,frequencies,start = 0,end = 400,step = 0.1):
    y_i = h_freq[0,0,0,0,0,0,:]
    y_i = tf.squeeze(y_i)
    y_i = tf.expand_dims(y_i, axis=0)
    y_i_H = tf.transpose(tf.math.conj(y_i))
    y_conv = tf.matmul(y_i_H, y_i)
    _, eig_vecs = tf.linalg.eigh(y_conv)
    tau_range = np.arange(start,end, step)
    frequencies_c = tf.cast(frequencies, dtype=tf.complex64)

    P_tau_array = tf.TensorArray(dtype=tf.complex64, size=len(tau_range))
    G_n = tf.cast(eig_vecs[:,:-1], dtype=tf.complex64)
    G_n_H = tf.math.conj(tf.transpose(G_n))
    for idx in range(len(tau_range)):
        a_m = tf.expand_dims(tf.math.exp(-1j * 2 * np.pi * frequencies_c * (tau_range[idx]/1e9)), axis=0)
        a_m_H = tf.math.conj(tf.transpose(a_m))
        P_tau_array = P_tau_array.write(idx, 1 / (a_m @ G_n @ G_n_H @ a_m_H))

    P_tau = P_tau_array.stack()
    # 计算谱函数
    P_tau_real, P_tau_imag = tf.math.real(P_tau), tf.math.imag(P_tau)
    P_abs = tf.math.sqrt(P_tau_real**2 + P_tau_imag**2)
    P_norm = 10 * tf.math.log(P_abs / tf.reduce_max(P_abs), 10)
    P_norm = tf.squeeze(P_norm)
    max_idx = tf.argmax(P_norm)
    tau_est = (start + int(max_idx) * step)
    return tau_est


def getPos(scene_name):
    map_center = scene_info.get(scene_name).get("map_center")
    map_size_x = scene_info.get(scene_name).get("map_size_x")
    map_size_y = scene_info.get(scene_name).get("map_size_y")
    cell_size = scene_info.get(scene_name).get("cell_size")
    # compute cell positions
    cell_num_x = int(map_size_x/cell_size) + 1 # Number of x cells in the map
    cell_num_y = int(map_size_y/cell_size) + 1 # Number of y cells in the map
    cell_positions = np.zeros((cell_num_x, cell_num_y, 3))
    # fill x
    x_fill = np.arange(0,cell_num_x) * cell_size + map_center[0] - map_size_x/2
    x_fill = np.tile(x_fill,cell_num_y)
    cell_positions[:,:,0] = x_fill.reshape([cell_num_y,cell_num_x]).transpose()
    # fill y
    y_fill = np.arange(0,cell_num_y) * cell_size + map_center[1] - map_size_y/2
    y_fill = np.tile(y_fill,cell_num_x)
    cell_positions[:,:,1] = y_fill.reshape([cell_num_x,cell_num_y])
    # fill z
    cell_positions[:,:,2] = np.tile(map_center[2],(cell_num_x,cell_num_y))
    cell_pos = tf.constant(cell_positions, dtype=tf.float32)
    # [num_cells_x*num_cells_y, 3]
    cell_pos = tf.reshape(cell_pos, [-1, 3]) 
    return cell_pos


def setScene(filename,tgname=None,tgv=None):
    # Set the scene
    scene = load_scene(filename)
    #################配置发端天线阵列#################
    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="dipole",
                                polarization="V")

    #################配置收端天线阵列#################
    scene.rx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="dipole",
                                polarization="V")
    scene.frequency = 2.14e9 # in Hz; implicitly updates RadioMaterials
    scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)
    #################配置场景材质属性#################
    p1 = LambertianPattern()
    p2 = DirectivePattern(20)
    scene.get("itu_plywood").scattering_coefficient = 0.4
    scene.get("itu_plywood").scattering_pattern = p2
    scene.get("itu_concrete").scattering_coefficient = 0.3
    scene.get("itu_concrete").scattering_pattern = p1
    scene.get("itu_floorboard").scattering_coefficient = 0.4
    scene.get("itu_floorboard").scattering_pattern = p2
    scene.get("itu_ceiling_board").scattering_coefficient = 0.4
    scene.get("itu_ceiling_board").scattering_pattern = p2
    #################配置感知目标#################
    if tgname is not None and tgv is not None:
        scene.target_names = tgname
        scene.target_velocities = tgv
    
    return scene
        

def main():
    frequencies = subcarrier_frequencies(subcarrier_num, subcarrier_spacing)
    for scene_pair in scenes:
        scene1 = scene_pair[0]
        scene2 = scene_pair[1]
        scene_name = scene1.split("/")[-1]
        
        # scene1
        tgname = scene_info.get(scene_name).get("tgname")
        tgv = scene_info.get(scene_name).get("tgv")
        scene = setScene(scene1,tgname,tgv)
        cell_pos = getPos(scene_name)
        h_list1,tau_true = CSI(scene,scene_name,cell_pos,return_tau=True)
        
        with open(f"{scene_name}_tau_true.txt","w") as f:
            for i in range(len(tau_true)):
                f.write(f"{tau_true[i]}\n")
    
        # scene2
        scene = setScene(scene2)
        scene.target_names = None
        scene.target_velocities = None
        cell_pos = getPos(scene_name)
        h_list2 = CSI(scene,scene_name,cell_pos)

        print("music...")
        for i,(h1,h2) in tqdm(enumerate(zip(h_list1,h_list2))):
            h = h1-h2
            tau_est = music(h,frequencies)
        
        # write tau_est to file
        with open(f"{scene_name}_tau_est.txt","w") as f:
            f.write(f"{tau_est}\n")


if __name__ == "__main__":
    main()