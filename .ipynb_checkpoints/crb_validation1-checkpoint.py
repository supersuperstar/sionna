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
import matplotlib.pyplot as plt
import numpy as np
import sionna
import tqdm
import json
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel
# Import Sionna RT components
from mysionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Scene
from mysionna.rt.scattering_pattern import *
tf.random.set_seed(1) # Set global random seed for reproducibility


def main():
    config = loadConfig("config_list.json")
    turns = config.get("active")
    for turn_name in turns:
        config_turn = config.get(turn_name)
        simulation(config_turn)
        


def simulation(info):
    subcarrier_num = info.get("subcarrier_num")
    subcarrier_spacing = info.get("subcarrier_spacing")
    scene_name = info.get("scene_name")
    scene_path = info.get("paths")[0]
    scene_env_path = info.get("paths")[1]
    map_center = info.get("map_center")
    x = info.get("map_size_x")
    y = info.get("map_size_y")
    cell_size = info.get("cell_size")
    look_at = info.get("look_at")
    tgname = info.get("tgname")
    tgv = info.get("tgv")
    max_depth = info.get("max_depth")
    los = info.get("los")
    reflection = info.get("reflection")
    diffraction = info.get("diffraction")
    scattering = info.get("scattering")
    edge_diffraction = info.get("edge_diffraction")
    num_samples = info.get("num_samples")
    subcarrier_num = info.get("subcarrier_num")
    subcarrier_spacing = info.get("subcarrier_spacing")
    num_time_steps = info.get("num_time_steps")
    step = info.get("step")
    batch_size = info.get("batch_size")
    ray_type = getRayType(info)
    env_title = f"{num_samples}-{ray_type}-{max_depth}-{x}-{y}-{cell_size}"
    title = f"{num_samples}-{ray_type}-{max_depth}-{step}-{batch_size}-{x}-{y}-{cell_size}"
    # print simulation info
    print (f"""
            ################################################
            simulation info:
              scene_name: {scene_name}
              map: {x}x{y} at {map_center},cell size is {cell_size}
              target: {tgname}
              rayTracing args: {num_samples}/{max_depth}/{ray_type}
              batch_size: {batch_size}
              MUSIC args: {subcarrier_num}/{subcarrier_spacing}/{num_time_steps}
            ################################################
           """)
    
    frequencies = subcarrier_frequencies(subcarrier_num, subcarrier_spacing)

def CSI(scene:Scene,info,cell_pos,return_tau=False):
    look_at = info.get("look_at")
    max_depth = info.get("max_depth")
    los = info.get("los")
    reflection = info.get("reflection")
    diffraction = info.get("diffraction")
    scattering = info.get("scattering")
    edge_diffraction = info.get("edge_diffraction")
    num_samples = info.get("num_samples")
    subcarrier_num = info.get("subcarrier_num")
    subcarrier_spacing = info.get("subcarrier_spacing")
    num_time_steps = info.get("num_time_steps")
    tgname = info.get("tgname")[0]
    
    h = []
    tau_true = []
    for pos in tqdm.tqdm(cell_pos):
        # Set the transmitter and receiver
        tx = Transmitter(name='tx',position=pos)
        rx = Receiver(name='rx',position=pos)
        tx.look_at(look_at)
        rx.look_at(look_at)
        if scene.get("tx") is not None:
            scene.remove("tx")
        scene.add(tx)
        if scene.get("rx") is not None:
            scene.remove("rx")
        scene.add(rx)
        # Compute the channel impulse response
        paths = scene.compute_paths(max_depth=max_depth,los=los,reflection=reflection,diffraction=diffraction,scattering=scattering,edge_diffraction=edge_diffraction,num_samples=num_samples)
        paths.normalize_delays = False
        if return_tau:
            v,obj_name = scene.compute_target_velocities(paths, return_obj_names=True)
            paths.apply_doppler(sampling_frequency=subcarrier_spacing, num_time_steps=num_time_steps,target_velocities=v)
        else: 
            paths.apply_doppler(sampling_frequency=subcarrier_spacing, num_time_steps=num_time_steps)
        a, tau = paths.cir()
        frequencies = subcarrier_frequencies(subcarrier_num, subcarrier_spacing)
        h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=False)
        h.append(h_freq)
        # 记录真实tau
        if return_tau:
            tau_true.append(999999.0)
            mask = tf.equal(obj_name, tgname)
            mask = tf.reduce_any(mask, axis=0)
            tau = tf.squeeze(tau)
            mask = tf.squeeze(mask)
            tau_obj = tf.gather(tau, tf.where(mask))
            tau_true[-1] = min(tau_true[-1],tf.reduce_min(tau_obj))
        
    if return_tau:
        return h,tau_true
    return h


def music_delay(h_freq,frequencies,start = 0,end = 400,step = 0.1,tgnum = 1):
    y_i = h_freq[0,0,0,0,0,0,:]
    y_i = tf.squeeze(y_i)
    y_i = tf.expand_dims(y_i, axis=0)
    y_i_H = tf.transpose(tf.math.conj(y_i))
    y_conv = tf.matmul(y_i_H, y_i)
    _, eig_vecs = tf.linalg.eigh(y_conv)
    tau_range = np.arange(start,end, step)
    frequencies_c = tf.cast(frequencies, dtype=tf.complex64)

    P_tau_array = tf.TensorArray(dtype=tf.complex64, size=len(tau_range))
    G_n = tf.cast(eig_vecs[:,:-tgnum], dtype=tf.complex64)
    G_n_H = tf.math.conj(tf.transpose(G_n))
    for idx in range(len(tau_range)):
        a_m = tf.expand_dims(tf.math.exp(-1j * 2 * np.pi * frequencies_c * (tau_range[idx]/1e9)), axis=0)
        a_m_H = tf.math.conj(tf.transpose(a_m))
        P_tau_array = P_tau_array.write(idx, 1 / (a_m @ G_n @ G_n_H @ a_m_H))

    P_tau = P_tau_array.stack()
    # 计算谱函数
    P_tau_real = tf.math.real(P_tau)
    P_tau_imag = tf.math.imag(P_tau)
    P_abs = tf.math.sqrt(P_tau_real**2 + P_tau_imag**2)
    P_norm = 10 * tf.math.log(P_abs / tf.reduce_max(P_abs), 10)
    P_norm = tf.squeeze(P_norm)
    max_idx = tf.argmax(P_norm)
    tau_est = (start + int(max_idx) * step)
    return tau_est*1e-9


def getPos(map_center, map_size_x, map_size_y, cell_size):   
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
                                pattern="tr38901",
                                polarization="V")

    #################配置收端天线阵列#################
    scene.rx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="tr38901",
                                polarization="V")
    scene.frequency = 2.14e9 # in Hz; implicitly updates RadioMaterials
    scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)
    #################配置场景材质属性#################
    p1 = LambertianPattern()
    p2 = DirectivePattern(20)
    scene.get("itu_plywood").scattering_coefficient = 0.6
    scene.get("itu_plywood").scattering_pattern = p1
    scene.get("itu_concrete").scattering_coefficient = 0.8
    scene.get("itu_concrete").scattering_pattern = p1
    scene.get("itu_glass").scattering_coefficient = 0.25
    scene.get("itu_glass").scattering_pattern = p2
    #################配置感知目标#################
    if tgname is not None and tgv is not None:
        scene.target_names = tgname
        scene.target_velocities = tgv
    
    return scene


def getRayType(info):
    ray_type=""
    los = info.get("los")
    reflection = info.get("reflection")
    diffraction = info.get("diffraction")
    scattering = info.get("scattering")
    edge_diffraction = info.get("edge_diffraction")
    
    if los:
        ray_type += "T"
    else:
        ray_type += "F"
    if reflection:
        ray_type += "T"
    else:
        ray_type += "F"
    if scattering:
        ray_type += "T"
    else:
        ray_type += "F"
    if diffraction:
        ray_type += "T"
    else:
        ray_type += "F"
    if edge_diffraction:
        ray_type += "T"
    else:
        ray_type += "F"
    return ray_type


def loadConfig(filename):
    with open(filename, 'r') as f:
        config = json.load(f)
    return config