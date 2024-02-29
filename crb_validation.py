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
import pandas as pd
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel
# Import Sionna RT components
from mysionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Scene
from mysionna.rt.scattering_pattern import *


#read config and turn it to global variables
config = json.load(open("crb_validation_config.json"))
subcarrier_spacing = config.get("subcarrier_spacing")
subcarrier_num = config.get("subcarrier_num")
num_time_steps = config.get("num_time_steps")
ebno_db = config.get("ebno_db")
num_samples = config.get("num_samples")
batch_size = config.get("batch_size")
max_depth = config.get("max_depth")
step = config.get("step")
los = config.get("los")
reflection = config.get("reflection")
scattering = config.get("scattering")
diffraction = config.get("diffraction")
edge_diffraction = config.get("edge_diffraction")

scene_info = [
    {
        "scene_name":"indoor",
        "paths":["./scenes/Indoor/indoor1.xml","./scenes/Indoor/indoor.xml"],
        "tgname":["human1"],
        "tgv":[(-0.0,0,0)],
        "map_center":[0,0,2.95],
        "map_size_x":10,
        "map_size_y":6,
        "cell_size":0.5,
        "look_at":[-3.37234,2.18367,1.20838],
    },
    {
        "scene_name":"indoor",
        "paths":["./scenes/Indoor/indoor2.xml","./scenes/Indoor/indoor.xml"],
        "tgname":["human2"],
        "tgv":[(0,-0.0,0)],
        "map_center":[0,0,2.95],
        "map_size_x":10,
        "map_size_y":6,
        "cell_size":0.5,
        "look_at":[-2.81027,-1.92977,1.20838]
    },
    {
        "scene_name":"indoor",
        "paths":["./scenes/Indoor/indoor3.xml","./scenes/Indoor/indoor.xml"],
        "tgname":["human3"],
        "tgv":[(0.0,0,0)],
        "map_center":[0,0,2.95],
        "map_size_x":10,
        "map_size_y":6,
        "cell_size":0.5,
        "look_at":[2.97116,-0.235489,1.20838]
    }
]

tf.random.set_seed(1) # Set global random seed for reproducibility


def CSI(scene:Scene,info,cell_pos,return_tau=False):
    look_at = info.get("look_at")
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
            tgname = info.get("tgname")
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
    scene.get("itu_plywood").scattering_coefficient = 0.3
    scene.get("itu_plywood").scattering_pattern = p1
    scene.get("itu_concrete").scattering_coefficient = 0.5
    scene.get("itu_concrete").scattering_pattern = p1
    scene.get("itu_glass").scattering_coefficient = 0.25
    scene.get("itu_glass").scattering_pattern = p2
    #################配置感知目标#################
    if tgname is not None and tgv is not None:
        scene.target_names = tgname
        scene.target_velocities = tgv
    
    return scene
        

def getRayType():
    ray_type=""
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
    

def saveFig(title,tau_true,tau_est,crb,col):
    pad = 0
    tau_true = np.array(tau_true)
    tau_est = np.array(tau_est)
    crb = crb*3e8
    crb = np.log10(crb)
    crb = np.reshape(crb,(-1,col))
    mse = np.abs(tau_true-tau_est)
    mse = np.reshape(mse,(-1,col))
    tau_true = np.reshape(tau_true,(-1,col))
    tau_est = np.reshape(tau_est,(-1,col))
    mask = mse >= 2
    mse[mask] = pad
    mask = tau_true==999999 
    mse[mask] = pad
    mask = tau_true==0
    mse[mask] = pad
    mask = tau_true==-1
    mse[mask] = pad
    mse = mse*3e8
    mse = np.log10(mse)
    
    mse = np.rot90(mse)
    crb = np.rot90(crb)
    # x轴对称
    mse = np.flip(mse,1)
    crb = np.flip(crb,1)
    mse = np.flip(mse,0)
    crb = np.flip(crb,0)

    # set figure size
    plt.figure(figsize=(10, 5))
    plt.subplots_adjust(wspace=0.5)
    plt.subplot(121)
    plt.title("Indoor Sensing MSE")
    plt.xlabel("x axis(0.5m)")
    plt.ylabel("y axis(0.5m)")
    plt.imshow(mse)
    # set colorbar size
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.subplot(122)
    plt.title("Indoor Sensing CRB")
    plt.xlabel("x axis(0.5m)")
    plt.ylabel("y axis(0.5m)")
    plt.imshow(crb)
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(f"{title}/out.png")


def main():
    saved = 0
    frequencies = subcarrier_frequencies(subcarrier_num, subcarrier_spacing)
    for info in scene_info:
        scene_name = info.get("scene_name")
        scene1 = info.get("paths")[0]
        scene_env = info.get("paths")[1]
        map_center = info.get("map_center")
        x = info.get("map_size_x")
        y = info.get("map_size_y")
        cell_size = info.get("cell_size")
        look_at = info.get("look_at")
        tgname = info.get("tgname")
        tgv = info.get("tgv")
        look_at = info.get("look_at")
        ray_type = getRayType()
        env_title = f"./Data/{scene_name}/{num_samples}-{ray_type}-{max_depth}-{x}-{y}-{cell_size}"
        title = f"./Data/{scene_name}/{tgname[0]}/{num_samples}-{ray_type}-{max_depth}-{step}-{batch_size}-{x}-{y}-{cell_size}"
        print(f"scene: {scene_name}, tgname: {tgname[0]}")
        
        # create folder
        if not os.path.exists(f"{title}"):
            os.makedirs(f"{title}")
        
        # 计算环境杂波信道
        if os.path.exists(f"{env_title}_env.npy"):
            h_list2 = np.load(f"{env_title}_env.npy",allow_pickle=True)
        else:
            print("computing env csi...")
            scene = setScene(scene_env)
            if info.get("pos") is not None:
                cell_pos = info.get("pos")
            else:
                cell_pos = getPos(map_center,x,y,cell_size)
            h_list2 = CSI(scene,info,cell_pos)
            h_np = np.stack(h_list2)
            np.save(f"{env_title}_env.npy",h_np)
            print("saved environment info")
        
        # 读取仿真进度
        if os.path.exists(f"{title}/saved.txt"):
            with open(f"{title}/saved.txt","r") as f:
                saved = int(f.read())
            scene = setScene(scene1,tgname,tgv)
            if info.get("pos") is not None:
                cell_pos = info.get("pos")
            else:
                cell_pos = getPos(map_center,x,y,cell_size)
        else:
            saved = 0
        
        # 计算含目标的CSI
        if saved == 0:
            print("computing target csi...")
            scene = setScene(scene1,tgname,tgv)
            if info.get("pos") is not None:
                cell_pos = info.get("pos")
            else:
                cell_pos = getPos(map_center,x,y,cell_size)
            h_list1,tau_true = CSI(scene,info,cell_pos,return_tau=True)
            
            # save data
            with open(f"{title}/tau_true.txt","w") as f:
                for i in range(len(tau_true)):
                    f.write(f"{tau_true[i]}\n")
            h_np = np.stack(h_list1)
            np.save(f"{title}/h.npy",h_np)
            print("saved")
            saved += 1
            with open(f"{title}/saved.txt","w") as f:
                f.write(f"{saved}")
        else:
            h_list1 = np.load(f"{title}/h.npy",allow_pickle=True)
            # 读取tau_true
            tau_true = []
            with open(f"{title}/tau_true.txt","r") as f:
                tau_true = f.readlines()
                tau_true = np.array([float(i) for i in tau_true])
            
        # 计算crb
        if saved == 1:
            print("computing crb...")
            if scene.get("tx") is not None:
                scene.remove("tx")
            if scene.get("rx") is not None:
                scene.remove("rx")
            crbs = scene.coverage_map_sensing(cell_pos=cell_pos,
                                    look_at=look_at,
                                    batch_size=batch_size,
                                    singleBS=True,
                                    num_samples=num_samples*batch_size,
                                    max_depth=max_depth,
                                    los=los,
                                    reflection=reflection,
                                    scattering=scattering,
                                    diffraction=diffraction,
                                    edge_diffraction=edge_diffraction,
                                    num_time_steps=num_time_steps)
            
            crb = None
            for i in range(0,len(crbs)):
                c = crbs[i][0]
                c = tf.squeeze(c)
                # c = tf.linalg.diag_part(c)
                c = c.numpy()
                if crb is None:
                    crb = c
                else:
                    crb = np.concatenate((crb,c),axis=None)
            crb = np.array(crb)
            np.save(f"{title}/crb.npy",crb)
            print("saved")
            saved += 1
            with open(f"{title}/saved.txt","w") as f:
                f.write(f"{saved}")
        else:
            crb = np.load(f"{title}/crb.npy")
        
        # 计算music估计值
        if saved == 2:
            print("music...")
            tau_est = []
            i = 0
            bar = tqdm.tqdm(total=len(tau_true))
            for i in range(len(tau_true)):
                tau = tau_true[i]
                h1 = h_list1[i]
                h2 = h_list2[i]
                h = h1-h2
                tau = tau*1e9
                start = tau-step*500
                if start < 0:
                    start = 0
                end = tau+step*500
                try:
                    t = music(h,frequencies,start=start,end=end,step=step)
                except:
                    t = 0
                tau_est.append(t)
                bar.update(1)
            
            # write tau_est and mse to file
            with open(f"{title}/tau_est.txt","w") as f:
                for i in range(len(tau_est)):
                    f.write(f"{tau_est[i]}\n")
            saved += 1
            with open(f"{title}/saved.txt","w") as f:
                f.write(f"{saved}")
            print("saved")
        else:
            tau_est = []
            with open(f"{title}/tau_est.txt","r") as f:
                tau_est = f.readlines()
                tau_est = np.array([float(i) for i in tau_est])
        
        saveFig(title,tau_true,tau_est,crb,int(y/cell_size)+1)
                
        print ("done")
        

if __name__ == "__main__":
    main()