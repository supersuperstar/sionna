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
# import pandas as pd
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel
# Import Sionna RT components
from mysionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Scene
from mysionna.rt.scattering_pattern import *
from mysionna.rt.scene import Target,load_sensing_scene


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
scat_keep_prob = config.get("scat_keep_prob")

scene_info = [
    {
        "scene_name":"indoor", # 场景名称
        "paths":"./scenes/Street/street.xml", # 场景路径
        "tgpath":"meshes/car.ply", # 目标路径
        "tgmat":"itu_metal", # 目标材质
        "tgname":"car", # 目标名称
        "tgpos":[[0,0,0]], # 目标位置
        "tgscales":[[1,1,1]], # 目标缩放
        "tgrots":[[0,0,0,0]], # 目标旋转
        "tgvs":[[0,0,0]], # 目标速度
        "map_center":[0,0,30],
        "map_size_x":20,
        "map_size_y":20,
        "cell_size":10,
    },
    # {
    #     "scene_name":"street", # 场景名称
    #     "paths":"./scenes/Street/street.xml", # 场景路径
    #     "tgpath":"meshes/car.ply", # 目标路径
    #     "tgmat":"itu_metal", # 目标材质
    #     "tgname":"car", # 目标名称
    #     "tgpos":[[0,0,0],[-14,28,0],[14,28,0]], # 目标位置
    #     "tgscales":[[1,1,1],[1,1,1]], # 目标缩放
    #     "tgrots":[[0,0,0,0],[0,0,0,0]], # 目标旋转
    #     "tgvs":[[0,10,0],[0,10,0],[0,10,0]], # 目标速度
    #     "cell_pos":[[0,0,6],[-20,27,6],[20,27,6],[20,-27,6],[-20,-27,6],[-10,-27,6],[-10,-8,6]], # 位置信息
    # }
]

tf.random.set_seed(1) # Set global random seed for reproducibility


def CSI(scene:Scene,info,cell_pos,look_at,return_tau=False,tgname=None):
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
        paths = scene.compute_paths(max_depth=max_depth,los=los,reflection=reflection,diffraction=diffraction,scattering=scattering,edge_diffraction=edge_diffraction,num_samples=num_samples,scat_keep_prob=scat_keep_prob)
        paths.normalize_delays = False
        if return_tau:
            v,obj_name = scene.compute_target_velocities(paths, return_obj_names=True)
            paths.apply_doppler(sampling_frequency=subcarrier_spacing, num_time_steps=num_time_steps,target_velocities=v)
            # split obj_name by '&'
            # obj_name = [i.split("&") for i in obj_name]
        else: 
            paths.apply_doppler(sampling_frequency=subcarrier_spacing, num_time_steps=num_time_steps)
        a, tau = paths.cir()
        frequencies = subcarrier_frequencies(subcarrier_num, subcarrier_spacing)
        h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=False)
        h.append(h_freq)
        # 记录真实tau
        if return_tau:
            tau_true.append(99999)
            mask = tf.equal(obj_name, tgname)
            mask = tf.reduce_any(mask, axis=0)
            tau = tf.squeeze(tau)
            mask = tf.squeeze(mask)
            tau_obj = tf.gather(tau, tf.where(mask))
            tau_obj = tf.gather(tau_obj,tf.where(tau_obj>0))
            if tau_obj.shape[0] > 0:
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

    G_n = tf.cast(eig_vecs[:,:-1], dtype=tf.complex64)
    G_n_H = tf.math.conj(tf.transpose(G_n))
    frequencies_c = tf.expand_dims(frequencies, axis=0)
    frequencies_c = tf.repeat(frequencies_c, len(tau_range), axis=0)
    frequencies_c = tf.cast(frequencies_c, dtype=tf.complex64)
    tau_range = tf.expand_dims(tau_range, axis=-1)
    tau_range = tf.repeat(tau_range, subcarrier_num, axis=-1)
    tau_range = tf.cast(tau_range, dtype=tf.complex64)
    a_m = tf.math.exp(-1j * 2 * np.pi * frequencies_c * (tau_range/1e9))
    a_m_H = tf.math.conj(tf.transpose(a_m))
    a_m_H = tf.expand_dims(a_m_H, axis=1)
    a_m_H = tf.transpose(a_m_H, perm=[2,0,1])
    a_m = tf.expand_dims(a_m, axis=1)
    G_n = tf.expand_dims(G_n, axis=0)
    G_n_H = tf.expand_dims(G_n_H, axis=0)
    P = 1 / (a_m @ G_n @ G_n_H @ a_m_H)
    P = tf.squeeze(P)
    # 计算谱函数
    P_tau_real = tf.math.real(P)
    P_tau_imag = tf.math.imag(P)
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


def setScene(filename,tg=None,tgname=None,tgv=None):
    # Set the scene
    if tg is None:
        scene = load_scene(filename)
    else:
        scene= load_sensing_scene(filename,tg)
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
    if scene.get("itu_plywood") is not None:
        scene.get("itu_plywood").scattering_coefficient = 0.3
        scene.get("itu_plywood").scattering_pattern = p1
    if scene.get("itu_concrete") is not None:
        scene.get("itu_concrete").scattering_coefficient = 0.5
        scene.get("itu_concrete").scattering_pattern = p1
    if scene.get("itu_glass") is not None:
        scene.get("itu_glass").scattering_coefficient = 0.25
        scene.get("itu_glass").scattering_pattern = p2
    if scene.get("itu_medium_dry_ground") is not None:
        scene.get("itu_medium_dry_ground").scattering_coefficient = 0.8
        scene.get("itu_medium_dry_ground").scattering_pattern = p1
    if scene.get("itu_metal") is not None:
        scene.get("itu_metal").scattering_coefficient = 0.1
        scene.get("itu_metal").scattering_pattern = p2
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
    

def saveFig(info,title,tau_true,tau_est,crb,tgpos):
    map_center = info.get("map_center")
    x = info.get("map_size_x")
    y = info.get("map_size_y")
    cell_size = info.get("cell_size")
    pad = 0
    tau_true = np.array(tau_true)
    tau_est = np.array(tau_est)
    mse = np.abs(tau_true-tau_est)
    np.save(f"{title}/mse_{step}.npy",mse)
        
    cell_pos = info.get("cell_pos")
    if cell_pos is None:
        mask = mse >= 1
        mse[mask] = pad
        mask = tau_true>=1
        mse[mask] = pad
        mask = tau_true==0
        mse[mask] = pad
        mask = tau_true==-1
        mse[mask] = pad
        tgpos_x = (tgpos[0]-map_center[0]+x/2)/cell_size
        tgpos_y = y/cell_size-(tgpos[1]-map_center[1]+y/2)/cell_size
        col = int(y/cell_size) + 1
        crb = np.reshape(crb,(-1,col))
        mse = np.reshape(mse,(-1,col))
        mse = np.rot90(mse)
        crb = np.rot90(crb)
        # x轴对称
        # mse = np.flip(mse,1)
        # crb = np.flip(crb,1)
        # mse = np.flip(mse,0)
        # crb = np.flip(crb,0)

        crb_lg = np.log10(crb)
        mse_lg = np.log10(mse)
        # crb_max = np.max(crb)
        # crb_min = np.min(crb)
        plt.figure(figsize=(18, 4))
        plt.subplots_adjust(wspace=0.25)
        plt.subplot(131)
        plt.title("Sensing MSE")
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        plt.imshow(mse_lg)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.scatter(tgpos_x,tgpos_y,marker='x',color='r')
        plt.subplot(132)
        plt.title("Sensing CRB")
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        plt.imshow(crb_lg)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.scatter(tgpos_x,tgpos_y,marker='x',color='r')
        plt.subplot(133)
        plt.title("Sensing MSE/CRB")
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        plt.imshow(np.log10(mse/crb))
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.scatter(tgpos_x,tgpos_y,marker='x',color='r')
        plt.savefig(f"{title}/out.png")
    else:
        # 3维散点图
        cell_pos = np.array(cell_pos)
        
        plt.figure(figsize=(10, 5))
        plt.subplots_adjust(wspace=0.5)
        plt.subplot(121)
        plt.scatter(cell_pos[:,0],cell_pos[:,1],c=mse,alpha=0.5)
        plt.scatter(tgpos[0],tgpos[1],marker='x',color='r')
        min_mse_id = np.argmin(mse)
        plt.scatter(cell_pos[min_mse_id,0],cell_pos[min_mse_id,1],marker='*',color='r')
        plt.colorbar()
        plt.title(f"Sensing MSE\nBest BS pos:{cell_pos[min_mse_id]}")
        plt.subplot(122)
        plt.scatter(cell_pos[:,0],cell_pos[:,1],c=crb,alpha=0.5)
        plt.scatter(tgpos[0],tgpos[1],marker='x',color='r')
        min_crb_id = np.argmin(crb)
        plt.scatter(cell_pos[min_crb_id,0],cell_pos[min_crb_id,1],marker='*',color='r')
        plt.colorbar()
        plt.title(f"Sensing CRB\nBest BS pos:{cell_pos[min_crb_id]}")
        plt.savefig(f"{title}/out.png")


def main():
    frequencies = subcarrier_frequencies(subcarrier_num, subcarrier_spacing)
    for info in scene_info:
        scene_name = info.get("scene_name")
        scene_path = info.get("paths")
        map_center = info.get("map_center")
        x = info.get("map_size_x")
        y = info.get("map_size_y")
        cell_size = info.get("cell_size")
        tgpath = info.get("tgpath")
        tgname = info.get("tgname")
        tgposes = info.get("tgpos")
        tgvs = info.get("tgvs")
        tgmat = info.get("tgmat")
        ray_type = getRayType()
        
        # 获取位置信息-----------------------------------
        if x is None or y is None or cell_size is None or map_center is None:
            cell_pos = info.get("cell_pos")
            if cell_pos is None:
                print("error: no map info")
                continue
            cell_pos = tf.constant(cell_pos, dtype=tf.float32)
            cell_pos = tf.reshape(cell_pos, [-1, 3])
            cell_num = cell_pos.shape[0]
            env_title = f"./Data/{scene_name}/{num_samples}-{ray_type}-{max_depth}-{subcarrier_num}-{cell_num}-{cell_pos[0]}-{cell_pos[-1]}"
        else:
            cell_pos = getPos(map_center,x,y,cell_size)
            env_title = f"./Data/{scene_name}/{num_samples}-{ray_type}-{max_depth}-{subcarrier_num}-{x}-{y}-{cell_size}"
        
        if not os.path.exists(f"./Data/{scene_name}/"):
            os.makedirs(f"./Data/{scene_name}/")
        np.save(f"{env_title}_pos.npy",cell_pos)
        
        for i,tgpos in enumerate(tgposes):
            try:
                tgv = tgvs[i]
            except:
                tgv = [0,0,0]
            if info.get("cell_pos") is not None:
                title = f"./Data/{scene_name}/{tgname}/{tgpos}/{tgv}/{num_samples}-{ray_type}-{max_depth}-{subcarrier_num}-{cell_num}-{cell_pos[0]}-{cell_pos[-1]}"
            else:
                title = f"./Data/{scene_name}/{tgname}/{tgpos}/{tgv}/{num_samples}-{ray_type}-{max_depth}-{subcarrier_num}-{x}-{y}-{cell_size}"
            print(f"scene: {scene_name}, tgname: {tgname}, pos: {tgpos}, v: {tgv}")
            
            # create folder-----------------------------------
            if not os.path.exists(f"{title}"):
                os.makedirs(f"{title}")
            
            # 获取位置信息-----------------------------------
            if info.get("cell_pos") is None:
                cell_pos = getPos(map_center,x,y,cell_size)
            
            np.save(f"{title}/cell_pos.npy",cell_pos)
            # 计算环境杂波信道-----------------------------------
            if os.path.exists(f"{env_title}_env.npy"):
                h_list2 = np.load(f"{env_title}_env.npy",allow_pickle=True)
            else:
                print("computing env csi...")
                scene = setScene(scene_path)
                h_list2 = CSI(scene,info,cell_pos,tgpos)
                h_np = np.stack(h_list2)
                np.save(f"{env_title}_env.npy",h_np)
                print("saved environment info")
            # 设置目标场景信息-----------------------------------
            if len(info.get("tgpos")) <= i :
                translate = [0,0,0]
            else:
                translate = info.get("tgpos")[i]
                
            if len(info.get("tgscales")) <= i :
                scale = [1,1,1]
            else:
                scale = info.get("tgscales")[i]
                
            if len(info.get("tgrots")) <= i :
                rotate = [0,0,0,0]
            else:
                rotate=info.get("tgrots")[i]
                
            target = Target(tgpath, tgmat, translate=translate,scale=scale, rotate=rotate)
            scene = setScene(scene_path,target,[tgname],[tgv])
            
            # 计算含目标的CSI-----------------------------------
            if os.path.exists(f"{title}/h.npy") is False or os.path.exists(f"{title}/tau_true.txt") is False:
                print("computing target csi...")
                h_list1,tau_true = CSI(scene,info,cell_pos,tgpos,return_tau=True,tgname=tgname)
                
                # save data
                with open(f"{title}/tau_true.txt","w") as f:
                    for i in range(len(tau_true)):
                        f.write(f"{tau_true[i]}\n")
                h_np = np.stack(h_list1)
                np.save(f"{title}/h.npy",h_np)
                print("saved CSI info")
            else:
                h_list1 = np.load(f"{title}/h.npy",allow_pickle=True)
                # 读取tau_true
                tau_true = []
                with open(f"{title}/tau_true.txt","r") as f:
                    tau_true = f.readlines()
                    tau_true = np.array([float(i) for i in tau_true])
                
            # 计算crb-----------------------------------
            if os.path.exists(f"{title}/crb_{batch_size}.npy") is False:
                print("computing crb...")
                if scene.get("tx") is not None:
                    scene.remove("tx")
                if scene.get("rx") is not None:
                    scene.remove("rx")
                crbs = scene.coverage_map_sensing(only_target=True,
                                        cell_pos=cell_pos,
                                        look_at=tgpos,
                                        batch_size=batch_size,
                                        singleBS=True,
                                        num_samples=num_samples*batch_size,
                                        max_depth=max_depth,
                                        los=los,
                                        reflection=reflection,
                                        scattering=scattering,
                                        diffraction=diffraction,
                                        edge_diffraction=edge_diffraction,
                                        num_time_steps=num_time_steps,
                                        scat_keep_prob=scat_keep_prob,
                                        snr=ebno_db)
                
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
                np.save(f"{title}/crb_{batch_size}.npy",crb)
                print("saved crb")            
            else:
                crb = np.load(f"{title}/crb_{batch_size}.npy")
            
            # 计算music估计值-----------------------------------
            if os.path.exists(f"{title}/tau_est_{step}.txt") is False:
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
                    # start = tau-step*500
                    # if start < 0:
                    #     start = 0
                    # end = tau+step*500
                    start = 0
                    end = 2000
                    try:
                        t = music(h,frequencies,start=start,end=end,step=step)
                    except:
                        t = 0
                    tau_est.append(t)
                    bar.update(1)
                
                # write tau_est and mse to file
                with open(f"{title}/tau_est_{step}.txt","w") as f:
                    for i in range(len(tau_est)):
                        f.write(f"{tau_est[i]}\n")
            else:
                tau_est = []
                with open(f"{title}/tau_est_{step}.txt","r") as f:
                    tau_est = f.readlines()
                    tau_est = np.array([float(i) for i in tau_est])
                
            saveFig(info,title,tau_true,tau_est,crb,tgpos)
                    
            print ("done")
        

if __name__ == "__main__":
    main()