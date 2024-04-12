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
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel,cir_to_time_channel,time_to_ofdm_channel,time_lag_discrete_time_channel
from sionna.ofdm import ResourceGrid
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
rg = ResourceGrid(num_ofdm_symbols=num_time_steps,fft_size=subcarrier_num,subcarrier_spacing=subcarrier_spacing)
l_min,l_max = time_lag_discrete_time_channel(rg.bandwidth)

scene_info = [
    {
        "scene_name":"street", # 场景名称
        "paths":"./scenes/Street/street.xml", # 场景路径
        "tgpath":"meshes/car.ply", # 目标路径
        "tgmat":"itu_concrete", # 目标材质
        "tgname":"car", # 目标名称
        "tgv":[0,0,0], # 目标速度
        "bspos":[[32.8,35.2,50.3],[-30.3,93,20.8],[-121.4,33.2,8.9],[27.2,-143.9,8.6],[-25.3,-78.4,45.3],[141.6,-28.7,24.9]],# ,[0,-1,2.95],[-4.9,0,2.7],[4.9,0,2.7]], # 基站位置
        "map_center":[0,0,0.05],
        "map_size_x":200,
        "map_size_y":200,
        "cell_size":2,
    },
]

tf.random.set_seed(1) # Set global random seed for reproducibility


def CSI(scene:Scene,info,cell_pos,look_at,return_tau=False,tgname=None):
    h = []
    tau_true = []
    
    for id,pos in enumerate(cell_pos):
        # Set the transmitter and receiver
        tx = Transmitter(name=f'tx-{id}',position=pos)
        rx = Receiver(name=f'rx-{id}',position=pos)
        tx.look_at(look_at)
        rx.look_at(look_at)
        scene.add(tx)
        scene.add(rx)
        # 记录真实tau
        if return_tau:
            tau = np.linalg.norm(np.array(look_at)-np.array(pos))/1.5e8
            tau_true.append(tau)
    # Compute the channel impulse response
    paths = scene.compute_paths(max_depth=max_depth,los=los,reflection=reflection,diffraction=diffraction,scattering=scattering,edge_diffraction=edge_diffraction,num_samples=num_samples*len(cell_pos),scat_keep_prob=scat_keep_prob)
    paths.normalize_delays = False
    """# if return_tau:
    #     # v,obj_name = scene.compute_target_velocities(paths, return_obj_names=True)
    #     paths.apply_doppler(sampling_frequency=subcarrier_spacing, num_time_steps=num_time_steps,target_velocities=None)
    #     # split obj_name by '&'
    #     # obj_name = [i.split("&") for i in obj_name]
    #     pass
    # else: 
    #     paths.apply_doppler(sampling_frequency=subcarrier_spacing, num_time_steps=num_time_steps)"""
    cir = paths.cir()
    mask = scene.get_obj_mask(paths,True)[0]
    crb = paths.crb_delay(diag=True,mask=mask)
    crb_target = tf.where(mask, crb, 1)
    a = tf.where(mask,paths.a,0)
    # [batch_size,num_rx_ant,num_tx_ant,max_num_paths,num_time_steps,num_rx,num_tx]
    a = tf.transpose(a,perm=[0,2,4,5,6,1,3])
    # [batch_size,num_rx_ant,num_tx_ant,max_num_paths,num_time_steps,num_rx]
    a = tf.linalg.diag_part(a)
    # [batch_size,num_rx_ant,num_tx_ant,max_num_paths,num_time_steps,num_rx,1]
    a = tf.expand_dims(a, axis=-1)
    a = tf.transpose(a,perm=[0,5,1,6,2,3,4])
    a = tf.abs(a)
    crb_target = tf.reduce_min(crb_target, axis=6)
    crb_target = tf.reduce_min(crb_target, axis=4)
    crb_target = tf.reduce_min(crb_target, axis=2)
    a = tf.reduce_max(a, axis=6)
    a = tf.reduce_max(a, axis=4)
    a = tf.reduce_max(a, axis=2)
    a_sortidx = tf.argsort(a, axis=-1, direction='DESCENDING')
    a_max_idx = tf.gather(a_sortidx, 0, axis=-1)
    a_max_idx = tf.reshape(a_max_idx, [-1])
    crb_target = tf.gather(crb_target, a_max_idx, axis=-1)
    crb_target = tf.reshape(crb_target, [-1,a_max_idx.shape[0]])
    crb_target = tf.linalg.diag_part(crb_target)
    crb_target = tf.reshape(crb_target, [a.shape[0], a.shape[1], a.shape[2]])
    h_time = cir_to_time_channel(rg.bandwidth,*cir,l_min=l_min,l_max=l_max,normalize=True)
    h.append(h_time)
    if return_tau:
        return h,tau_true,crb_target
    return h


def music(h_freq,frequencies,start = 0,end = 3000,step = 0.1):
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
    # P_norm = 10 * tf.math.log(P_abs / tf.reduce_max(P_abs), 10)
    # P_norm = tf.squeeze(P_norm)
    max_idx = tf.argmax(P_abs)
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
    scene.frequency = 6e9 # in Hz; implicitly updates RadioMaterials
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
    

def saveFig(title,tau_True,tau_Est,Crb,col,step,bspos):
    for idx,pos in enumerate(bspos):
        tau_true = tau_True[:,idx]
        tau_est = tau_Est[:,idx]
        crb = Crb[:,idx]
        pad = 0
        tau_true = np.array(tau_true)
        tau_est = np.array(tau_est)
        # crb = crb*3e8
        crb = np.log10(crb)
        crb = np.reshape(crb,(-1,col))
        mse = np.abs(tau_true-tau_est)
        np.save(f"{title}/mse_{step}.npy",mse)
        mse = np.reshape(mse,(-1,col))
        tau_true = np.reshape(tau_true,(-1,col))
        tau_est = np.reshape(tau_est,(-1,col))
        mask = mse >= 0.1
        mse[mask] = pad
        mask = tau_true>=0.1
        mse[mask] = pad
        mask = tau_true==0
        mse[mask] = pad
        mask = tau_true==-1
        mse[mask] = pad
        # mse = mse*3e8
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
        plt.title("Sensing MSE")
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        plt.imshow(mse)
        # set colorbar size
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.subplot(122)
        plt.title("Sensing CRB")
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        plt.imshow(crb)
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.savefig(f"{title}/out-{pos}.png")


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
        tgv = info.get("tgv")
        tgmat = info.get("tgmat")
        bspos = info.get("bspos")
        ray_type = getRayType()
        env_title = f"./Data/{scene_name}/{tgname}/BS-{len(bspos)}-{bspos[0]}-{bspos[-1]}-{tgv}/"
        title = f"./Data/{scene_name}/{tgname}/BS-{len(bspos)}-{bspos[0]}-{bspos[-1]}-{tgv}/{num_samples}-{ray_type}-{max_depth}-{subcarrier_num}-{x}-{y}-{cell_size}"
        print(f"scene: {scene_name}, tgname: {tgname}, BSpos: {bspos}, v: {tgv}")
        
        # create folder
        if not os.path.exists(f"{title}"):
            os.makedirs(f"{title}")

        # 获取位置信息
        if info.get("pos") is not None:
            cell_pos = info.get("pos")
        else:
            cell_pos = getPos(map_center,x,y,cell_size)
        np.save(f"{title}/cell_pos.npy",cell_pos)
        
        # 计算环境杂波信道
        if os.path.exists(f"{env_title}/h_env.npy"):
            h_env = np.load(f"{env_title}/h_env.npy")
        else:
            print("computing env csi...")
            scene = setScene(scene_path)
            h_env = CSI(scene,info,bspos,[0,0,0])
            np.save(f"{env_title}/h_env.npy",h_env)
            
        h_tgs = []
        tau_trues = []
        tau_ests = []
        tg_crbs = []
        for tgpos in tqdm.tqdm(cell_pos):
            # 设置目标场景信息
            target = Target(tgpath, tgmat, translate=tgpos)
            scene = setScene(scene_path,target,[tgname],[tgv])
            
            # 计算含目标的CSI
            h_list1,tau_true,crb = CSI(scene,info,bspos,tgpos,return_tau=True,tgname=tgname)
            crb = tf.squeeze(crb)
            h_tgs.append(h_list1)
            tau_trues.append(tau_true)
            # 计算crb
            if scene.get("tx") is not None:
                scene.remove("tx")
            if scene.get("rx") is not None:
                scene.remove("rx")
            # tqdm.tqdm.write(f"crb-{tgpos}")
            tg_crbs.append(crb)
            
            # 计算music估计值
            # tqdm.tqdm.write(f"music-{tgpos}")
            for i in range(len(h_list1)):
                tau = tau_true[i]
                h = h_list1[i]-h_env[i]
                h_freq = time_to_ofdm_channel(h,rg,l_min)
                tau = tau*1e9
                # start = tau-step*500
                # if start < 0:
                #     start = 0
                # end = tau+step*500
                start = 0
                end = 200
                try:
                    t = music(h_freq,frequencies,start=start,end=end,step=step)
                except:
                    t = 0
                tau_est = t
                tau_ests.append(tau_est)
        
        h_tgs = np.array(h_tgs)
        tau_trues = np.array(tau_trues)
        tau_ests = np.array(tau_ests)
        tau_ests = np.reshape(tau_ests,(-1,len(h_list1)))
        tg_crbs = np.array(tg_crbs)
        np.save(f"{title}/h_tgs.npy",h_tgs)
        np.savetxt(f"{title}/tau_trues.npy",tau_trues)
        np.savetxt(f"{title}/tau_ests.npy",tau_ests)
        np.savetxt(f"{title}/tg_crbs.npy",tg_crbs) 
        # 保存结果
        saveFig(title,tau_trues,tau_ests,tg_crbs,int(y/cell_size)+1,step,bspos)
                
        print ("done")
        

if __name__ == "__main__":
    main()