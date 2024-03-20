import os
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sionna
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel
from mysionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Scene
from mysionna.rt.scattering_pattern import *
from mysionna.rt.scene import Target,load_sensing_scene

model_save_path = './models/street/' # 模型保存路径
DAS = 100 # Default Area Size,默认目标活动区域范围（100m）
VCRT = 0.05 # Velocity Change Rate,速度变化概率 m/s
VCS = 4 # Velocity Change Size,速度变化大小，即一次最多变化多少 m/s
VCRG = 22.2 # Velocity Change Range,速度变化范围 m/s (0.28m/s~~1km/h)
TIME_SLOT = 0.5 # 时间间隔 s
# 目标移动策略 random,graph
# random: 在区域内随机移动，需更改配置DAS，目标将在以原点为中心，边长为2*DAS的正方形区域内随机移动
# area: 按照指定路线移动,主要用于模拟车辆轨迹/路上行人轨迹。需构建环境时提供如下参数：
#       end_points: [num1,3] float,指定的起点/终点
#       points: [num2,3] float,指定的移动点
#       point_bias: float,移动点偏移范围,目标会把目的点设置为以point为中心，point_bias为半径的圆内的随机点
#       point_path:[num1+num2,num1+num2] int,邻接矩阵，表示点之间的路径关系（有向图表示）,前num1个为end_points,后num2个为points
MOVE_STRATEGY = 'graph' 

class CNN(tf.keras.Model):
    def __init__(self, num_feature):
        super(CNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')
        self.flat = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(units=512, activation='relu')
        self.out = tf.keras.layers.Dense(units=num_feature)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flat(x)
        x = self.dense(x)
        return self.out(x)

class DQNnet(tf.keras.Model):
    def __init__(self, num_action,trainable=True):
        super().__init__('mlp_q_network')
        if trainable:
            self.dense1 = tf.keras.layers.Dense(units=64, activation='relu')
            self.dense2 = tf.keras.layers.Dense(units=32, activation='relu')
            self.out = tf.keras.layers.Dense(units=num_action,activation='linear')
        else:
            self.dense1 = tf.keras.layers.Dense(units=64, activation='relu',trainable=False)
            self.dense2 = tf.keras.layers.Dense(units=32, activation='relu',trainable=False)
            self.out = tf.keras.layers.Dense(units=num_action,activation='linear',trainable=False)
    def call(self, inputs):
        x = tf.convert_to_tensor(inputs)
        x = self.dense1(inputs)
        x = self.dense2(x)
        return self.out(x)

class DQN():
    def __init__(self, num_feature, num_action, learning_rate=0.01, reward_decay=0.9, e_greedy=0.2,replace_target_iter=100, memory_size=1000, batch_size=32,path=None):
        self.num_feature = num_feature
        self.num_action = num_action
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.epsilon_increment = 0.001
        self.epsilon_max = 0.9
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = np.zeros((self.memory_size, num_feature*2+2)) # feature + action + reward + feature_ 
        self.memory_counter = 0
        self.learn_step_counter = 0
        self.cost_his = []
        self.mean_loss = 99999
        
        self.eval_net = DQNnet(self.num_action)
        self.target_net = DQNnet(self.num_action,False)
        self.eval_net.compile(optimizer=tf.keras.optimizers.Adam(self.lr),loss='mse')
        # self.loss_func = tf.losses.mean_squared_error
        # self.optimizer = tf.keras.optimizers.Adam(self.lr)
        if path is not None and isinstance(path,str) and os.path.exists(path):
            self.eval_net = tf.keras.models.load_model(path)
            self.target_net = tf.keras.models.load_model(path.replace('eval','target'))
        
    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1
    
    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            actions_value = self.eval_net(observation)
            action = np.argmin(actions_value)
        else:
            best_action = np.random.randint(0, self.num_action)
            best_mse = 9999999
            for action in range(self.num_action):
                range_true = np.linalg.norm(env.BS_pos[action:] - env.pos_now)
                range_est = env._music_range(env.h_freq-env.h_env,action,env.frequencies,**env.music_params)
                mse = np.abs(range_true-range_est)
                if mse < best_mse:
                    best_mse = mse
                    best_action = action
            action = best_action
        return action

    def _replace_target_params(self):
        self.target_net.set_weights(self.eval_net.get_weights())
    
    def learn(self):
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        
        q_next = self.target_net.predict(batch_memory[:, -self.num_feature:])
        q_eval = self.eval_net.predict(batch_memory[:, :self.num_feature])
        
        q_target = q_eval.copy()
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        action = batch_memory[:, self.num_feature].astype(int)
        reward = batch_memory[:, self.num_feature+1]
        q_target[batch_index, action] = reward + self.gamma * tf.reduce_min(q_next, axis=1)
        
        # train
        self.cost = self.eval_net.train_on_batch(batch_memory[:, :self.num_feature], q_target)
        
        self.cost_his.append(self.cost)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            new_mean_loss = np.mean(self.cost_his)
            print('\ntarget_params_replaced\n')
            # save model
            if self.mean_loss > new_mean_loss:
                self.mean_loss = new_mean_loss
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
            self.save_model(model_save_path,self.mean_loss)
            print(f"model saved, mean loss: {self.mean_loss}")
    
    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()
    
    def save_model(self,loss,path):
        #time: Month-Day-Hour-Minute
        precent = time.strftime('%m-%d-%H-%M',time.localtime(time.time()))
        self.eval_net.save(f"{path}/eval_{loss:.5f}_{precent}.h5")
        self.target_net.save(f"{path}/target_{loss:.5f}_{precent}.h5")
        
class Environment():
    def __init__(self,**kwargs):
        self.scene = None
        self.action_space = kwargs.get('action_space',6)
        # 基站参数---------------------------------------------------------
        self.BS_num = self.action_space
        self.BS_pos = kwargs.get('BS_pos',np.array([[32.8,0,50.3],[-30.3,93,20.8],[-121.4,33.2,8.9],[27.2,-143.9,8.6],[-25.3,-88.4,45.3],[108.6,-121.1,24.9]]))
        self.env_path = kwargs.get('env_path','./scenes/Street/street.xml')
        # 目标移动范围参数---------------------------------------------------------
        if MOVE_STRATEGY == 'random': 
            self.target_move_area = kwargs.get('move_area',np.array([[DAS,DAS],[DAS,-DAS],[-DAS,-DAS],[-DAS,DAS]]))
        elif MOVE_STRATEGY == 'graph':
            self.end_points = kwargs.get('end_points',np.array([[172,-98,0],[0,-180,0],[0,180,0],[-180,0,0]]))
            self.points = kwargs.get('points',np.array([[0,0,0],[0,-98,0]]))
            self.point_bias = kwargs.get('point_bias',15)
            self.point_path = kwargs.get('point_path',np.array([[0,0,0,0,0,1],[0,0,0,0,0,1],[0,0,0,0,1,0],[0,0,0,0,1,0],[0,0,1,1,0,1],[1,1,0,0,1,0]]))
            self.num_points = len(self.points)
            self.num_end_points = len(self.end_points)
            num_path = len(self.point_path)
            if num_path != self.num_points + self.num_end_points:
                raise ValueError('point_path must be a (num_points+num_end_points) x (num_points+num_end_points) matrix')
        # 目标参数---------------------------------------------------------
        self.target_num = kwargs.get('target_num',1)
        self.target_name = kwargs.get('target_name','car')
        self.target_path = kwargs.get('target_path','meshes/car.ply')
        self.target_material = kwargs.get('target_material','itu_metal')
        # 天线配置参数 ---------------------------------------------------------
        self.tx_params = {
            "num_rows": kwargs.get('num_tx_rows',1),
            "num_cols": kwargs.get('num_tx_cols',1),
            "vertical_spacing": kwargs.get('vertical_spacing',0.5),
            "horizontal_spacing": kwargs.get('horizontal_spacing',0.5),
            "pattern": kwargs.get('pattern','dipole'),
            "polarization": kwargs.get('polarization','V'),
            "polarization_model": kwargs.get('polarization',2)
        }
        self.rx_params = {
            "num_rows": kwargs.get('num_rx_rows',1),
            "num_cols": kwargs.get('num_rx_cols',1),
            "vertical_spacing": kwargs.get('vertical_spacing',0.5),
            "horizontal_spacing": kwargs.get('horizontal_spacing',0.5),
            "pattern": kwargs.get('pattern','dipole'),
            "polarization": kwargs.get('polarization','V'),
            "polarization_model": kwargs.get('polarization',2)
        }
        self.frequncy = kwargs.get('frequency',2.14e9)
        self.synthetic_array = kwargs.get('synthetic_array',True)
        self.BS_pos_trainable = kwargs.get('BS_pos_trainable',False)
        # 光线追踪参数 ---------------------------------------------------------
        self.ray_tracing_params = {
            "max_depth": kwargs.get('max_depth',1),
            "method": kwargs.get('method','fibonacci'),
            "num_samples": kwargs.get('num_samples',int(1e5 * self.BS_num)),
            "los": kwargs.get('los',True),
            "reflection": kwargs.get('reflection',True),
            "diffraction": kwargs.get('diffraction',True),
            "scattering": kwargs.get('scattering',True),
            "scat_keep_prob": kwargs.get('scat_keep_prob',0.01),
            "edge_diffraction": kwargs.get('edge_diffraction',False),
            "check_scene": kwargs.get('check_scene',True),
            "scat_random_phases": kwargs.get('scat_random_phases',False)
        }
        # 多普勒参数 ---------------------------------------------------------
        self.doppler_params = {
            "sampling_frequency": kwargs.get('subcarrier_spacing',15e3),
            "num_time_steps": kwargs.get('num_time_steps',14),
            "target_velocities": kwargs.get('target_velocity',None)
        }
        # 频域信道参数 ---------------------------------------------------------
        self.subcarrier_num = kwargs.get('subcarrier_num',32)
        self.frequencies = subcarrier_frequencies(self.subcarrier_num, self.doppler_params["sampling_frequency"])
        # MUSIC估计参数 ---------------------------------------------------------
        self.music_params = {
            "start": kwargs.get('start',0),
            "end": kwargs.get('end',1000),
            "step": kwargs.get('step',0.2)
        } 
        # 初始化环境 ---------------------------------------------------------
        self.scene = self.mk_sionna_env()
        paths = self.scene.compute_paths(**self.ray_tracing_params)
        paths.normalize_delays = False
        paths.apply_doppler(**self.doppler_params)
        a,tau = paths.cir()
        self.h_env = cir_to_ofdm_channel(self.frequencies, a, tau, normalize=False)
        # 特征数量 ---------------------------------------------------------
        # num_BS * tx_num * rx_num * num_subcarrier * num_time_steps * (real+img) + pos_now + velocity_now
        self.n_features = self.h_env.shape[1] * self.h_env.shape[6] * 2 + 6
        self.h_env,_ = self._normalize_h(self.h_env)
        
    def mk_sionna_env(self,tg=None,tgname=None,tgv=None):
        if tg is None:
            scene = load_scene(self.env_path)
        else:
            scene = load_sensing_scene(self.env_path,tg)
        #配置天线阵列------------------------------------------------
        scene.tx_array = PlanarArray(**self.tx_params)
        scene.rx_array = PlanarArray(**self.rx_params)
        scene.frequency = self.frequncy # in Hz; implicitly updates RadioMaterials
        scene.synthetic_array = self.synthetic_array # If set to False, ray tracing will be done per antenna element (slower for large arrays)
        if self.BS_pos_trainable:
            self.BS_pos = [tf.Variable(pos) for pos in self.BS_pos]
        for idx in range(self.BS_num):
            pos = self.BS_pos[idx]
            tx = Transmitter(name=f'tx{idx}',position=pos)
            rx = Receiver(name=f'rx{idx}',position=pos)
            if scene.get(f'tx{idx}') is not None:
                scene.remove(f'tx{idx}')
            scene.add(tx)
            if scene.get(f'rx{idx}') is not None:
                scene.remove(f'rx{idx}')
            scene.add(rx)
        #配置场景材质属性--------------------------------------------
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
    
    def get_observation(self):
        paths = self.scene.compute_paths(**self.ray_tracing_params)
        paths.normalize_delays = False
        self.doppler_params["target_velocities"],obj_name = self.scene.compute_target_velocities(paths,True)
        paths.apply_doppler(**self.doppler_params)
        a,tau = paths.cir()
        self.h_freq = cir_to_ofdm_channel(self.frequencies, a, tau, normalize=False)
        self.h_freq,observation = self._normalize_h(self.h_freq)
        del paths
        return tf.concat([observation,tf.constant(self.pos_now,dtype=tf.float32),tf.constant(self.velocity_now,dtype=tf.float32)],axis=0)
    
    def reset(self):
        self.next_end = False # 用于标记一轮模拟结束
        self.pos_list = []
        self.step_count = 0
        # 生成目标移动路径
        if MOVE_STRATEGY == 'random':
            # 只在平面移动
            while np.random.rand() < 0.5 or len(self.pos_list) < 2: # 以0.7的概率继续移动,或者至少移动1次
                pos_now = np.random.rand(3)*DAS
                pos_now[2] = 0
                self.pos_list.append(pos_now)
        elif MOVE_STRATEGY == 'graph':
            # now pos
            pos_now_idx = np.random.randint(0,self.num_end_points)
            pos_now = self.end_points[pos_now_idx]
            # bias
            x_bias = (np.random.rand()*2-1)*self.point_bias
            pos_now[0] = pos_now[0] + x_bias
            y_bias = (np.random.rand()*2-1)*self.point_bias
            pos_now[1] = pos_now[1] + y_bias
            # next pos
            self.pos_list.append(pos_now)
            while True:
                has_path = np.where(self.point_path[pos_now_idx])[0]
                pos_next_idx = has_path[np.random.randint(0,len(has_path))]
                if pos_next_idx < self.num_end_points:
                    # end point
                    pos_next = self.end_points[pos_next_idx,:]
                    self.next_end = True
                else:
                    # point
                    pos_next = self.points[pos_next_idx-self.num_end_points,:]
                x_bias = (np.random.rand()*2-1)*self.point_bias
                pos_next[0] = pos_next[0] + x_bias
                y_bias = (np.random.rand()*2-1)*self.point_bias
                pos_next[1] = pos_next[1] + y_bias
                self.pos_list.append(pos_next)
                pos_now_idx = pos_next_idx
                if self.next_end:
                    break
        # 生成初始位置和速度、状态
        self.path_len = len(self.pos_list)
        self.next_pos_idx = 1
        self.pos_now = self.pos_list[self.next_pos_idx-1]
        pos_dis = self.pos_list[self.next_pos_idx]-self.pos_list[self.next_pos_idx-1]
        self.velocity_now = (pos_dis)/(np.linalg.norm((pos_dis))) * np.random.rand() * VCRG # 单位向量*速度*随机范围
        # 设置场景，获取CSI
        target = Target(self.target_path, self.target_material, translate=self.pos_now)
        self.scene = self.mk_sionna_env(tg=target,tgname=[self.target_name],tgv=[self.velocity_now])
        observation = self.get_observation()
        self.reward = 0
        self.done = False
        return observation
    
    def step(self, action):
        # reward------------------------------------------------------------------------------------
        range_true = np.linalg.norm(self.BS_pos[action:] - self.pos_now)
        range_est = self._music_range(self.h_freq-self.h_env,action,self.frequencies,**self.music_params) 
        self.reward = self._get_reward(range_true,range_est)
        self.step_count = self.step_count + 1
        # 目标移动-----------------------------------------------------------------------------------
        move_length = np.linalg.norm(self.velocity_now * TIME_SLOT)
        rest_length = np.linalg.norm(self.pos_list[self.next_pos_idx]-self.pos_now)
        if move_length >= rest_length:
            self.pos_now = self.pos_list[self.next_pos_idx]
            self.next_pos_idx += 1
            if self.next_pos_idx == self.path_len: # 当前要到达的点是最后一个点
                self.done = True
            else:
                pos_dis = self.pos_list[self.next_pos_idx]-self.pos_list[self.next_pos_idx-1]
                self.velocity_now = pos_dis/(np.linalg.norm(pos_dis)) * np.linalg.norm(self.velocity_now)# 变更速度方向
        else:
            self.pos_now = self.pos_now + self.velocity_now * TIME_SLOT
        # 速度随机变化-----------------------------------------------------------------------------------
        if np.random.rand() < VCRT:
            self.velocity_now = self.velocity_now * (((np.random.rand()*2-1)*VCS + np.linalg.norm(self.velocity_now))/np.linalg.norm(self.velocity_now))
        # 下一次state-----------------------------------------------------------------------------------
        tg = Target(self.target_path, self.target_material, translate=self.pos_now)
        self.scene = self.mk_sionna_env(tg=tg,tgname=[self.target_name],tgv=[self.velocity_now])
        self.state_ = self.get_observation()            
        return self.state_, self.reward, self.done
    
    def _normalize_h(self,h):
        # h:[batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size]
        # 去对角线，然后展开并分为实部和虚部
        h = tf.transpose(h,perm=[0,2,4,5,6,1,3])
        h = tf.linalg.diag_part(h)
        # h:[batch size, num_tx, num_tx_ant, num_rx_ant, num_time_steps, fft_size]
        h = tf.transpose(h,perm=[0,5,1,2,3,4])
        # h:[batch size, num_tx, fft_size]
        h = h[0,:,0,0,0,:]
        h_flatten = tf.reshape(h,[-1])
        h_real = tf.math.real(h_flatten)
        h_img = tf.math.imag(h_flatten)
        h_real = h_real * 1e5
        h_img = h_img * 1e5
        h_flatten = tf.concat([h_real,h_img],axis=0)
        return h,h_flatten

    def _music_range(self,h_freq,BS_id,frequencies,start = 0,end = 1000,step = 0.2):
        try:
            y_i = h_freq[BS_id,:]
            y_i = tf.squeeze(y_i)
            y_i = tf.expand_dims(y_i, axis=0)
            y_i_H = tf.transpose(tf.math.conj(y_i))
            y_conv = tf.matmul(y_i_H, y_i)
            _, eig_vecs = tf.linalg.eigh(y_conv)
            tau_range = np.arange(start,end, step)
            G_n = tf.cast(eig_vecs[:,:-4], dtype=tf.complex64)
            G_n_H = tf.math.conj(tf.transpose(G_n))
            frequencies_c = tf.expand_dims(frequencies, axis=0)
            frequencies_c = tf.repeat(frequencies_c, len(tau_range), axis=0)
            frequencies_c = tf.cast(frequencies_c, dtype=tf.complex64)
            tau_range = tf.expand_dims(tau_range, axis=-1)
            tau_range = tf.repeat(tau_range, self.subcarrier_num, axis=-1)
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
            return tau_est*0.15
        except:
            return 0

    def _get_reward(self,true_value,est_value):
        # 如果估计值和真实值的相差在真实值的10%以内，那么依据误差大小奖励在0.5~1之间
        # 如果估计值和真实值的相差在真实值的10%~20%，那么依据误差大小奖励在0.1~0.5之间
        # 否则，惩罚值在-1~0之间
        diff = np.abs(true_value-est_value)
        if diff <= true_value*0.1:
            return 0.5 + 0.5*(1-diff/(true_value*0.01))
        elif diff <= true_value*0.2:
            return 0.5*(1-diff/(true_value*0.05))
        else:
            return -(diff/true_value)
    
def run():
    step = 0
    for episode in range(3000):
        print(f"====={episode}th episode start=====")
        observation = env.reset()
        inner_step = 0
        while True:
            action = RL.choose_action(observation)
            print(f"\r【{step}-{inner_step}th step】pos:{env.pos_now},velocity:{env.velocity_now},BS:{action},reward:{env.reward}")
            observation_, reward, done = env.step(action)
            RL.store_transition(observation, action, reward, observation_)
            if (step >= 32) and (step % 5 == 0):
                RL.learn()
            observation = observation_
            if done:
                # print(f"====={episode}th episode done=====")
                break
            step += 1
            inner_step += 1

if __name__ == "__main__":
    np.set_printoptions(precision=1)
    env = Environment()
    model_save_path = f'./models/street/{env.n_features}-{env.action_space}/'
    RL = DQN(env.n_features,env.action_space)
    
    run()




