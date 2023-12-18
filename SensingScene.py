import sionna
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import mysionna
from mysionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

class SensingScene():
    def __init__(self,scene,tx_array=None,rx_array=None,tx_positions=np.array([[0,0,30]]),tx_look_dir=np.array([[0,0,0]])) -> None:
        self.crb_delay = None
        self.crb_angle = None
        self.crb_speed = None
        self._cell_positions = None # 3D array of cell positions
        self._cell_map = None # 2D array of cell map
        self._cell_num_x = 1
        self._cell_num_y = 1
        self._paths = None
        self._tx_positions = tx_positions
        self._tx_look_dir = tx_look_dir
        print("Initializing scene...")
        self._scene = load_scene(scene)
        self._scene.frequency = 2.14e9 # in Hz; implicitly updates RadioMaterials
        self._scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)
        if tx_array is not None:
            self._scene.tx_array = tx_array
        else:
            self._scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="tr38901",
                             polarization="V")
        # Configure antenna array for all receivers
        if rx_array is not None:
            self._scene.rx_array = rx_array
        else:
            self._scene.rx_array = PlanarArray(num_rows=1,
                                    num_cols=1,
                                    vertical_spacing=0.5,
                                    horizontal_spacing=0.5,
                                    pattern="dipole",
                                    polarization="V")
        print("Scene initialized.")
    
    def get_cell_positions(self,map_center,map_size_x,map_size_y, cell_size):
        cell_num_x = int(map_size_x/cell_size) + 1 # Number of cells in the map
        cell_num_y = int(map_size_y/cell_size) + 1 # Number of cells in the map
        self._cell_num_x = cell_num_x
        self._cell_num_y = cell_num_y
        self._map_center = map_center
        self._map_size_x = map_size_x
        self._map_size_y = map_size_y
        self._cell_size = cell_size
        self._cell_positions = np.zeros((cell_num_x, cell_num_y, 3))
        x_fill = np.arange(0,cell_num_x) * cell_size + map_center[0] - map_size_x/2
        x_fill = np.tile(x_fill,cell_num_y)
        self._cell_positions[:,:,0] = x_fill.reshape([cell_num_y,cell_num_x]).transpose()
        y_fill = np.arange(0,cell_num_y) * cell_size + map_center[1] - map_size_y/2
        y_fill = np.tile(y_fill,cell_num_x)
        self._cell_positions[:,:,1] = y_fill.reshape([cell_num_x,cell_num_y])
        self._cell_positions[:,:,2] = np.tile(map_center[2],(cell_num_x,cell_num_y))
        return self._cell_positions

    def set_tx(self,tx_positions,tx_look_dir):
        """_summary_
        Args:
            tx_positions (nparray[num_tx,3]): tx antenna positions
        """
        if tx_positions.shape[0] == 0 or tx_look_dir.shape[0] == 0:
            raise Exception("The number of tx antenna positions and tx antenna look directions should be greater than 0.")
        if tx_positions.shape[0] != tx_look_dir.shape[0]:
            raise Exception("The number of tx antenna positions and tx antenna look directions should be the same.")
        if tx_positions.shape[1] != 3 or tx_look_dir.shape[1] != 3:
            raise Exception("The shape of tx antenna positions and look directions should be (num_tx,3).")
        self._tx_positions = np.array(tx_positions)
        self._tx_look_dir = np.array(tx_look_dir)
    
    def del_receiver(self):
        cell_num_x = self._cell_num_x
        cell_num_y = self._cell_num_y
        if cell_num_x is None or cell_num_y is None:
            return
        for i in range(0, cell_num_x):
            for j in range(0, cell_num_y):
                rx_name = f"rx_{i}_{j}"
                if self._scene.get(rx_name) is not None:
                    self._scene.remove(rx_name)
    
    def compute_paths(self,max_depth=3,num_samples=1000000,max_rx_each_turn = 100,los=True,reflection=True,diffraction=False,scattering=False):
        self._paths = None
        scene = self._scene
        cell_num_x = self._cell_num_x # Number of cells in the map
        cell_num_y = self._cell_num_y # Number of cells in the map
        cell_positions = self._cell_positions # 3D array of cell positions
        num_tx = self._tx_positions.shape[0] # Number of transmitters
        self.crb_delay = np.zeros((num_tx,cell_num_x,cell_num_y))
        for tx_idx in range(0,num_tx):
            # remove previous transmitter
            if scene.get(f"tx_{tx_idx-1}") is not None:
                scene.remove(f"tx_{tx_idx-1}")
            # Add new transmitter
            tx = Transmitter(name=f"tx_{tx_idx}",position=self._tx_positions[tx_idx,:])
            if scene.get(f"tx_{tx_idx}") is not None:
                scene.remove(f"tx_{tx_idx}")
            scene.add(tx)
            # Set transmitter look direction
            rx_tmp = Receiver(name="rx_tmp",position=self._tx_look_dir[tx_idx,:])
            tx.look_at(rx_tmp)
            scene.remove("rx_tmp")
            num = 0
            i = 0
            rx_name_list = []
            rx_index_list = []
            # Compute paths for receivers
            while i < cell_num_x:
                j = 0
                while j < cell_num_y:
                    # Add receiver
                    rx_name = f"rx_{i}_{j}"
                    rx_name_list.append(rx_name)
                    rx_index_list.append([i,j])
                    rx = Receiver(name=rx_name,position=cell_positions[i,j,:])
                    if scene.get(rx_name) is not None:
                        scene.remove(rx_name)
                    scene.add(rx)
                    num += 1
                    if num % max_rx_each_turn == 0: # Compute paths
                        # print(num,num // max_rx_each_turn)
                        paths:mysionna.rt.Paths = scene.compute_paths(max_depth=max_depth,num_samples=num_samples,los=los,reflection=reflection,diffraction=diffraction,scattering=scattering)
                        a,tau = paths.cir()
                        print (num,a.shape)
                        for rx_name in rx_name_list:
                            scene.remove(rx_name)
                        for num_idx in range(0,max_rx_each_turn):
                            i_idx = rx_index_list[num_idx][0]
                            j_idx = rx_index_list[num_idx][1]
                            a_,tau_ = a[0,num_idx,0,0,0,:,0],tau[0,num_idx,0,:]
                            if crb_method == 0:
                                self.crb_delay[tx_idx,i_idx,j_idx] = self._crb_delay0(20,a_,tau_)
                            elif crb_method == 1:
                                self.crb_delay[tx_idx,i_idx,j_idx] = self._crb_delay1(20,a_,tau_)
                        rx_name_list = []
                        rx_index_list = []
                    j += 1
                i += 1
            # Compute paths for the rest receivers
            if num % max_rx_each_turn != 0:
                paths:mysionna.rt.Paths = scene.compute_paths(max_depth=max_depth,num_samples=num_samples,los=los,reflection=reflection,diffraction=diffraction,scattering=scattering)
                paths.normalize_delays = False
                a,tau = paths.cir()
                print (num,a.shape)
                for rx_name in rx_name_list:
                    scene.remove(rx_name)
                for num_idx in range(0,num % max_rx_each_turn):
                    i_idx = rx_index_list[num_idx][0]
                    j_idx = rx_index_list[num_idx][1]
                    a_,tau_ = a[0,num_idx,0,0,0,:,0],tau[0,num_idx,0,:]
                    if crb_method == 0:
                        self.crb_delay[tx_idx,i_idx,j_idx] = self._crb_delay0(20,a_,tau_)
                    elif crb_method == 1:
                        self.crb_delay[tx_idx,i_idx,j_idx] = self._crb_delay1(20,a_,tau_)
                rx_name_list = []
                rx_index_list = []
        
        # rotate and flip
        for tx_idx in range(0,num_tx):
            crb_tmp = self.crb_delay[tx_idx,:,:]
            crb_tmp = np.flip(crb_tmp,axis=1)
            self.crb_delay[tx_idx,:,:] = crb_tmp
        return self.crb_delay
    
    def compute_paths_crb(self,crb_method=0,max_depth=3,num_samples=1000000,max_rx_each_turn = 100,los=True,reflection=True,diffraction=False,scattering=False):
        self._paths = None
        scene = self._scene
        cell_num_x = self._cell_num_x # Number of cells in the map
        cell_num_y = self._cell_num_y # Number of cells in the map
        cell_positions = self._cell_positions # 3D array of cell positions
        num_tx = self._tx_positions.shape[0] # Number of transmitters
        self.crb_delay = np.zeros((num_tx,cell_num_x,cell_num_y))
        for tx_idx in range(0,num_tx):
            # remove previous transmitter
            if scene.get(f"tx_{tx_idx-1}") is not None:
                scene.remove(f"tx_{tx_idx-1}")
            # Add new transmitter
            tx = Transmitter(name=f"tx_{tx_idx}",position=self._tx_positions[tx_idx,:])
            if scene.get(f"tx_{tx_idx}") is not None:
                scene.remove(f"tx_{tx_idx}")
            scene.add(tx)
            # Set transmitter look direction
            rx_tmp = Receiver(name="rx_tmp",position=self._tx_look_dir[tx_idx,:])
            tx.look_at(rx_tmp)
            scene.remove("rx_tmp")
            num = 0
            i = 0
            rx_name_list = []
            rx_index_list = []
            # Compute paths for receivers
            while i < cell_num_x:
                j = 0
                while j < cell_num_y:
                    # Add receiver
                    rx_name = f"rx_{i}_{j}"
                    rx_name_list.append(rx_name)
                    rx_index_list.append([i,j])
                    rx = Receiver(name=rx_name,position=cell_positions[i,j,:])
                    if scene.get(rx_name) is not None:
                        scene.remove(rx_name)
                    scene.add(rx)
                    num += 1
                    if num % max_rx_each_turn == 0: # Compute paths
                        # print(num,num // max_rx_each_turn)
                        paths:mysionna.rt.Paths = scene.compute_paths(max_depth=max_depth,num_samples=num_samples,los=los,reflection=reflection,diffraction=diffraction,scattering=scattering)
                        a,tau = paths.cir()
                        print (num,a.shape)
                        for rx_name in rx_name_list:
                            scene.remove(rx_name)
                        for num_idx in range(0,max_rx_each_turn):
                            i_idx = rx_index_list[num_idx][0]
                            j_idx = rx_index_list[num_idx][1]
                            a_,tau_ = a[0,num_idx,0,0,0,:,0],tau[0,num_idx,0,:]
                            if crb_method == 0:
                                self.crb_delay[tx_idx,i_idx,j_idx] = self._crb_delay0(20,a_,tau_)
                            elif crb_method == 1:
                                self.crb_delay[tx_idx,i_idx,j_idx] = self._crb_delay1(20,a_,tau_)
                        rx_name_list = []
                        rx_index_list = []
                    j += 1
                i += 1
            # Compute paths for the rest receivers
            if num % max_rx_each_turn != 0:
                paths:mysionna.rt.Paths = scene.compute_paths(max_depth=max_depth,num_samples=num_samples,los=los,reflection=reflection,diffraction=diffraction,scattering=scattering)
                paths.normalize_delays = False
                a,tau = paths.cir()
                print (num,a.shape)
                for rx_name in rx_name_list:
                    scene.remove(rx_name)
                for num_idx in range(0,num % max_rx_each_turn):
                    i_idx = rx_index_list[num_idx][0]
                    j_idx = rx_index_list[num_idx][1]
                    a_,tau_ = a[0,num_idx,0,0,0,:,0],tau[0,num_idx,0,:]
                    if crb_method == 0:
                        self.crb_delay[tx_idx,i_idx,j_idx] = self._crb_delay0(20,a_,tau_)
                    elif crb_method == 1:
                        self.crb_delay[tx_idx,i_idx,j_idx] = self._crb_delay1(20,a_,tau_)
                rx_name_list = []
                rx_index_list = []
        
        # rotate and flip
        for tx_idx in range(0,num_tx):
            crb_tmp = self.crb_delay[tx_idx,:,:]
            crb_tmp = np.flip(crb_tmp,axis=1)
            self.crb_delay[tx_idx,:,:] = crb_tmp
        return self.crb_delay
        
    def _crb_delay0(self,snr,a,tau):
        frequency = self._scene.frequency
        # tau = tau[a!=0]
        # a = a[a!=0]
        # a = a[tau>0]
        # tau = tau[tau>0]
        # if len(a) == 0:
        #     return 0
        # phase = tf.complex(tf.zeros_like(tau),2*np.pi*frequency*tau)
        # e = tf.exp(phase)
        # a = a * e
        # length = len(a)
        # if length == 0:
        #     return 0
        # tau_i = tf.repeat(tau,length)
        # tau_i = tf.reshape(tau_i, (length,length))
        # tau_j = tf.transpose(tau_i)
        # tau_i_mine_j = tau_i- tau_j
        # tau_i_mul_j = tau_i* tau_j
        # alpha_ij = tf.reshape(a, (length,1)) @ tf.reshape(a, (1,length))
        # one = tf.ones((length,length))
        # F_alpha= 2*snr*tf.math.abs(alpha_ij)/(tau_i_mul_j**2)
        # F_cos = (one+4*(np.pi**2)*(frequency) * tau_i_mul_j)*tf.math.cos(2*np.pi*frequency*tau_i_mine_j)
        # F_sin = 2*np.pi*frequency*tau_i_mine_j*tf.math.sin(2*np.pi*frequency*tau_i_mine_j)
        # F = F_alpha*(F_cos+F_sin)
        # try:
        #     crb_F = tf.linalg.inv(F)
        # except:
        #     return 0
        # crb = tf.linalg.diag_part(crb_F)
        a = a[a!=0]
        l = len(a)
        if len(a) == 0:
            return 0
        a_H = np.conj(a)
        B = tf.reshape(a_H, [l,1]) @ tf.reshape(a, [1,l])
        try:
            B_inv = tf.linalg.inv(B)
        except:
            return 0
        # B_inv = tf.linalg.inv(B)
        B_diag = tf.linalg.diag_part(B_inv)
        crb = tf.abs(tf.cast(B_diag,tf.complex128)) / tf.cast((8 * np.pi**2 * snr * frequency **2),tf.float64)
        a_sortidx = np.argsort(np.abs(a))
        return crb[a_sortidx[-1]]
    
    def _crb_delay1(self,snr,a,tau):
        frequency = self._scene.frequency
        tau = tau[a!=0]
        a = a[a!=0]
        a = a[tau>0]
        tau = tau[tau>0]
        if len(a) == 0:
            return 0
        phase = tf.complex(tf.zeros_like(tau),2*np.pi*frequency*tau)
        e = tf.exp(phase)
        a = a * e
        length = len(a)
        if length == 0:
            return 0
        tau_i = tf.repeat(tau,length)
        tau_i = tf.reshape(tau_i, (length,length))
        tau_j = tf.transpose(tau_i)
        tau_i_mine_j = tau_i- tau_j
        tau_i_mul_j = tau_i* tau_j
        alpha_ij = tf.reshape(a, (length,1)) @ tf.reshape(a, (1,length))
        one = tf.ones((length,length))
        F_alpha= 2*snr*tf.math.abs(alpha_ij)/(tau_i_mul_j**2)
        F_cos = (one+4*(np.pi**2)*(frequency) * tau_i_mul_j)*tf.math.cos(2*np.pi*frequency*tau_i_mine_j)
        F_sin = 2*np.pi*frequency*tau_i_mine_j*tf.math.sin(2*np.pi*frequency*tau_i_mine_j)
        F = F_alpha*(F_cos+F_sin)
        try:
            crb_F = tf.linalg.inv(F)
        except:
            return 0
        crb = tf.linalg.diag_part(crb_F)
        a_sortidx = np.argsort(np.abs(a))
        return crb[a_sortidx[-1]]
        
    def render(self,heat=False,delay=True,angle=False,speed=False):
        cell_num_x = self._cell_num_x # Number of cells in the map
        cell_num_y = self._cell_num_y # Number of cells in the map
        cell_positions = self._cell_positions
        scene = self._scene
        colors = np.zeros((cell_num_x,cell_num_y,3))
        if heat:
            if delay:
                if self.crb_delay is None:
                    raise Exception("Please compute CRB first.")   
            elif angle:
                if self.crb_angle is None:
                    raise Exception("Please compute CRB first.")
            elif speed:
                if self.crb_speed is None:
                    raise Exception("Please compute CRB first.")
        
        for i in range(0, cell_num_x):
            for j in range(0, cell_num_y):
                rx_name = f"rx_{i}_{j}"
                if scene.get(rx_name) is not None:
                    scene.remove(rx_name)
                rx = Receiver(name=rx_name,position=cell_positions[i,j,:])
                scene.add(rx)
        for i in range(0,self._tx_positions.shape[0]):
            tx_name = f"tx_{i}"
            if scene.get(tx_name) is not None:
                scene.remove(tx_name)
            tx = Transmitter(name=tx_name,position=self._tx_positions[i,:])
            scene.add(tx)
    
    def heat_map(self,tx=0,delay=True,angle=False,speed=False):
        cell_num_x = self._cell_num_x  
        cell_num_y = self._cell_num_y
        num_pic = 0
        pic_id = 1
        fig = plt.figure()
        if delay:
            num_pic = num_pic + 1
        if angle:
            num_pic = num_pic + 1
        if speed:
            num_pic = num_pic + 1
        if delay:
            if self.crb_delay is None:
                raise Exception("Please compute CRB_delay first.")
            crb = self.crb_delay[tx,:,:]
            plt.subplot(num_pic,1,pic_id)
            pic_id = pic_id + 1
            plt.imshow(np.log10(crb),origin='lower')
            plt.colorbar()
            tx_position = (self._tx_positions[tx,:] - self._map_center + np.array([self._map_size_y/2,self._map_size_x/2,0])) / self._cell_size
            plt.scatter(tx_position[0],tx_position[1],marker='x',color='r')
        
        return fig
    