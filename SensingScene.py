import sionna
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import mysionna
from mysionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

class SensingScene():
    def __init__(self,scene,tx_positions=np.array([[0,0,30]])) -> None:
        self.crb_delay = None
        self.crb_angle = None
        self.crb_speed = None
        self._cell_positions = None # 3D array of cell positions
        self._cell_map = None # 2D array of cell map
        self._cell_num = 1
        self._paths = None
        self._tx_positions = tx_positions
        print("Initializing scene...")
        self._scene = load_scene(scene)
        self._scene.frequency = 2.14e9 # in Hz; implicitly updates RadioMaterials
        self._scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)
        self._scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="dipole",
                             polarization="V")
        # Configure antenna array for all receivers
        self._scene.rx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="dipole",
                                polarization="V")
        print("Scene initialized.")
    
    def get_cell_positions(self,map_position,map_size, cell_size):
        cell_num = int(map_size/cell_size) # Number of cells in the map
        self.cell_num = cell_num
        self._cell_positions = np.zeros((cell_num, cell_num, 3))
        self._cell_map = np.zeros((cell_num, cell_num))
        self._cell_positions[0,0,:] = map_position # Set the first cell position
        for i in range(0, cell_num):
            # Set the first cell position of each row
            if i > 0:
                self._cell_positions[i,0,:] = self._cell_positions[i-1,0,:] + [cell_size,0,0]
            for j in range(1, cell_num):
                self._cell_positions[i,j,:] = self._cell_positions[i,j-1,:] + [0,cell_size,0]
        return self._cell_positions
    
    @property
    def cell_num(self):
        return self._cell_num
    
    @cell_num.setter
    def cell_num(self,cell_num):
        self._cell_num = cell_num

    def set_tx_position(self,tx_positions):
        """_summary_
        Args:
            tx_positions (nparray[num_tx,3]): tx antenna positions
        """
        self._tx_positions = np.array(tx_positions)
    
    def del_receiver(self):
        cell_num = self.cell_num
        if cell_num is None:
            return
        for i in range(0, cell_num):
            for j in range(0, cell_num):
                rx_name = f"rx_{i}_{j}"
                if self._scene.get(rx_name) is not None:
                    self._scene.remove(rx_name)
    
    def compute_paths(self,max_depth=3,num_samples=1000000,max_rx_each_turn = 100,los=True,reflection=True,diffraction=False,scattering=False):
        self._paths = None
        scene = self._scene
        cell_num = self.cell_num # Number of cells in the map
        cell_positions = self._cell_positions # 3D array of cell positions
        num_tx = self._tx_positions.shape[0] # Number of transmitters
        for tx_idx in range(0,num_tx):
            # remove previous transmitter
            if scene.get(f"tx_{tx_idx-1}") is not None:
                scene.remove(f"tx_{tx_idx-1}")
            # Add new transmitter
            tx = Transmitter(name=f"tx_{tx_idx}",position=self._tx_positions[tx_idx,:])
            if scene.get(f"tx_{tx_idx}") is not None:
                scene.remove(f"tx_{tx_idx}")
            scene.add(tx)
            num = 0
            i = 0
            rx_name_list = []
            # Compute paths for receivers
            while i < cell_num:
                j = 0
                while j < cell_num:
                    # Add receiver
                    rx_name = f"rx_{i}_{j}"
                    rx_name_list.append(rx_name)
                    rx = Receiver(name=rx_name,position=cell_positions[i,j,:])
                    if scene.get(rx_name) is not None:
                        scene.remove(rx_name)
                    scene.add(rx)
                    num += 1
                    if num == max_rx_each_turn: # Compute paths
                        num = 0
                        paths:mysionna.rt.Paths = scene.compute_paths(max_depth=max_depth,num_samples=num_samples,los=los,reflection=reflection,diffraction=diffraction,scattering=scattering)
                        if self._paths is None:
                            self._paths = paths
                        else:
                            self._paths = self._paths.merge_different_rx(paths)
                        for rx_name in rx_name_list:
                            scene.remove(rx_name)
                        rx_name_list = []
                    j += 1
                i += 1
            # Compute paths for the rest receivers
            if num > 0:
                paths:mysionna.rt.Paths = scene.compute_paths(max_depth=max_depth,num_samples=num_samples,los=los,reflection=reflection,diffraction=diffraction,scattering=scattering)
                if self._paths is None:
                    self._paths = paths
                else:
                    self._paths = self._paths.merge_different_rx(paths)
                for rx_name in rx_name_list:
                    scene.remove(rx_name)
                rx_name_list = []
        
        self._paths.finalize_different_rx()
        
        return self._paths
    
    def sensing_crb(self,snr,delay=True,angle=False,speed=False):
        """_summary_
        Args:
            snr (float): snr in dB
        """
        if self._paths is None:
            raise Exception("Please compute paths first.")
        paths = self._paths
        # Compute CRB
        a = paths.a
        tau = paths.tau
        num_rx = a.shape[1]
        num_rx_ant = a.shape[2]
        num_tx = a.shape[3]
        num_tx_ant = a.shape[4]
        max_num_paths = tau.shape[-1]
        cell_num = self.cell_num
        result = None
        if delay:
            crb_delay = np.zeros((num_tx,cell_num,cell_num))
            for tx in range(0,num_tx):
                for i in range(0,cell_num):
                    for j in range(0,cell_num):
                        a_ij = a[0,i*cell_num+j,0,tx,0,:,0]
                        tau_ij = tau[0,i*cell_num+j,tx,:]
                        crb_delay[tx,i,j] = self._crb_delay(snr,a_ij,tau_ij)
            self.crb_delay = crb_delay
            
        return self.crb_delay,self.crb_angle,self.crb_speed
            
    def _crb_delay(self,snr,a,tau):
        frequency = self._scene.frequency
        tau = tau[a!=0]
        a = a[a!=0]
        a = a[tau>0]
        tau = tau[tau>0]
        # a_H = np.conj(a)
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
        if tf.rank(F) != length:
            return 0
        crb_F = tf.linalg.inv(F)
        crb = tf.linalg.diag_part(crb_F)
        a_sortidx = np.argsort(np.abs(a))
        return crb[a_sortidx[-1]]
        
    def render(self,heat=False,delay=True,angle=False,speed=False):
        cell_num = self.cell_num # Number of cells in the map
        cell_positions = self._cell_positions
        scene = self._scene
        colors = np.zeros((cell_num,cell_num,3))
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
        
        for i in range(0, cell_num):
            for j in range(0, cell_num):
                rx_name = f"rx_{i}_{j}"
                if scene.get(rx_name) is not None:
                    scene.remove(rx_name)
                rx = Receiver(name=rx_name,position=cell_positions[i,j,:])
                scene.add(rx)
    
    def heat_map(self,tx=0,delay=True,angle=False,speed=False):
        cell_num = self.cell_num  
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
                raise Exception("Please compute CRB first.")
            crb = self.crb_delay[tx,:,:]
            plt.subplot(num_pic,1,pic_id)
            pic_id = pic_id + 1
            plt.imshow(crb)
            plt.colorbar()
            plt.title("Delay")
            plt.xlabel("x")
            plt.ylabel("y")
        
        return fig
    