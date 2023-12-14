import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import sionna
from sionna.mimo import StreamManagement
from sionna.ofdm import ResourceGrid, ResourceGridMapper, LSChannelEstimator, LMMSEEqualizer
from sionna.ofdm import OFDMModulator, OFDMDemodulator, ZFPrecoder, RemoveNulledSubcarriers
from sionna.channel import subcarrier_frequencies, cir_to_ofdm_channel, cir_to_time_channel, time_lag_discrete_time_channel
from sionna.channel import ApplyOFDMChannel, ApplyTimeChannel, OFDMChannel, TimeChannel
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.mapping import Mapper, Demapper
from sionna.utils import BinarySource, ebnodb2no, sim_ber
from sionna.utils.metrics import compute_ber
# Import Sionna RT components
import mysionna
from mysionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

class SensingScene():
    def __init__(self) -> None:
        print("Initializing scene...")
        # self._scene = load_scene("/root/autodl-tmp/sionna_sensing/Single Box/Single Box.xml")# Try also sionna.rt.scene.etoile
        self._scene = load_scene(sionna.rt.scene.munich)
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
        tx = Transmitter(name="tx",
                 position=[8.5,21,27])
        self._scene.add(tx)
        self._cell_positions = None # 3D array of cell positions
        self._cell_map = None # 2D array of cell map
        self._cell_num = 1
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
    
    @property.setter
    def cell_num(self,cell_num):
        self.del_receiver()
        self._cell_num = cell_num

    def del_receiver(self):
        cell_num = self.cell_num
        if cell_num is None:
            return
        for i in range(0, cell_num):
            for j in range(0, cell_num):
                rx_name = f"rx_{i}_{j}"
                if self._scene.get(rx_name) is not None:
                    self._scene.remove(rx_name)
    
    def compute_paths(self,max_depth=3,num_samples=10000000,los=True,reflection=True,diffraction=False,scattering=False):
        scene = self._scene
        tx = scene.get("tx")
        cell_num = self.cell_num
        cell_positions = self._cell_positions
        for i in range(0, cell_num):
            for j in range(0, cell_num):
                rx_name = f"rx_{i}_{j}"
                rx = Receiver(name=rx_name,position=cell_positions[i,j,:])
                if scene.get(rx_name) is not None:
                    scene.remove(rx_name)
                scene.add(rx)
        self._paths:sionna.rt.Paths = scene.compute_paths(max_depth=max_depth,num_samples=num_samples,los=los,reflection=reflection,diffraction=diffraction,scattering=scattering)
        return self._paths
        

def main(**kwargs):
    # Set up the simulation parameters
    # [x,y,z],z is the hight and all cells has the same hight
    map_position = kwargs.get("map_position", [45,90,1.5]) 
    # meter
    map_size = kwargs.get("map_size", 10)
    # meter
    cell_size = kwargs.get("cell_size", 0.5)
    # ------------
    scene = SensingScene()
    scene.get_cell_positions(map_position, map_size, cell_size)
    scene.compute_paths()
    
if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            print(e)
    tf.random.set_seed(1) # Set global random seed for reproducibility
    print("Starting simulation...")
    main()
    print("Simulation finished.")