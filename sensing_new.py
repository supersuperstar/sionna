import sionna
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
import pickle
import time

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
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

class SensingScene():
    def __init__(self) -> None:
        self._scene = load_scene(sionna.rt.scene.munich) # Try also sionna.rt.scene.etoile
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
        self._cm = self._scene.coverage_map()

def main(**kwargs):
    # Set up the simulation parameters
    # ------------
    scene = SensingScene()
    
    

if __name__ == "__main__":
    print("Starting simulation...")
    main()
    print("Simulation finished.")