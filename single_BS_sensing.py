
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
import time

# Import Sionna RT components
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, Camera

# For link-level simulations
from sionna.channel import cir_to_ofdm_channel, subcarrier_frequencies, OFDMChannel, ApplyOFDMChannel, CIRDataset
from sionna.nr import PUSCHConfig, PUSCHTransmitter, PUSCHReceiver
from sionna.utils import compute_ber, ebnodb2no, PlotBER
from sionna.ofdm import KBestDetector, LinearDetector
from sionna.mimo import StreamManagement

# Load integrated scene
scene = load_scene(sionna.rt.scene.munich) # Try also sionna.rt.scene.etoile


# Configure antenna array for all transmitters
scene.tx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="dipole",
                             polarization="V")

# Configure antenna array for all receivers
scene.rx_array = PlanarArray(num_rows=1,
                             num_cols=1,
                             vertical_spacing=0.5,
                             horizontal_spacing=0.5,
                             pattern="dipole",
                             polarization="V")

# Add transmitter instance to scene
tx_object = scene.get("tx")
if tx_object is not None:
    scene.remove("tx")
# Create transmitter
tx = Transmitter(name="tx",
                 position=[8.5,21,27])
scene.add(tx)

rx1_object = scene.get("rx1")
if rx1_object is not None:
    scene.remove("rx1")
# Create a receiver
rx_tmp = Receiver(name="rx1",
              position=[45,90,1.5],
              orientation=[0,0,0])

# Add receiver instance to scene
scene.add(rx_tmp)

tx.look_at(rx_tmp) # Transmitter points towards receiver
rx_object = scene.get("rx")
if rx_object is not None:
    scene.remove("rx")
rx = Receiver(name="rx",
              position=[8.5,21,27],
              orientation=[0,0,0])
scene.add(rx)
rx.look_at(rx_tmp)
scene.remove("rx1")


scene.frequency = 2.14e9 # in Hz; implicitly updates RadioMaterials
scene.synthetic_array = True # If set to False, ray tracing will be done per antenna element (slower for large arrays)

paths = scene.compute_paths(max_depth=1,num_samples=1e7,los=False,reflection=True,scattering=True,diffraction=True)
paths.normalize_delays = False

subcarrier_spacing = 15e3
fft_size = 48

a, tau = paths.cir()
print("Shape of tau: ", tau.shape)


# %%
# CRB 1
snr = 1
H=a[0,0,0,0,0,:,0]
H=H[H!=0]
H_H = np.conj(H)
l = len(H)
B = tf.reshape(H_H, [l,1]) @ tf.reshape(H, [1,l])
B_inv = tf.linalg.inv(B)
B_diag = tf.linalg.diag_part(B_inv)
crb1 = tf.abs(tf.cast(B_diag,tf.complex128)) / tf.cast((8 * np.pi**2 * snr * scene.frequency**2),tf.float64)

# %%
#CRB 2
snr = 1
alpha=paths.a[0,0,0,0,0,:,0]
tau=paths.tau[0,0,0,:]
tau=tau[alpha!=0]
alpha=alpha[alpha!=0]
length = len(alpha)
tau_i = tf.repeat(tau,length)
tau_i = tf.reshape(tau_i, (length,length))
tau_j = tf.transpose(tau_i)
tau_i_mine_j = tau_i- tau_j
tau_i_mul_j = tau_i* tau_j
alpha_ij = tf.reshape(alpha, (length,1)) @ tf.reshape(alpha, (1,length))
one = tf.ones((length,length))
F_alpha= 2*snr*tf.math.abs(alpha_ij)/(tau_i_mul_j**2)
F_cos = (one+4*(np.pi**2)*(scene.frequency**2) * tau_i_mul_j)*tf.math.cos(2*np.pi*scene.frequency*tau_i_mine_j)
F_sin = 2*np.pi*scene.frequency*tau_i_mine_j*tf.math.sin(2*np.pi*scene.frequency*tau_i_mine_j)
F = F_alpha*(F_cos+F_sin)
crb_F = tf.linalg.inv(F)
crb2 = tf.linalg.diag_part(crb_F)

# %%
#CRB 3
snr = 1
H=a[0,0,0,0,0,:,0]
tau=paths.tau[0,0,0,:]
tau = tau[H!=0]
H = H[H!=0]
H_H = np.conj(H)
l = len(H)
length = len(H)
tau_i = tf.repeat(tau,length)
tau_i = tf.reshape(tau_i, (length,length))
tau_j = tf.transpose(tau_i)
tau_i_mine_j = tau_i- tau_j
tau_i_mul_j = tau_i* tau_j
B_1 = tf.reshape(H_H, [l,1]) @ tf.reshape(H, [1,l])
one = tf.ones((length,length))
real = one + 4*(np.pi**2)*(scene.frequency**2) * tau_i_mul_j
img = 2*np.pi*scene.frequency*tau_i_mine_j
B_2 = tf.complex(real, img)
B_total = tf.abs(B_1*B_2)
B_total = B_total/tau_i_mul_j**2
crb3 = tf.linalg.diag_part(tf.linalg.inv(tf.abs(B_total)))

scene.preview(paths=paths,show_paths=True,show_orientations=True)

#CRB 2 float64
snr = 1
pi = tf.constant(np.pi, dtype=tf.float64)
alpha=paths.a[0,0,0,0,0,:,0]
tau=paths.tau[0,0,0,:]
length = len(alpha)
tau_i = tf.repeat(tau,length)
tau_i = tf.reshape(tau_i, (length,length))
tau_j = tf.transpose(tau_i)
tau_i_mine_j = tf.cast(tau_i,tf.float64) - tf.cast(tau_j,tf.float64)
tau_i_mul_j = tf.cast(tau_i,tf.float64) * tf.cast(tau_j,tf.float64)
alpha_ij = tf.reshape(alpha, (length,1)) @ tf.reshape(alpha, (1,length))
alpha_ij = tf.cast(alpha_ij,tf.float64)
one = tf.cast(tf.ones((length,length)),tf.float64)
F_alpha= 2*snr*tf.math.abs(alpha_ij)/(tau_i_mul_j**2)
F_cos = (one+(tf.cast(4*(np.pi**2)*(scene.frequency**2),tf.float64) * tau_i_mul_j))*tf.math.cos(tf.cast(2*np.pi*scene.frequency,tf.float64)*tau_i_mine_j)
F_sin = tf.cast(2*np.pi*scene.frequency,tf.float64)*tau_i_mine_j*tf.math.sin(tf.cast(2*np.pi*scene.frequency,tf.float64)*tau_i_mine_j)
F = F_alpha*(F_cos+F_sin)
crb_F = tf.linalg.inv(F)
crb2 = tf.linalg.diag_part(crb_F)

# %%
#CRB 3 float64
snr = 1
H=a[0,0,0,0,0,:,0]
H_H = np.conj(H)
l = len(H)
tau=paths.tau[0,0,0,:]
length = len(alpha)
tau_i = tf.repeat(tau,length)
tau_i = tf.reshape(tau_i, (length,length))
tau_j = tf.transpose(tau_i)
tau_i_mine_j = tf.cast(tau_i,tf.float64) - tf.cast(tau_j,tf.float64)
tau_i_mul_j = tf.cast(tau_i,tf.float64) * tf.cast(tau_j,tf.float64)
B_1 = tf.reshape(H_H, [l,1]) @ tf.reshape(H, [1,l])
one = tf.cast(tf.ones((length,length)),tf.float64)
real = one + tf.cast(4*(np.pi**2)*(scene.frequency**2),tf.float64) * tau_i_mul_j
img = tf.cast(2*np.pi*scene.frequency,tf.float64)*tau_i_mine_j
B_2 = tf.complex(real, img)
B_1 = tf.cast(B_1,tf.complex128)
B_2 = tf.cast(B_2,tf.complex128)
B_total = tf.abs(B_1*B_2)
B_total = tf.cast(B_total,tf.float64)
B_total = B_total/tau_i_mul_j**2
crb3 = tf.linalg.diag_part(tf.linalg.inv(tf.abs(B_total)))

# %%
phi_r = paths.phi_r[0,0,0,:]
phi_t = paths.phi_t[0,0,0,:]
theta_r = paths.theta_r[0,0,0,:]
theta_t = paths.theta_t[0,0,0,:]
print("phi_r: ", phi_r)
print("phi_t: ", phi_t)
print("theta_r: ", theta_r)
print("theta_t: ", theta_t)


