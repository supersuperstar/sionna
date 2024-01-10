from .scene import Scene
import tensorflow as tf
import mitsuba as mi
import numpy as np

class SensingScene(Scene):
    
    # Default frequency
    DEFAULT_FREQUENCY = 3.5e9 # Hz
    
    # This object is a singleton, as it is assumed that only one scene can be
    # loaded at a time.
    _instance = None
    def __new__(cls, *args, **kwargs): # pylint: disable=unused-argument
        if cls._instance is None:
            instance = object.__new__(cls)

            # Creates fields if required.
            # This is done only the first time the instance is created.

            # Transmitters
            instance._transmitters = {}
            # Receivers
            instance._receivers = {}
            # Cameras
            instance._cameras = {}
            # Transmitter antenna array
            instance._tx_array = None
            # Receiver antenna array
            instance._rx_array = None
            # Radio materials
            instance._radio_materials = {}
            # Scene objects
            instance._scene_objects = {}
            # Sensing targets
            instance._sensing_targets = {}
            # By default, the antenna arrays is applied synthetically
            instance._synthetic_array = True
            # Holds a reference to the interactive preview widget
            instance._preview_widget = None

            # Set the unique instance of the scene
            cls._instance = instance

            # By default, no callable is used for radio materials
            cls._instance._radio_material_callable = None

            # By default, no callable is used for scattering patterns
            cls._instance._scattering_pattern_callable = None

        return cls._instance
    
    def __init__(self, target_names=None, target_velocity=None, env_filename = None, dtype = tf.complex64):
        super().__init__(env_filename = env_filename, dtype = dtype)
        # check if target_names is a list of string
        if target_names is not None:
            if not isinstance(target_names, list):
                raise ValueError('target_names must be a list of string')
            for name in target_names:
                if not isinstance(name, str):
                    raise ValueError('target_names must be a list of string')
        self._target_names = np.array(target_names)
        
        #check if target_velocity is a list of 3D vector
        if target_velocity is not None:
            if not isinstance(target_velocity, list):
                raise ValueError('target_velocity must be a list of 3D vector')
            for v in target_velocity:
                if not isinstance(v, list):
                    raise ValueError('target_velocity must be a list of 3D vector')
                if len(v) != 3:
                    raise ValueError('target_velocity must be a list of 3D vector')
        self._target_velocity = np.array(target_velocity)
    
    # override
    def compute_paths_sensing(self, max_depth=3, method="fibonacci",
                      num_samples=int(1e6), los=True, reflection=True,
                      diffraction=False, scattering=False, scat_keep_prob=0.001,
                      edge_diffraction=False, check_scene=True,
                      scat_random_phases=True, testing=False,return_obj_names=False):
        # pylint: disable=line-too-long
        """_summary_

        Input
        ------
        max_depth : int
            Maximum depth (i.e., number of bounces) allowed for tracing the
            paths. Defaults to 3.

        method : str ("exhaustive"|"fibonacci")
            Ray tracing method to be used.
            The "exhaustive" method tests all possible combinations of primitives.
            This method is not compatible with scattering.
            The "fibonacci" method uses a shoot-and-bounce approach to find
            candidate chains of primitives. Initial ray directions are chosen
            according to a Fibonacci lattice on the unit sphere. This method can be
            applied to very large scenes. However, there is no guarantee that
            all possible paths are found.
            Defaults to "fibonacci".

        num_samples : int
            Number of rays to trace in order to generate candidates with
            the "fibonacci" method.
            This number is split equally among the different transmitters
            (when using synthetic arrays) or transmit antennas (when not using
            synthetic arrays).
            This parameter is ignored when using the exhaustive method.
            Tracing more rays can lead to better precision
            at the cost of increased memory requirements.
            Defaults to 1e6.

        los : bool
            If set to `True`, then the LoS paths are computed.
            Defaults to `True`.

        reflection : bool
            If set to `True`, then the reflected paths are computed.
            Defaults to `True`.

        diffraction : bool
            If set to `True`, then the diffracted paths are computed.
            Defaults to `False`.

        scattering : bool
            if set to `True`, then the scattered paths are computed.
            Only works with the Fibonacci method.
            Defaults to `False`.

        scat_keep_prob : float
            Probability with which a scattered path is kept.
            This is helpful to reduce the number of computed scattered
            paths, which might be prohibitively high in some scenes.
            Must be in the range (0,1). Defaults to 0.001.

        edge_diffraction : bool
            If set to `False`, only diffraction on wedges, i.e., edges that
            connect two primitives, is considered.
            Defaults to `False`.

        check_scene : bool
            If set to `True`, checks that the scene is well configured before
            computing the paths. This can add a significant overhead.
            Defaults to `True`.

        scat_random_phases : bool
            If set to `True` and if scattering is enabled, random uniform phase
            shifts are added to the scattered paths.
            Defaults to `True`.

        testing : bool
            If set to `True`, then additional data is returned for testing.
            Defaults to `False`.

        Output
        ------
        :paths : :class:`~sionna.rt.Paths`
            Simulated paths
        """
        # Trace the paths
        paths = self.compute_paths(max_depth=max_depth, method=method,
                      num_samples=num_samples, los=los, reflection=reflection,
                      diffraction=diffraction, scattering=scattering, scat_keep_prob=scat_keep_prob,
                      edge_diffraction=edge_diffraction, check_scene=check_scene,
                      scat_random_phases=scat_random_phases, testing=testing)
        
        paths_obj_names = self.get_interacting_objects(paths)
        
        paths = self._apply_doppler(paths, self.target_names, self.target_velocity, paths_obj_names)
        
        if return_obj_names:
            return paths, paths_obj_names
        return paths
        
    def get_interacting_objects(self, paths):
        obj_names,wedges_names = self._get_sensing_path_objects_name(paths)
        objects = paths.objects
        
        # expand types to the shape of objects
        types = paths.types[0]
        paths_obj_names = tf.fill(objects.shape, 'None')
        types_repeat = tf.repeat(tf.expand_dims(types, axis=0), repeats=objects.shape[2], axis=0)
        types_repeat = tf.repeat(tf.expand_dims(types_repeat, axis=0), repeats=objects.shape[1], axis=0)
        types_repeat = tf.repeat(tf.expand_dims(types_repeat, axis=0), repeats=objects.shape[0], axis=0)
        
        # mask 
        mask_diff = tf.where(types_repeat == 2, True, False)
        mask_diff = tf.logical_and(mask_diff, objects != -1)
        mask_ref_scatt = tf.where(tf.logical_or(types_repeat == 1, types_repeat == 3), True, False)
        mask_ref_scatt = tf.logical_and(mask_ref_scatt, objects != -1)
        indices_diff = tf.where(mask_diff)
        indices_ref_scatt = tf.where(mask_ref_scatt)
        
        # update
        updates_diff = tf.gather(wedges_names, tf.reshape(objects[mask_diff], [-1]))
        updates_ref_scatt = tf.gather(obj_names, tf.reshape(objects[mask_ref_scatt], [-1]))
        paths_obj_names = tf.tensor_scatter_nd_update(paths_obj_names, indices_diff, updates_diff)
        paths_obj_names = tf.tensor_scatter_nd_update(paths_obj_names, indices_ref_scatt, updates_ref_scatt)

        return paths_obj_names
        
    def _get_sensing_path_objects_name(self, paths):
        """find the names of objects in the xml file,
        objects' names must be with the type 'mesh-name-XXX',XXX is the meterial or other string.
        for scattering and reflection ,this method return the name of object.
        and for diffraction ,this method return the name of two objects that make the wedge and concatenate them with '&'.
        Args:
            paths (_type_): from compute_paths method

        Raises:
            ValueError: _description_

        Returns:
            obj_names : [len(shapes)], list
                
                
            wedges_names : [len(wedges)], list
                the names of wedges
        """
        mi_scene = self.mi_scene
        wedges_2_objects = self._solver_paths._wedges_objects
        wedges_names = []
        obj_names = []
        for _,s in enumerate(mi_scene.shapes()):
            name = s.id().split('-')[1] 
            obj_names.append(name)

        for [obj1,obj2] in wedges_2_objects:
            if obj1==obj2:
                wedges_names.append(obj_names[obj1])
            else:
                wedges_names.append(obj_names[obj1]+'&'+obj_names[obj2])
        
        return obj_names, wedges_names
    
    def _apply_doppler(self,paths,names,velocity,paths_obj_names):
        # check if names and velocity are the same length
        if names.shape[0] != velocity.shape[0]:
            raise ValueError('names and velocity must be the same length')
        
        return paths
        
    @property
    def target_names(self):
        return self._target_names
    
    @target_names.setter
    def target_names(self, value):
        if not isinstance(value, list):
            raise ValueError('target_names must be a list of string')
        for name in value:
            if not isinstance(name, str):
                raise ValueError('target_names must be a list of string')
        self._target_names = np.array(value)
    
    @property
    def target_velocity(self):
        return self._target_velocity
    
    @target_velocity.setter
    def target_velocity(self, value):
        if not isinstance(value, list):
            raise ValueError('target_velocity must be a list of 3D vector')
        for v in value:
            if not isinstance(v, list):
                raise ValueError('target_velocity must be a list of 3D vector')
            if len(v) != 3:
                raise ValueError('target_velocity must be a list of 3D vector')
        self._target_velocity = np.array(value)
        

def load_sensing_scene(filename=None, target_names=None, target_velocity=None, dtype = tf.complex64):
    """_summary_

    Input
    ------
    filename : str
        Path to the scene file.

    target_names : list of str
        List of target names.

    target_velocity : list of 3D vectors
        List of target velocities.

    dtype : tf.dtype
        Data type of the tensors. Defaults to `tf.complex64`.

    Output
    ------
    :scene : :class:`~sionna.rt.SensingScene`
        Scene object
    """
    # Load the scene
    if filename is None:
        filename = "__empty__"
    return SensingScene(target_names, target_velocity, filename, dtype=dtype)