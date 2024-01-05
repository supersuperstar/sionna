from typing import Any
import mysionna
import mitsuba as mi
from sionna.constants import PI
import tensorflow as tf
from importlib_resources import files
import sensing.targets as targets

class SensingTarget():
    def __init__(self, filename = str(files(targets).joinpath('human.ply')), position=[0,0,0], orientation=[0,0,0], size=[1,1,1],velocity=[0.5,0.5,0],dtype = tf.complex64):
        """_summary_

        Args:
            filename (_type_, optional): _description_. Defaults to str(files(targets).joinpath('human.ply')).
            position (list, optional): _description_. Defaults to [0,0,0].
            orientation (list, optional): _description_. Defaults to [0,0,0].
            size (list, optional): _description_. Defaults to [1,1,1].
            velocity (list, optional): _description_. Defaults to [0.5,0.5,0].
            dtype (_type_, optional): _description_. Defaults to tf.complex64.

        Raises:
            "target must be a .ply file": _description_
            "loading target failed.": _description_
        """
        self._filename = filename
        self._dtype = dtype
        self._rdtype = dtype.real_dtype
        self._position = tf.cast(position,self._rdtype)
        self._orientation = tf.cast(orientation,self._rdtype)
        self._size = tf.cast(size,self._rdtype)
        self._velocity = tf.cast(velocity,self._rdtype)
        # if filename isn't end with '.ply'
        if filename[-4:] != '.ply':
            raise ValueError("target must be a .ply file")
        try:
            self._shape = mi.load_dict({
                'type':'ply',
                'filename':filename,
                'to_world':to_world(self._position,self._orientation,self._size)
            })
        except:
            raise ValueError("loading target failed.")
    
    @property
    def position(self):
        return self._position
    
    @position.setter
    def position(self, value):
        try:
            self._position = tf.cast(value,self._rdtype)
        except:
            raise ValueError("position must be a 3-dim vector")
        self._shape = mi.load_dict({
            'type':'ply',
            'filename':self._filename,
            'to_world':to_world(self._position,self._orientation,self._size)
        })
        
        
    @property
    def orientation(self):
        return self._orientation
    
    @orientation.setter
    def orientation(self, value):
        try:
            self._orientation = tf.cast(value,self._rdtype)
        except:
            raise ValueError("orientation must be a 3-dim vector")
        self._shape = mi.load_dict({
            'type':'ply',
            'filename':self._filename,
            'to_world':to_world(self._position,self._orientation,self._size)
        })
    
    @property
    def size(self):
        return self._size
    
    @size.setter
    def size(self, value):
        self._size = value
        self._shape = mi.load_dict({
            'type':'ply',
            'filename':self._filename,
            'to_world':to_world(self._position,self._orientation,self._size)
        })
    
    @property
    def velocity(self):
        return self._velocity
    
    @velocity.setter
    def velocity(self, value):
        try:
            self._velocity = tf.cast(value,self._rdtype)
        except:
            raise ValueError("velocity must be a 3-dim vector")

def to_world(position,orientation,size):
    orientation = 180. * orientation / PI
    return (
        mi.ScalarTransform4f.translate(position.numpy())
        @ mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=orientation[0])
        @ mi.ScalarTransform4f.rotate(axis=[0, 1, 0], angle=orientation[1])
        @ mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=orientation[2])
        @ mi.ScalarTransform4f.scale(size.numpy())
    )