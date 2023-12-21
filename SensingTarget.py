from typing import Any
import mysionna
import mitsuba as mi
from sionna.constants import PI

class SensingTarget():
    def __init__(self, position, orientation, size, filename):
        """_summary_

        Args:
            position (_type_): _description_
            orientation (_type_): _description_
            shape (_type_): must be string path or mitsuba dict
        """
        self._position = position
        self._orientation = orientation
        self._size = size
        self._filename = filename
        
        self._shape = mi.load_dict({
            'type':'ply',
            'filename':filename,
            'to_world':to_world(position,orientation,size)
        })
    
    @property
    def position(self):
        return self._position
    
    @position.setter
    def position(self, value):
        self._position = value
        self._shape = mi.load_dict({
            'type':'ply',
            'filename':self._filename,
            'to_world':to_world(value,self._orientation,self._size)
        })
        
        
    @property
    def orientation(self):
        return self._orientation
    
    @orientation.setter
    def orientation(self, value):
        self._orientation = value
        self._shape = mi.load_dict({
            'type':'ply',
            'filename':self._filename,
            'to_world':to_world(self._position,value,self._size)
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
            'to_world':to_world(self._position,self._orientation,value)
        })

def to_world(position,orientation,size):
    orientation = 180. * orientation / PI
    return (
        mi.ScalarTransform4f.translate(position.numpy())
        @ mi.ScalarTransform4f.rotate(axis=[0, 0, 1], angle=orientation[0])
        @ mi.ScalarTransform4f.rotate(axis=[0, 1, 0], angle=orientation[1])
        @ mi.ScalarTransform4f.rotate(axis=[1, 0, 0], angle=orientation[2])
        @ mi.ScalarTransform4f.scale([0.5 * size[0], 0.5 * size[1], 1])
    )