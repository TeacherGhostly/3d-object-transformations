#region
from OpenGL.GL import glUniformMatrix3fv, glUniformMatrix4fv, GL_TRUE, GL_FALSE
import numpy as np
import math
#endregion
#region
Vec3 = list[float, float, float]

class Mat4:

    def __init__(self, 
        p : "Mat4 | Mat3 | np.ndarray | list[list[float]]" = None):
        """
            Construct a Mat4 from a description.

            Parameters:

                p: description, can be an iterable python object
                    like list or tuple, a numpy array, or another
                    Mat3 or Mat4.
        """
        if p is None:
            self.mat_data = np.identity(4, dtype=np.float32)
        elif isinstance(p, Mat3):
            self.mat_data = np.identity(4, dtype=np.float32)
            self.mat_data[0:3, 0:3] = p.mat_data
        elif isinstance(p, Mat4):
            self.mat_data = p.mat_data.copy()
        else:
            self.mat_data = np.array(p, dtype = np.float32)

    def __mul__(self, 
        other: "np.ndarray | list[float] | Mat4") \
        -> "np.ndarray | list[float] | Mat4":
        """
            overload the multiplication operator to enable sane 
            looking transformation expressions!
        """

        """
            if it is a list, we let numpy attempt to convert the data
            we then return it as a list also (the typical use case is 
            for transforming a vector). Could be made more robust...
        """
        other_data = other
        if isinstance(other, Mat4):
            other_data = other.mat_data
        result = np.dot(other_data, self.mat_data)
        if isinstance(other, list):
            #python list
            return list(result)
        if isinstance(other, Mat4):
            #Mat4
            return Mat4(result)
        #numpy array
        return result
    
    def get_data(self) -> np.ndarray:
        """
            Returns the matrix's data as a contiguous array for
            upload to OpenGL
        """
        return self.mat_data

    def _inverse(self) -> "Mat4":
        """
            Returns an inverted copy, does not change the object 
            (for clarity use the global function instead)
            only implemented as a member to make it easy to overload
            based on matrix class (i.e. 3x3 or 4x4)
        """
        return Mat4(np.linalg.inv(self.mat_data))
    
    def _affine_inverse(self) -> "Mat4":
        """
            Returns an inverted copy, does not change the object.
            
            Matrices which represent affine transformations have
            closed form inverses. This is actually how the lookat
            transform is calculated.
        """
        A = self.mat_data
        data = (
            (A[0][0], A[1][0], A[2][0], 0.0),
            (A[0][1], A[1][1], A[2][1], 0.0),
            (A[0][2], A[1][2], A[2][2], 0.0),
            (-np.dot(A[0], A[3]), -np.dot(A[1], A[3]), -np.dot(A[2], A[3]), 1.0)
        )
        return Mat4(data)

    def _transpose(self) -> "Mat4":
        """
            Returns a matrix representing the transpose of
            this matrix. This matrix is not altered.
        """
        return Mat4(self.mat_data.T)

    def _set_open_gl_uniform(self, location: int) -> None:
        """
            Uploads the matrix to the given location.
        """
        glUniformMatrix4fv(location, 1, GL_FALSE, self.mat_data)

class Mat3:
    
    def __init__(self, 
        p : "Mat4 | Mat3 | np.ndarray | list[list[float]]" = None):
        """
            Construct a Mat3 from a description.

            Parameters:

                p: description, can be an iterable python object
                    like list or tuple, a numpy array, or another
                    Mat3 or Mat4.
        """
        if p is None:
            self.mat_data = np.identity(3, dtype=np.float32)
        elif isinstance(p, Mat3):
            self.mat_data = p.mat_data.copy()
        elif isinstance(p, Mat4):
            self.mat_data = np.identity(3, dtype = np.float32)
            self.mat_data[0:3, 0:3] = p.mat_data[0:3, 0:3]
        else:
            self.mat_data = np.array(p, dtype = np.float32)

    def __mul__(self, 
        other: "np.ndarray | list[float] | Mat3") \
        -> "np.ndarray | list[float] | Mat3":
        """
            overload the multiplication operator to enable sane 
            looking transformation expressions!
        """

        """
            if it is a list, we let numpy attempt to convert the data
            we then return it as a list also (the typical use case is 
            for transforming a vector). Could be made more robust...
        """
        other_data = other
        if isinstance(other, Mat3):
            other_data = other.mat_data
        result = np.dot(other_data, self.mat_data)
        if isinstance(other, list):
            #python list
            return list(result)
        if isinstance(other, Mat3):
            #Mat4
            return Mat3(result)
        #numpy array
        return result
    
    def get_data(self) -> np.ndarray:
        """
            Returns the matrix's data as a contiguous array for
            upload to OpenGL
        """
        return self.mat_data

    def _inverse(self) -> "Mat3":
        """
            Returns an inverted copy, does not change the object 
            (for clarity use the global function instead) only 
            implemented as a member to make it easy to overload 
            based on matrix class (i.e. 3x3 or 4x4)
        """
        return Mat3(np.linalg.inv(self.mat_data))

    def _transpose(self) -> "Mat3":
        """
            Returns a transposed copy, does not change the object 
            (for clarity use the global function instead) only 
            implemented as a member to make it easy to overload 
            based on matrix class (i.e. 3x3 or 4x4)
        """
        return Mat3(self.mat_data.T)

    def _set_open_gl_uniform(self, location: int) -> None:
        """
            Uploads the matrix to the given location.
        """

        glUniformMatrix3fv(location, 1, GL_FALSE, self.mat_data)
#endregion
#
# matrix consruction functions
#
#region
def make_translation(x: float, y: float, z: float) -> Mat4:
    """
        Returns a 4x4 matrix representing a translation by the
        given amounts in the x,y,z axes.
    """

    return Mat4([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,1,0],
                 [x,y,z,1]])
 
def make_scale(x: float, y: float, z: float) -> Mat4:
    """
        Returns a 4x4 matrix representing a scale transform
        of the given amount in the x,y,z axes.
    """
    return Mat4([[x,0,0,0],
                 [0,y,0,0],
                 [0,0,z,0],
                 [0,0,0,1]])

def make_rotation_y(angle: float) -> Mat4:
    """
        Returns a 4x4 matrix representing a rotation around
        the y axis by the given angle (in radians).
    """

    c = math.cos(angle)
    s = math.sin(angle)
    return Mat4([[ c, 0, s, 0],
                 [ 0, 1, 0, 0],
                 [-s, 0, c, 0],
                 [ 0, 0, 0, 1]])

def make_rotation_x(angle: float) -> Mat4:
    """
        Returns a 4x4 matrix representing a rotation around
        the x axis by the given angle (in radians).
    """
    c = math.cos(angle)
    s = math.sin(angle)
    return Mat4([[1,  0, 0, 0],
                 [0,  c, s, 0],
                 [0, -s, c, 0],
                 [0,  0, 0, 1]])

def make_rotation_z(angle: float) -> Mat4:
    """
        Returns a 4x4 matrix representing a rotation around
        the z axis by the given angle (in radians).
    """
    c = math.cos(angle)
    s = math.sin(angle)
    return Mat4([[ c, s, 0, 0],
                 [-s, c, 0, 0],
                 [ 0, 0, 1, 0],
                 [ 0, 0, 0, 1]])
#endregion
# 
# Matrix operations
#
#region
def inverse(mat: Mat3 | Mat4) -> Mat3 | Mat4:
    """
        returns an inverted copy, does not change the object.
    """
    return mat._inverse()

def transpose(mat: Mat3 | Mat4) -> Mat3 | Mat4:
    """
        returns a transposed copy, does not change the object.
    """
    return mat._transpose()
#endregion
#
# vector operations
#
#region
def normalize(v: np.ndarray) -> np.ndarray:
    """
        Returns a normalized copy of the given vector.
    """
    norm = np.linalg.norm(v)
    return v / norm
#endregion