import torch
import numpy as np
import time
from .camera_types import get_cam_params, default_cam_type


def read_cod_data_X3D(filename, cam_type=None, raw_flag=0, verbose=0):
    if cam_type is None:
        cam_type = default_cam_type()
    
    shp2 = get_cam_params(cam_type)[0]
    
    # Initial Time (Total Performance)
    t = time.perf_counter()
    
    header_size = 140  # header size of the '.cod' file
    m = 16  # components of the polarimetric data in the '.cod' file
    
    # Reading the Data (loading) NB: binary data is float, but the output array is *cast* to double
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=np.single).astype(np.double)
        X3D = torch.from_numpy(data)

    if raw_flag:
        # Calibration files WITH HEADER (to be discarded)
        # Reshaping and transposing the shape of the imported array (from Fortran- -> C-like)
        X3D = X3D[header_size:].reshape([shp2[1], shp2[0], m]).permute(1, 0, 2)
    else:
        # Other Files from libmpMuelMat processing WITHOUT HEADER
        # Reshaping the imported array (already C-like)
        X3D = X3D.reshape([shp2[0], shp2[1], m])
    
    # Final Time (Total Performance)
    telaps = time.perf_counter() - t
    if verbose:
        print(' ')
        print(' >> read_cod_data_X3D Performance: Elapsed time = {:.3f} s'.format(telaps))
    
    return X3D


def write_cod_data_X3D(X3D, filename, verbose=1):
    shp3 = X3D.shape
    if len(shp3) != 3:
        raise Exception(
            'Input: "X3D" should have shape of a 3D image, e.g. (idx0, idx1, idx2). The shape value was found: {}'.format(shp3))
    if shp3[-1] != 16:
        raise Exception(
            'Input: "X3D" should have shape 16 components in the last dimension. The number of elements in the last dimension is: {}'.format(shp3[-1]))
    
    with open(filename, "wb") as f:
        X3D.to(torch.float32).numpy().tofile(f)
    
    if verbose:
        print(' ')
        print(' >> Exported X3D .cod file as:', filename)
    
    return None


def read_cod_data_X2D(filename, cam_type=None, verbose=0):
    if cam_type is None:
        cam_type = default_cam_type()
    
    shp2 = get_cam_params(cam_type)[0]
    
    # Initial Time (Total Performance)
    t = time.perf_counter()
    
    # Reading the Data (loading) NB: binary data is float, but the output array is *cast* to double
    with open(filename, "rb") as f:
        data = np.fromfile(f, dtype=np.single).astype(np.double)
        X2D = torch.from_numpy(data)

    # Reshaping the imported array (already C-like)
    X2D = X2D.reshape(shp2)
    
    # Final Time (Total Performance)
    telaps = time.perf_counter() - t
    if verbose:
        print(' ')
        print(' >> read_cod_data_X2D Performance: Elapsed time = {:.3f} s'.format(telaps))
    
    return X2D


def write_cod_data_X2D(X2D, filename, verbose=1):
    shp2 = X2D.shape
    if len(shp2) != 2:
        raise Exception(
            'Input: "X2D" should have shape of a 2D image, e.g. (idx0, idx1). The shape value was found: {}'.format(shp2))
    
    with open(filename, "wb") as f:
        X2D.to(torch.float32).numpy().tofile(f)
    
    if verbose:
        print(' ')
        print(' >> Exported X2D .cod file as:', filename)
    
    return None
