import torch
from .mm_filter import EIG as mm_filter


def lu_chipman(
        M, 
        mask=None, 
        transpose=True, 
        svd_fun=lambda x: torch.linalg.svd(x, full_matrices=False),
        ):

    # init
    hw = M.shape[:-2]
    M = M.reshape(*hw, 4, 4)
    if transpose: M = M.transpose(-2, -1)
    if mask is None: mask = mm_filter(M)

    M_0, MD = diattenuation_matrix(M)

    # retardance
    U_R = torch.zeros_like(M_0[..., 1:4, 1:4])
    V_R = torch.zeros_like(M_0[..., 1:4, 1:4])
    U_R[mask], _, V_R[mask] = svd_fun(M_0[..., 1:4, 1:4][mask])

    # unit vector to replace rectangular diagonal matrix (capital sigma)
    S_R = torch.eye(3, dtype=M.dtype, device=M.device)[None,]*len(hw)
    S_R = S_R.repeat(*hw, 1, 1)
    S_R[..., -1, -1][torch.sign(torch.det(M)) < 0] = -1 # modification of MR when the determinant of M is negative

    # Construct MR
    MR = torch.eye(4, dtype=M.dtype, device=M.device)[None,]*len(hw)
    MR = MR.repeat(*hw, 1, 1)
    MR[..., 1:4, 1:4] = U_R @ (S_R @ V_R)    #.transpose(-2, -1)

    # depolarization
    Mdelta = torch.matmul(M_0, MR.transpose(-2, -1))

    # exclude imaginary parts
    MD = MD.real
    MR = MR.real
    Mdelta = Mdelta.real

    if transpose:
        MD = MD.transpose(-2, -1)
        MR = MR.transpose(-2, -1)
        Mdelta = Mdelta.transpose(-2, -1)

    return torch.stack([MD, MR, Mdelta])

def diattenuation_matrix(M):

    hw = M.shape[:-2]

    dvec = torch.stack([M[..., 0, 1], M[..., 0, 2], M[..., 0, 3]], dim=-1) / (M[..., 0, 0][..., None] +  1e-13)
    D = (M[..., 0, 1]**2 + M[..., 0, 2]**2 + M[..., 0, 3]**2)**.5
    D1 = (1 - D**2)**.5
    D, D1 = D[..., None, None], D1[..., None, None]
    
    MD = torch.eye(4, dtype=M.dtype, device=M.device)[None,]*len(hw)
    MD = MD.repeat(*hw, 1, 1)
    MD[..., 0, 1:] = dvec
    MD[..., 1:, 0] = dvec
    outer_product = dvec[..., None] * dvec[..., None, :] # torch.outer(dvec, dvec)
    eye = torch.eye(3, device=M.device)[None,]*len(hw)
    eye = eye.repeat(*hw, 1, 1)
    MD[..., 1:, 1:] = D1 * eye + (1 - D1) * outer_product / D**2
    M_0 = M @ torch.linalg.inv(MD)
    MD[D.squeeze()==0] = torch.eye(4, dtype=M.dtype, device=M.device)
    M_0[D.squeeze()==0] = M[D.squeeze()==0]

    return M_0, MD

def batched_lc(M, mask=None, transpose=True):

    if M.shape[1] == 16: M = M.permute(0, 2, 3, 1)
    x= lu_chipman(M.flatten(0, 1), mask=mask.flatten(0, 1), transpose=transpose).flatten(-2, -1)
    x = x.view(x.shape[0], *M.shape).permute(1, 0, 2, 3, 4)
    #x = x.view(x.shape[0], *M.shape).permute(1, 0, 4, 2, 3)
    
    return x
