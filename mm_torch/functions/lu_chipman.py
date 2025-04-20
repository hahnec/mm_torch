import torch
from .mm_filter import mm_filter, charpoly


def lu_chipman(
        M, 
        mask=None, 
        transpose=True, 
        filter_opt=False, 
        svd_fun=lambda x: torch.linalg.svd(x, full_matrices=False),
        ):

    # init
    chs = M.shape[-1]
    if chs == 16: M = M.view(*M.shape[:-1], 4, 4)
    if transpose: M = M.transpose(-2, -1)
    if filter_opt: M, _ = mm_filter(M)
    if mask is None: mask = charpoly(M)
    shape = M.shape[:-2]

    M_0, MD = diattenuation_matrix(M)

    # retardance
    mask = mask & ~torch.isnan(M_0[..., 1:4, 1:4].sum(-1).sum(-1))
    U_R = torch.zeros_like(M_0[..., 1:4, 1:4])
    V_R = torch.zeros_like(M_0[..., 1:4, 1:4])
    U_R[mask], _, V_R[mask] = svd_fun(M_0[..., 1:4, 1:4][mask])

    # unit vector to replace diagonal matrix (capital sigma)
    S_R = torch.eye(3, dtype=M.dtype, device=M.device)[None,].repeat(*shape, 1, 1)
    S_R[..., -1, -1][torch.sign(torch.det(M)) < 0] = -1 # modification of MR when the determinant of M is negative

    # construct MR
    MR = torch.eye(4, dtype=M.dtype, device=M.device)[None,].repeat(*shape, 1, 1)
    MR[..., 1:4, 1:4] = U_R @ S_R @ V_R

    # depolarization
    Mdelta = torch.matmul(M_0, MR.transpose(-2, -1))

    if transpose:
        MD = MD.transpose(-2, -1)
        MR = MR.transpose(-2, -1)
        Mdelta = Mdelta.transpose(-2, -1)

    y = torch.stack([MD, MR, Mdelta])
    if chs == 16: y = y.flatten(-2, -1)

    return y

def diattenuation_matrix(M):

    shape = M.shape[:-2]

    dvec = torch.stack([M[..., 0, 1], M[..., 0, 2], M[..., 0, 3]], dim=-1) / (M[..., 0, 0][..., None] +  1e-13)
    D = (M[..., 0, 1]**2 + M[..., 0, 2]**2 + M[..., 0, 3]**2)**.5
    #D1 = (1 - D**2)**.5
    D1 = (torch.clamp(1 - D**2, min=0))**.5
    D, D1 = D[..., None, None], D1[..., None, None]
    
    #MD = torch.eye(4, dtype=M.dtype, device=M.device)[None,]*len(shape)
    #MD = MD.repeat(*shape, 1, 1)
    MD = torch.eye(4, dtype=M.dtype, device=M.device).expand(*shape, 4, 4).clone()
    MD[..., 0, 1:] = dvec
    MD[..., 1:, 0] = dvec
    outer_product = dvec[..., None] * dvec[..., None, :] # torch.outer(dvec, dvec)
    #eye = torch.eye(3, device=M.device)[None,]*len(shape)
    #eye = eye.repeat(*shape, 1, 1)
    eye = torch.eye(3, device=M.device).expand(*shape, 3, 3)
    MD[..., 1:, 1:] = D1 * eye + (1 - D1) * outer_product / (D**2 + 1e-13)
    M_0 = M @ torch.linalg.inv(MD)
    
    #MD[D.squeeze(-1).squeeze(-1)==0] = torch.eye(4, dtype=M.dtype, device=M.device).repeat(torch.sum(D==0), 1, 1)
    #M_0[D.squeeze(-1).squeeze(-1)==0] = M[D.squeeze(-1).squeeze(-1)==0]
    zero_mask = (D.squeeze(-1).squeeze(-1) == 0)
    MD = torch.where(zero_mask[..., None, None], torch.eye(4, device=M.device), MD)
    M_0 = torch.where(zero_mask[..., None, None], M, M_0)

    return M_0, MD

def batched_lc(M, mask=None, transpose=True):

    if M.shape[1] == 16: M = M.permute(0, 2, 3, 1)
    y = lu_chipman(M, mask=mask, transpose=transpose)
    if M.shape[1] == 16: y = y.permute(0, 1, 4, 2, 3)

    return y
