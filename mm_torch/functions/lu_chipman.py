import torch
from .mm import mm_filter


def lu_chipman(M, transpose=True, filter=False):

    # init
    h, w = M.shape[:2]
    M = M.reshape(h, w, 4, 4)
    if filter: M = mm_filter(M)
    if transpose: M = M.transpose(-2, -1)

    # diattenuation matrix
    dvec = torch.stack([M[..., 0, 1], M[..., 0, 2], M[..., 0, 3]], dim=-1) / (M[..., 0, 0][..., None] +  1e-13)
    D = (M[..., 0, 1]**2 + M[..., 0, 2]**2 + M[..., 0, 3]**2)**.5
    D1 = (1 - D**2)**.5
    D = D[..., None, None]
    D1 = D1[..., None, None]
    
    MD = torch.eye(4, dtype=M.dtype, device=M.device)[None, None].repeat(h, w, 1, 1)
    MD[..., 0, 1:] = dvec
    MD[..., 1:, 0] = dvec
    outer_product = dvec[..., None] * dvec[..., None, :] # torch.outer(dvec, dvec)
    MD[..., 1:, 1:] = D1 * torch.eye(3, device=M.device)[None, None].repeat(h, w, 1, 1) + (1 - D1) * outer_product / D**2
    M_0 = torch.linalg.solve(MD, M)  # Equivalent to M / MD
    
    # retardance
    U_R, S_R, V_R = torch.linalg.svd(M_0[..., 1:4, 1:4], full_matrices=True)
    #V_R = V_R.transpose(-1, -2).conj() # matlab's V

    # unit vector to replace rectangular diagonal matrix (capital sigma)
    S_R = torch.diag(torch.ones(3, dtype=M.dtype, device=M.device))[None, None].repeat(h, w, 1, 1)
    S_R[..., -1, -1][torch.sign(torch.det(M)) < 0] = -1 # modification of MR when the determinant of M is negative

    # Construct mR and MR
    mR = U_R @ S_R @ V_R    #.transpose(-2, -1)
    MR = torch.eye(4, dtype=M.dtype, device=M.device)[None, None].repeat(h, w, 1, 1)
    MR[..., 1:4, 1:4] = mR

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


def batched_lc(M, transpose=True, filter=False):

    if M.shape[1] == 16: M = M.permute(0, 2, 3, 1)
    x = lu_chipman(M.flatten(0, 1), transpose=transpose, filter=filter).flatten(-2, -1)
    x = x.view(x.shape[0], *M.shape).permute(1, 0, 2, 3, 4)
    #x = x.view(x.shape[0], *M.shape).permute(1, 0, 4, 2, 3)
    
    return x
