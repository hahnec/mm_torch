import torch


def mm(A, W, I):
    return torch.matmul(torch.matmul(torch.inverse(A), I), torch.inverse(W))

def mm_pinv(A, W, I):
    return torch.matmul(torch.matmul(torch.linalg.pinv(A), I), torch.linalg.pinv(W))

def mm_solver(A, W, I):
    return torch.linalg.solve(W.transpose(-2, -1), torch.linalg.solve(A, I).transpose(-2, -1)).transpose(-2, -1)

def compute_mm(A, W, I, transpose=True, norm=True, filter=False):

    # HxWx16 to HxWx4x4 matrix reshaping
    shape = (*A.shape[:2], 4, 4)
    A, W, I = [el.reshape(shape) for el in [A, W, I]]

    if transpose: A, W, I = [el.transpose(-2, -1) for el in [A, W, I]]

    # algebra
    M = mm(A, W, I)

    if transpose: M = M.transpose(-2, -1)

    # normalization
    if norm: M = M / M[..., 0, 0][..., None, None]

    # filtering
    if filter: M, mask = mm_filter(M)

    # flattening from HxWx4x4 to HxWx16 
    return M.reshape(*shape[:2], -1)

def mm_filter(M, criterion = 1e-4):

    h, w = M.shape[:2]
    chs = M.shape[-1]
    M = M.reshape(h, w, 4, 4) if chs == 16 else M

    # Complex-valued covariance matrix
    N = 0.25 * torch.stack([
        M[..., 0, 0] + M[..., 1, 1] + M[..., 0, 1] + M[..., 1, 0],
        M[..., 0, 2] + M[..., 1, 2] + 1j * (M[..., 0, 3] + M[..., 1, 3]),
        M[..., 2, 0] + M[..., 2, 1] - 1j * (M[..., 3, 0] + M[..., 3, 1]),
        M[..., 2, 2] + M[..., 3, 3] + 1j * (M[..., 2, 3] - M[..., 3, 2]),
        M[..., 0, 2] + M[..., 1, 2] - 1j * (M[..., 0, 3] + M[..., 1, 3]),
        M[..., 0, 0] - M[..., 1, 1] - M[..., 0, 1] + M[..., 1, 0],
        M[..., 2, 2] - M[..., 3, 3] - 1j * (M[..., 2, 3] + M[..., 3, 2]),
        M[..., 2, 0] - M[..., 2, 1] - 1j * (M[..., 3, 0] - M[..., 3, 1]),
        M[..., 2, 0] + M[..., 2, 1] + 1j * (M[..., 3, 0] + M[..., 3, 1]),
        M[..., 2, 2] - M[..., 3, 3] + 1j * (M[..., 2, 3] + M[..., 3, 2]),
        M[..., 0, 0] - M[..., 1, 1] + M[..., 0, 1] - M[..., 1, 0],
        M[..., 0, 2] - M[..., 1, 2] + 1j * (M[..., 0, 3] - M[..., 1, 3]),
        M[..., 2, 2] + M[..., 3, 3] - 1j * (M[..., 2, 3] - M[..., 3, 2]),
        M[..., 2, 0] - M[..., 2, 1] + 1j * (M[..., 3, 0] - M[..., 3, 1]),
        M[..., 0, 2] - M[..., 1, 2] - 1j * (M[..., 0, 3] - M[..., 1, 3]),
        M[..., 0, 0] + M[..., 1, 1] - M[..., 0, 1] - M[..., 1, 0]
    ], dim=-1).reshape(*M.shape[:-2], 4, 4)

    # Eigen decomposition
    #D, P = torch.linalg.eig(N)
    #D = torch.sort(D.real, dim=-1)[0] # matlab style
    D, P = torch.linalg.eigh(N)

    # set negative eigenvalues to zero and 
    invalid_mask = torch.any(D.real < -1*criterion, dim=-1)#[..., None, None].repeat(1,1,4,4)
    eigenvalues = torch.diag_embed(D).to(P.dtype)
    eigenvalues[invalid_mask, ...] = 1e-5
    newN = torch.matmul(torch.matmul(P, eigenvalues), torch.inverse(P))
    #newN = P @ eigenvalues @ P.transpose(-2, -1).conj()
    
    A = torch.tensor([
        [1, 0, 0, 1],
        [1, 0, 0, -1],
        [0, 1, 1, 0],
        [0, 1j, -1j, 0]
    ], dtype=N.dtype, device=M.device).unsqueeze(0).unsqueeze(0)
    
    F = torch.stack([
        newN[..., 0,0], newN[..., 0,1], newN[..., 1,0], newN[..., 1,1],
        newN[..., 0,2], newN[..., 0,3], newN[..., 1,2], newN[..., 1,3],
        newN[..., 2,0], newN[..., 2,1], newN[..., 3,0], newN[..., 3,1],
        newN[..., 2,2], newN[..., 2,3], newN[..., 3,2], newN[..., 3,3],
    ], dim=-1).reshape(h,w,4,4)
    
    M_filtered = torch.matmul(torch.matmul(A, F), torch.inverse(A))
    M_filtered = M_filtered / M_filtered[..., 0, 0][..., None, None].real
    M_filtered = M_filtered.real
        
    M[invalid_mask, ...] = M_filtered[invalid_mask, ...]

    return M.reshape(h, w, 16) if chs == 16 else M, invalid_mask

def batched_mm(A, W, I, transpose=True, norm=True, filter=False):
    shape = I.shape
    if shape[1] == 16: A, W, I = A.permute(0, 2, 3, 1), W.permute(0, 2, 3, 1), I.permute(0, 2, 3, 1)
    res = compute_mm(A.flatten(1, 2), W.flatten(1, 2), I.flatten(1, 2), transpose, norm, filter)
    res = res.view(shape[0], *shape[2:], shape[1]) if shape[1] == 16 else res.view(*shape)
    if shape[1] == 16: res = res.permute(0, 3, 1, 2)
    return res


if __name__ == '__main__':

    torch.manual_seed(3008)

    A = torch.randn(128, 128, 16, dtype=torch.double)
    W = torch.randn(128, 128, 16, dtype=torch.double)
    I = torch.randn(128, 128, 16, dtype=torch.double)

    # torch
    M_torch = compute_mm(A, W, I, transpose=True, norm=True)

    # batched inputs
    A = A.unsqueeze(0).repeat(8, 1, 1, 1)
    W = W.unsqueeze(0).repeat(8, 1, 1, 1)
    I = I.unsqueeze(0).repeat(8, 1, 1, 1)

    batched_M_torch = batched_mm(A, W, I, filter=True)

    assert batched_M_torch.shape == (8, 128, 128, 16), 'dimensions mismatch'
