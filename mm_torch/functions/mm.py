import torch


def mm(A, W, I):
    return torch.matmul(torch.matmul(torch.inverse(A), I), torch.inverse(W))

def mm_pinv(A, W, I):
    return torch.matmul(torch.matmul(torch.linalg.pinv(A), I), torch.linalg.pinv(W))

def mm_solver(A, W, I):
    return torch.linalg.solve(W.transpose(-2, -1), torch.linalg.solve(A, I).transpose(-2, -1)).transpose(-2, -1)

def compute_mm(A, W, I, transpose=True, norm=True):

    # HxWx16 to HxWx4x4 matrix reshaping
    shape = (*A.shape[:2], 4, 4)
    A, W, I = [el.reshape(shape) for el in [A, W, I]]

    if transpose: A, W, I = [el.transpose(-2, -1) for el in [A, W, I]]

    # algebra
    M = mm(A, W, I)

    if transpose: M = M.transpose(-2, -1)

    # normalization
    if norm: M = M / M[..., 0, 0][..., None, None]

    # flattening from HxWx4x4 to HxWx16 
    return M.reshape(*shape[:2], -1)

def batched_mm(A, W, I, transpose=True, norm=True):

    shape = I.shape
    if shape[1] == 16: A, W, I = A.permute(0, 2, 3, 1), W.permute(0, 2, 3, 1), I.permute(0, 2, 3, 1)
    res = compute_mm(A.flatten(1, 2), W.flatten(1, 2), I.flatten(1, 2), transpose, norm)
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
