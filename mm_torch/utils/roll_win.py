import torch


def rolling_window_metric(input_tensor, patch_size=4, function=torch.std, step_size=1):
    """
    Computes a provided metric function over a patch_size x patch_size patch of entries that moves
    across the entire 2-dimensional input in a rolling window fashion.
    
    Args:
        input_tensor (torch.Tensor): The 2D input tensor of shape (H, W)
        patch_size (int): The size of the patch to compute the metric function. Default is 10.
        function (Callable): The metric function
        
    Returns:
        torch.Tensor: A 2D tensor of results with shape (H - patch_size + 1, W - patch_size + 1)
    """

    # ensure the input tensor is 2D
    if input_tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D")
    
    # unfold the input tensor to create patches
    unfolded = input_tensor.unfold(0, patch_size, step_size).unfold(1, patch_size, step_size)
    
    # unfolded shape is (H - patch_size + 1, W - patch_size + 1, patch_size, patch_size)
    unfolded = unfolded.contiguous().view(-1, patch_size * patch_size)
    
    # compute the metric along the last dimension
    result = function(unfolded, dim=-1)
    
    # reshape the result back to 2D
    output_shape = input_tensor.shape[0] - patch_size + 1, input_tensor.shape[1] - patch_size + 1
    result = result.view(output_shape)
    
    return result


def batched_rolling_window_metric(input_tensor, patch_size=4, function=torch.std, perc=1, step_size=1):
    """
    Computes a provided metric function over a patch_size x patch_size patch of entries that moves
    across the entire batched 2-dimensional input in a rolling window fashion.
    
    Args:
        input_tensor (torch.Tensor): The batched 2D input tensor of shape (B, H, W)
        patch_size (int): The size of the patch to compute the metric function. Default is 10.
        function (Callable): The metric function
        
    Returns:
        torch.Tensor: A batched 2D tensor of results with shape (B, H - patch_size + 1, W - patch_size + 1)
    """

    shape = input_tensor.shape
    if len(shape) != 3: input_tensor = input_tensor.reshape(-1, *shape[-2:])

    # unfold the input tensor to create patches
    unfolded = input_tensor.unfold(1, patch_size, step_size).unfold(2, patch_size, step_size)
    
    # unfolded shape is (B, H - patch_size + 1, W - patch_size + 1, patch_size, patch_size)
    unfolded = unfolded.contiguous().view(-1, patch_size * patch_size)
    
    # compute the metric
    result = function(unfolded)
    
    # reshape the result back to batched 2D
    output_shape = *shape[:-2], (input_tensor.shape[1]-patch_size)//step_size + 1, (input_tensor.shape[2]-patch_size)//step_size + 1
    result = result.view(output_shape)

    if 0 < perc < 1: result = percentile_clip(result, perc, mode='peaks')

    # padding
    pad_half = (patch_size-1) // 2
    pad_odd = (patch_size-1) % 2
    result = torch.nn.functional.pad(result, [pad_half, pad_half+pad_odd, pad_half, pad_half+pad_odd], mode='replicate')

    if len(shape) != 3: result = result.view(*shape[:-2], *result.shape[-2:])
    
    return result


def percentile_clip(img, perc=0.95, mode=None):

    mode = 'both' if mode == None else mode
    if mode in ('peaks', 'both'):
        pvals_max = torch.quantile(img.flatten(-2, -1), perc, dim=-1)
        pvals_max_expanded = pvals_max[..., None, None].expand_as(img)
        img = torch.minimum(img, pvals_max_expanded)
    if mode in ('lows', 'both'):
        pvals_min = torch.quantile(img.flatten(-2, -1), 1-perc, dim=-1)
        pvals_min_expanded = pvals_min[..., None, None].expand_as(img)
        img = torch.maximum(img, pvals_min_expanded)

    return img


def circstd(samples, high=2*torch.pi, low=0, dim=-1):
    """
    Calculate the circular standard deviation for an array of circular data along a specified dimension.

    Parameters:
    samples (torch.Tensor): Input tensor of circular data.
    high (float): Upper bound of the circular range (default: 2*pi).
    low (float): Lower bound of the circular range (default: 0).
    dim (int): Dimension along which to calculate the circular standard deviation (default: -1).

    Returns:
    torch.Tensor: Circular standard deviation along the specified dimension.
    """

    # convert samples to radians
    samples = (samples - low) * 2 * torch.pi / (high - low)
    
    # compute sum of sines and cosines along the specified dimension
    sin_sum = torch.sum(torch.sin(samples), dim=dim)
    cos_sum = torch.sum(torch.cos(samples), dim=dim)
    
    # compute resultant vector length R
    R = torch.sqrt(sin_sum**2 + cos_sum**2) / samples.size(dim)

    # compute circular standard deviation
    circ_std = torch.sqrt(-2 * torch.log(R))

    return circ_std * (high - low) / (2 * torch.pi)

if __name__ == '__main__':

    angles = torch.tensor([0, torch.pi/2, torch.pi, 3*torch.pi/2])  # 0, 90, 180, 270 degrees

    # custom implementation
    print(circstd(angles))

    # reference implementation
    from scipy.stats import circstd
    print(circstd(angles))
