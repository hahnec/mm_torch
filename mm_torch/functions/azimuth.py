import torch


def compute_azimuth(M):
    return torch.atan2(M[..., 7], -M[..., 11]) / 2 + torch.pi/2

def rolling_window_metric(input_tensor, function=torch.std, patch_size=10, step_size=1):
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
    # Ensure the input tensor is 2D
    if input_tensor.dim() != 2:
        raise ValueError("Input tensor must be 2D")
    
    # Unfold the input tensor to create patches
    unfolded = input_tensor.unfold(0, patch_size, step_size).unfold(1, patch_size, step_size)
    
    # unfolded shape is (H - patch_size + 1, W - patch_size + 1, patch_size, patch_size)
    unfolded = unfolded.contiguous().view(-1, patch_size * patch_size)
    
    # Compute the metric along the last dimension
    result = function(unfolded, dim=-1)
    
    # Reshape the result back to 2D
    output_shape = input_tensor.shape[0] - patch_size + 1, input_tensor.shape[1] - patch_size + 1
    result = result.view(output_shape)
    
    return result

def batched_rolling_window_metric(input_tensor, function=torch.std, patch_size=10, step_size=1):
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
    # Ensure the input tensor is batched 2D
    if input_tensor.dim() != 3:
        raise ValueError("Input tensor must be batched 2D")
    
    # Unfold the input tensor to create patches
    unfolded = input_tensor.unfold(1, patch_size, step_size).unfold(2, patch_size, step_size)
    
    # unfolded shape is (B, H - patch_size + 1, W - patch_size + 1, patch_size, patch_size)
    unfolded = unfolded.contiguous().view(-1, patch_size * patch_size)
    
    # Compute the metric along the last dimension
    result = function(unfolded, dim=-1)
    
    # Reshape the result back to batched 2D
    output_shape = input_tensor.shape[0], input_tensor.shape[1] - patch_size + 1, input_tensor.shape[2] - patch_size + 1
    result = result.view(output_shape)
    
    return result
