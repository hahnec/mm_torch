import torch

acos_nansafe = lambda x: torch.acos(torch.clip(x, -1., 1.))

def extract_depolarization(Mdelta):
    """
    Extraction of the total depolarization from a Mueller matrix Mdelta using the criterion from the Lu and Chipman polar decomposition

    Parameters:
    Mdelta: Tensor of shape (H x W x 4 x 4), representing the batch of Mueller matrices of the depolarizer

    Returns:
    total_depolarization: Tensor of shape (H x W), representing the total depolarization for each matrix
    """

    if Mdelta.shape[-1] == 16: Mdelta = Mdelta.view(*Mdelta.shape[:-1], 4, 4)
    values = torch.stack([Mdelta[..., 1, 1], Mdelta[..., 2, 2], Mdelta[..., 3, 3]], dim=-1)
    values = values.abs().nanmean(-1)

    # treat outliers as zeros (after subtracting) for numerical consistency
    values[torch.isnan(values) | (values <= 0)] = 0

    return 1 - values


def extract_diattenuation(MD, transpose=True):
    """
    Extraction of diattenuation parameters from a diattenuation matrix.

    Parameters:
        MD (torch.Tensor): Mueller matrix of the diattenuator (shape: H x W x 4 x 4 or 4 x 4).

    Returns:
        tot_d (torch.Tensor): Total diattenuation.
        lin_d (torch.Tensor): Linear diattenuation.
        cir_d (torch.Tensor): Circular diattenuation.
        ori_d (torch.Tensor): Orientation of the linear diattenuation axes.
    """

    if MD.shape[-1] == 16: MD = MD.view(*MD.shape[:-1], 4, 4)
    if transpose: MD = MD.transpose(-2, -1)

    # Ensure MD is a complex tensor (for handling potential negative numbers in square root)
    MD = MD.to(torch.complex64) if torch.any(MD < 0) else MD.to(torch.float64)

    # Determine the parameters of total, linear, and circular diattenuation
    tot_d = torch.sqrt((MD[..., 0, 1].real ** 2 + MD[..., 0, 2].real ** 2 + MD[..., 0, 3].real ** 2)).real
    lin_d = torch.sqrt(MD[..., 0, 1].real ** 2 + MD[..., 0, 2].real ** 2).real
    cir_d = torch.abs(MD[..., 0, 3])

    # Determine the orientation of linear diattenuation
    ori_d = 0.5 * torch.atan2(MD[..., 0, 2].real, MD[..., 0, 1].real).real

    return torch.stack([tot_d, lin_d, cir_d, ori_d], dim=1)


def extract_retardance(MR, decomposition_choice='LIN-CIR', tol=1e-9, transpose=True):
    """
    Extraction of polarimetric retardance parameters from a retardance matrix MR.

    Parameters:
    MR - Mueller matrix of retardance (torch tensor of shape [..., 4, 4])
    decomposition_choice - String. If 'LIN-CIR' is chosen, the retardance matrix
    will be decomposed as a product of a linear retarder and a circular retarder.
    If 'CIR-LIN' is chosen, it will be the inverse.

    Returns:
    linear_retarder_matrix - Linear retarder matrix
    circular_retarder_matrix - Circular retarder matrix
    total_retardance - Total phase retardance
    retardance_vector - Phase retardance vector
    linear_retardance - Linear phase retardance
    circular_retardance - Circular phase retardance
    orientation_linear_retardance - Orientation of the linear phase retardance axes (0° to 90°)
    orientation_linear_retardance_full - Orientation of the linear phase retardance axes (0° to 180°)
    """

    chs = MR.shape[-1]
    if chs == 16: MR = MR.view(*MR.shape[:-1], 4, 4)

    if transpose: MR = MR.transpose(-2, -1)

    # Argument calculation
    argument = 0.5 * (MR[..., 1, 1] + MR[..., 2, 2] + MR[..., 3, 3]) - 0.5

    # Determination of total, linear, and circular retardance parameters
    R = torch.where(torch.abs(argument) > 1-tol, 
                    torch.where(argument > tol, torch.acos(torch.tensor(1.0, device=MR.device)), torch.acos(torch.tensor(-1.0, device=MR.device))), 
                    torch.acos(argument.real))
    tot_MR = R / torch.pi * 180
    retardance_normalization_index = 1 / (2 * torch.sin(R))

    # Components of the retardance vector
    a1 = R * retardance_normalization_index * (MR[..., 2, 3] - MR[..., 3, 2])
    a2 = R * retardance_normalization_index * (MR[..., 3, 1] - MR[..., 1, 3])
    a3 = R * retardance_normalization_index * (MR[..., 1, 2] - MR[..., 2, 1])
    retardance_vector = torch.stack([torch.ones_like(a1), a1, a2, a3], dim=-1)

    # Extraction of linear and circular phase retardance (according to the Matlab implementation by )
    #linear_retardance = torch.acos(torch.clip(MR[..., 3, 3].real, -1., 1.)) / torch.pi * 180
    #circular_retardance = torch.atan2((MR[..., 2, 1] - MR[..., 1, 2]).real, (MR[..., 2, 2] + MR[..., 1, 1]).real) / torch.pi * 180

    # rectification?
    circular_retardance = torch.acos(torch.clip(MR[..., 3, 3].real, -1., 1.)) / torch.pi * 180
    
    # Decomposition of the retardance matrix into a linear-circular product
    MRC = rota(circular_retardance / 2)

    # Determination of the linear phase retardance azimuth between 0° and 90°
    circular_retardance_temp = lambda MRL: torch.atan2( (MRL[..., 2, 1] - MRL[..., 1, 2]).real, 
                                                        (MRL[..., 2, 2] + MRL[..., 1, 1]).real) / torch.pi * 180
    if decomposition_choice == 'LIN-CIR':
        MRL = MR @ MRC.transpose(-2, -1)
        cir_temp = circular_retardance_temp(MRL)
        mask = cir_temp.abs() > circular_retardance.abs()
        MRL[mask] = (MR @ MRC)[mask]
    elif decomposition_choice == 'CIR-LIN':
        MRL = MRC.transpose(-2, -1) @ MR
        cir_temp = circular_retardance_temp(MRL)
        mask = cir_temp.abs() > circular_retardance.abs()
        MRL[mask] = (MRC @ MR)[mask]
    MRL[tot_MR<tol] = MR[tot_MR<tol]

    # linear retardance according to HORAO papers
    M_n = (MRL[..., 1, 1] + MRL[..., 2, 2])**2 + (MRL[..., 2, 1] - MRL[..., 1, 2])**2
    linear_retardance = acos_nansafe(M_n**.5 - 1) / torch.pi * 180
    circular_retardance = acos_nansafe(MRL[..., 3, 3].real) / torch.pi * 180

    orientation_linear_retardance = torch.atan2(MRL[..., 1, 3], MRL[..., 3, 2]) / torch.pi * 180
    #orientation_linear_retardance[tot_MR<0] = tol

    mask = orientation_linear_retardance < tol
    orientation_linear_retardance[mask] = 360 - abs(orientation_linear_retardance[mask])
    orientation_linear_retardance = orientation_linear_retardance / 2
    #orientation_linear_retardance = 90 + orientation_linear_retardance

    rimgs = torch.stack([tot_MR, circular_retardance, linear_retardance, orientation_linear_retardance], dim=1)
    rmats = torch.stack([MRL, MRC], dim=1)
    if chs == 16: rmats = rmats.flatten(-2, -1)

    return rmats, rimgs, retardance_vector


def rota(optical_rotation):
    """
    rota - Calculation of the circular retarder matrix from the "optical rotation" parameter
    (circular phase retardance divided by 2)

    === Inputs and Outputs ===
    Inputs:
        optical_rotation - Tensor of shape (N, M) representing angles of "optical rotation"
    Outputs:
        rot: Tensor of shape (N, M, 4, 4) representing circular retarder matrices for each input angle
    """

    optical_rotation_rad = optical_rotation * torch.pi / 180

    cos_vals = torch.cos(2 * optical_rotation_rad)
    sin_vals = torch.sin(2 * optical_rotation_rad)

    rot = torch.zeros(optical_rotation.shape + (4, 4), dtype=optical_rotation.dtype, device=optical_rotation.device)
    rot[..., 0, 0] = 1
    rot[..., 1, 1] = cos_vals
    rot[..., 1, 2] = -sin_vals
    rot[..., 2, 1] = sin_vals
    rot[..., 2, 2] = cos_vals
    rot[..., 3, 3] = 1
    
    return rot


def batched_polarimetry(args):

    MD, MR, Mdelta = args

    datts = extract_diattenuation(MD)
    rmats, rimgs, rvecs = extract_retardance(MR)
    dpols = extract_depolarization(Mdelta)[:, None]
    
    rmats = rmats.moveaxis(-1, 1).flatten(1, 2)
    rvecs = rvecs.moveaxis(-1, 1)

    return torch.cat([datts, rimgs, rmats, rvecs, dpols], dim=1)
