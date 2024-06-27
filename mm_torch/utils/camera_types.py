def default_cam_type(CamID=None):
    '''# Define the Default Camera Type used in the Polarimetric Acquisitions
	# Default:  'Stingray IPM2'
	#
	# Call: CamType = default_CamType([CamID])
	#
	# *Inputs*
	# [CamID]: optional scalar integer identifying the Camera device as listed below
	#
	# *Outputs*
	# CamType: string with the camera device identifier
	#
	# Possible Options for different input CamID:
	#
	# CamID		CamType
	# 0 	-> 	'Stingray IPM2' (default)
	# 1 	-> 	'Prosilica'
	# 2 	-> 	'JAI'
	# 3 	-> 	'JAI Packing 2x2'
	# 4 	-> 	'JAI Binning'
	# 5 	-> 	'Stingray'
	# 6 	-> 	'Stingray IPM1'
	#
	# -1    ->  'TEST' -- used for (Unit)Testing
	'''

    CamType = None

    if (CamID == None):
        CamID = 0

    if (CamID == -1):
        CamType = 'TEST'
    if (CamID == 0):
        CamType = 'Stingray IPM2'
    if (CamID == 1):
        CamType = 'Prosilica'
    if (CamID == 2):
        CamType = 'JAI'
    if (CamID == 3):
        CamType = 'JAI Packing 2x2'
    if (CamID == 4):
        CamType = 'JAI Binning'
    if (CamID == 5):
        CamType = 'Stingray'
    if (CamID == 6):
        CamType = 'Stingray IPM1'
    if (int(CamID) < -1 | int(CamID) > 6):
        CamType = 'Stingray IPM2'  # Default

    return CamType


def get_cam_params(CamType):
    '''# Function to retrieve the polarimetric Camera Parameters
	# Several camera types are listed below
	#
	# Call: (ImgShape2D,GammaDynamic) = get_Cam_Params(CamType)
	#
	# *Inputs*
	# CamType: string identifying the camera type (see: default_CamType()).
	#
	# * Outputs *
	# ImgShape2D: list containing the camera pixel-wise size as [dim[0],dim[1]]
	# GammaDynamic: scalar intensity as maximum value for detecting saturation/reflection'''

    ImgShape2D = None
    GammaDynamic = None

    if (CamType == 'Prosilica'):
        GammaDynamic = 16384
        ImgShape2D = [600, 800]

    if (CamType == 'JAI'):
        GammaDynamic = 16384
        ImgShape2D = [768, 1024]

    if (CamType == 'JAI Packing 2x2'):
        GammaDynamic = 16384
        ImgShape2D = [384, 512]

    if (CamType == 'JAI Binning'):
        GammaDynamic = 16384
        ImgShape2D = [384, 1024]

    if (CamType == 'Stingray'):
        GammaDynamic = 65530
        ImgShape2D = [600, 800]

    if (CamType == 'Stingray IPM1'):
        GammaDynamic = 65530
        ImgShape2D = [388, 516]

    if (CamType == 'Stingray IPM2'):
        GammaDynamic = 65530
        ImgShape2D = [388, 516]

    if (CamType == 'TEST'):
        GammaDynamic = 65530
        ImgShape2D = [128, 128]

    return ImgShape2D, GammaDynamic
