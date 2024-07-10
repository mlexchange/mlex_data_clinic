import glob

import pyFAI.detectors as detectors


def get_mask_options():
    """
    This function gets the mask options
    Returns:
        mask_options:       List of mask options
    """
    # Get the mask files
    mask_files = glob.glob("assets/masks/*.tif")
    mask_options = []
    for mask_file in mask_files:
        mask_options.append({"label": mask_file.split("/")[-1], "value": mask_file})

    # Get the pyFAI detector masks
    pyfai_detectors = detectors.ALL_DETECTORS.keys()
    mask_options = [
        {"label": detector, "value": detector} for detector in pyfai_detectors
    ]
    return mask_options
