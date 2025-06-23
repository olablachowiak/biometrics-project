import pandas as pd

def generate_description(contrast_element):
    """
    Given a single contrast element, return a description for in-context learning.
    """

    # Handle None, NaN, and string 'none'
    if contrast_element is None or pd.isna(contrast_element) or str(contrast_element).lower() == "none":
        return "The image is compliant for biometric use. No defects detected."
    
    descriptions = {
        "bkg": "Background is not uniform. Please ensure a clean and consistent background.",
        "cap": "Subject is wearing a head covering such as a hat, cap, bandana, bowler, or tifler. Please ensure the subject's head is uncovered.",
        "ce": "Subject has closed eyes. Please ensure the subject's eyes are open.",
        "expos": "Exposure problems such as underexposure or overexposure. Please ensure proper exposure settings.",
        "la_1": "Subject is looking away to the right. Please ensure the subject is looking straight ahead.",
        "la_2": "Subject is looking away to the left. Please ensure the subject is looking straight ahead.",
        "light": "No uniform face light. Please ensure even illumination across the face.",
        "mkup": "Subject is wearing makeup, which may obscure facial features. Please ensure the subject is wearing minimal makeup.",
        "oof": "Image is out of focus, causing blurriness. Please ensure the sharpness of the image.",
        "pixel": "Pixelation or low resolution affecting image quality. Please ensure high resolution and clarity.",
        "poster": "Image has posterization, which is a loss of color depth caused by insufficient dynamic range. Please ensure sufficient color depth.",
        "sat": "Image has unnatural colors, making them too vivid or washed out. Please ensure balanced color saturation.",
        "sm": "Subject does not have a neutral expression. Please ensure the subject has a neutral facial expression.",
        "sun": "Subject's face is being occluded by sunglasses. Please ensure the subject is not wearing sunglasses and eyes are fully visible.",
        "tq": "Subject's head is rolled, pitched, or yawed. Please ensure the subject is facing the capturing device directly.",
        "zoom": "Subject's face is being cropped or zoomed in too much. Please ensure the subject's face is fully visible in the frame and the image has correct margins."
    }
    desc = descriptions.get(contrast_element, "Unknown or unclassified image problem.")
    return "The image is not compliant for biometric use. The primary defect is: "+ desc
