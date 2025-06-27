import pandas as pd

def generate_description(contrast_element, image_ofiq_scores):
    """
    Given a single contrast element, return a description and the actionable feedback for in-context learning.
    """

    # Helper to safely get a scalar value
    def get_scalar(score_name):
        return image_ofiq_scores.get(score_name)

    description = []
    actionable_feedback = []

    # Handle None, NaN, and string 'none'
    if contrast_element == "ONOT":
        description.append("This image is excelent and fully complaint for biometric use. The face is perfectly centered and optimally cropped. No changes needed.")
    else:
        description.append("The image is not compliant for biometric use.") 

        if contrast_element == "bkg":
            score = get_scalar("BackgroundUniformity.scalar")
            description.append(f"The background is not uniform (BackroundUniformity.scalar: {score}). This low score indicates inconsistencies or distracting elements in the background.")
            actionable_feedback.append("Please ensure a clean, plain, and consistent background and retake a photo.")

        elif contrast_element == "cap":
            no_head_coverings_score = get_scalar("NoHeadCoverings.scalar")
            description.append(f"The subject is wearing a head covering (NoHeadCoverings.scalar: {no_head_coverings_score}), which is non-compliant.")
            actionable_feedback.append("Please remove head covering from the subject and retake a photo.")

        elif contrast_element == "ce":
            score = get_scalar("EyesOpen.scalar")
            description.append(f"The subject has closed or partially closed eyes (EyesOpen.scalar: {score}). This low score prevents proper detection of eye features crucial for recognition.")
            actionable_feedback.append("Please ensure the subject's eyes are clearly open and visible and retake a photo.")
        
        elif contrast_element == "ceg":
            score = get_scalar("EyesOpen.scalar")
            description.append(f"The subject has closed or partially closed eyes and it wearing glasses (EyesOpen.scalar: {score}). This low score prevents proper detection of eye features crucial for recognition.")
            actionable_feedback.append("Please ensure the subject's eyes are clearly open and visible, without any occlusions and retake a photo.")

        elif contrast_element == "expos":
            luminance_mean_score = get_scalar("LuminanceMean.scalar")
            luminance_variance_score = get_scalar("LuminanceVariance.scalar")
            under_exposure_score = get_scalar("UnderExposurePrevention.scalar")
            # over_exposure_score = get_scalar("OverExposurePrevention.scalar") Was removed as it didn't show correlation with the dataset
            dynamic_range_score = get_scalar("DynamicRange.scalar")
            description.append(f"There are exposure problems: the image is too dark (UnderExposurePrevention.scalar: {under_exposure_score}) or too bright (LuminanceMean.scalar: {luminance_mean_score}) and (LuminanceVariance.scalar) . It can lead to a loss of detail in highlights or shadows (DynamicRange.scalar: {dynamic_range_score}).")
            actionable_feedback.append("Please ensure proper exposure settings in your camera and retake a photo.")

        elif contrast_element in ["la_1", "la_2"]:
            # QFIQ assessment does not incclude frontal gaze evaluation, so no scalar values will be used for this
            direction_text = "away to the right" if contrast_element == "la_1" else "away to the left"
            description.append(f"The subject is looking {direction_text}. This deviation from a straight gaze is non-compliant.")
            actionable_feedback.append("Please change the subject to be looking straight ahead at the camera with a frontal head pose and retake a photo.")

        elif contrast_element == "light":
            illumination_score = get_scalar("IlluminationUniformity.scalar")
            over_exposure_score = get_scalar("OverExposurePrevention.scalar")
            description.append(f"The face illumination is not uniform (IlluminationUniformity.scalar: {illumination_score}) (OverExposurePrevention: {over_exposure_score}). This low score indicates uneven lighting, causing harsh shadows or bright spots that obscure facial features.")
            actionable_feedback.append("Please use uniformed light and retake the photo.")

        elif contrast_element == "mkup":
            # Not all the TONO data captures makeup, but if it does, we can use the FaceOcclusionPrevention scalar
            occlusion_score = get_scalar("FaceOcclusionPrevention.scalar") # Heavy makeup can cause occlusion
            description.append(f"The subject is wearing makeup, which may obscure facial features (FaceOcclusionPrevention.scalar: {occlusion_score}). This can impact biometric matching.")
            actionable_feedback.append("Please remove makeup from the subject and retake a photo.")

        elif contrast_element == "oof":
            score = get_scalar("Sharpness.scalar")
            description.append(f"The image is not sharp (score: {score}), causing blurriness across facial features.")
            actionable_feedback.append("Please ensure proper focus on your camera and retake a photo.")

        elif contrast_element == "pixel":
            # No specific scalar for pixelation
            description.append(f"The image exhibits pixelation or low resolution artifacts. This impacts overall clarity and detail.")
            actionable_feedback.append("Please ensure high resolution and minimal compression and retake the photo.")

        elif contrast_element == "poster":
            dynamic_range_score = get_scalar("DynamicRange.scalar")
            description.append(f"The image has posterization, indicating a loss of color depth and smooth tonal gradients . This can be caused by insufficient dynamic range capture (DynamicRange.scalar: {dynamic_range_score}).")
            actionable_feedback.append("Please ensure sufficient color depth and dynamic range in the camera settings and retake a photo.")

        elif contrast_element == "sat":
            natural_colour_score = get_scalar("NaturalColour.scalar")
            description.append(f"The image colors are unnatural (NaturalColour.scalar: {natural_colour_score}, appearing either too vivid or washed out.")
            actionable_feedback.append("Please ensure balanced color saturation and natural color reproduction in your camera settings and retake the photo.")

        elif contrast_element == "sm":
            expression_score = get_scalar("ExpressionNeutrality.scalar")
            mouth_closed_score = get_scalar("MouthClosed.scalar")
            description.append(f"The subject does not have a neutral facial expression (ExpressionNeutrality.scalar: {expression_score}), (MouthClosed.scalar: {mouth_closed_score}).")
            actionable_feedback.append("Please ensure the subject maintains a neutral expression with a closed mouth and retake a photo.")

        elif contrast_element == "sun":
            occlusion_score = get_scalar("FaceOcclusionPrevention.scalar")
            eyes_visible_score = get_scalar("EyesVisible.scalar")
            description.append(f"The subject's face is occluded by sunglasses (FaceOcclusionPrevention.scalar: {occlusion_score}), specifically obscuring the eyes (EyesVisible.scalar: {eyes_visible_score}).")
            actionable_feedback.append("Please remove sunglasses from the subject and retake a photo.")

        elif contrast_element == "tq":
            yaw_score = get_scalar("HeadPoseYaw.scalar")
            pitch_score = get_scalar("HeadPosePitch.scalar")
            roll_score = get_scalar("HeadPoseRoll.scalar")
            description.append(f"The subject's head pose is incorrect, with deviation from the center (HeadPoseYaw.scalar: {yaw_score}, HeadPosePitch.scalar: {pitch_score}, HeadPoseRoll.scalar: {roll_score}).")
            actionable_feedback.append("Please change the pose of the subject to be facing the camera directly and retake the photo.")

        elif contrast_element == "zoom":
            head_size_score = get_scalar("HeadSize.scalar")
            intereye_distance_score = get_scalar("InterEyesDistance.scalar")
            above_margin_score = get_scalar("MarginAboveOfTheFaceImage.scalar")
            below_margin_score = get_scalar("MarginBelowOfTheFaceImage.scalar")
            sharpness_score = get_scalar("Sharpness.scalar") # Can be low if zoomed in too much

            description.append(f"The image is improperly zoomed, with the head being too close to the camera appearing big (HeadSize.scalar: {head_size_score},InterEyeDistance.scalar: {intereye_distance_score} ), and therefore loosing on sharpness (Sharpness.scalar: {sharpness_score}). It causes too small margins above (MarginAboveOfTheFaceImage.scalar: {above_margin_score}) and  below the face image (MarginBelowOfTheFaceImage.scalar: {below_margin_score}).  ")
            actionable_feedback.append("Please zoom out to ensure the subject's face is properly framed with adequate margins and retake a photo.")
        else:
            description.append(f"Unknown contrast element: {contrast_element}. Please review the input data.")

    full_desc = " ".join(description)

    if actionable_feedback:
        full_desc += " Actionable Feedback: " + ", ".join(actionable_feedback)
    
    return full_desc.strip()