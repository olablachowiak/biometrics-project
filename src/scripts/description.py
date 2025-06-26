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
    if contrast_element is None or pd.isna(contrast_element) or str(contrast_element).lower() == "none":
        description.append("This image is excelent and fully complaint for biometric use. The face is perfectly centered and optimally cropped. No changes needed.")
    else:
        description.append("The image is not compliant for biometric use. Please address the following issues:") 

        if contrast_element == "bkg":
            score = get_scalar("BackgroundUniformity.scalar")
            description.append(f"The background is not uniform (score: {score}). This low score indicates inconsistencies or distracting elements in the background.")
            actionable_feedback.append("Please ensure a clean, plain, and consistent background.")

        elif contrast_element == "cap":
            no_head_coverings_score = get_scalar("NoHeadCoverings.scalar")
            description.append(f"The subject is wearing a head covering (NoHeadCoverings.scalar: {no_head_coverings_score}), which is non-compliant.")
            actionable_feedback.append("Please ensure the subject's head is uncovered, with the entire face clearly visible.")

        elif contrast_element == "ce":
            score = get_scalar("EyesOpen.scalar")
            description.append(f"The subject has closed or partially closed eyes (score: {score}). This low score prevents proper detection of eye features crucial for recognition.")
            actionable_feedback.append("Please ensure the subject's eyes are clearly open and visible.")
        
        elif contrast_element == "ceg":
            score = get_scalar("EyesOpen.scalar")
            description.append(f"The subject has closed eyes with glasses (EyesOpen.scalar: {score}). This prevents proper detection of eye features crucial for recognition.")
            actionable_feedback.append("Please ensure the subject's eyes are clearly open and visible, without any occlusions.")

        elif contrast_element == "expos":
            luminance_mean_score = get_scalar("LuminanceMean.scalar")
            under_exposure_score = get_scalar("UnderExposurePrevention.scalar")
            over_exposure_score = get_scalar("OverExposurePrevention.scalar")
            dynamic_range_score = get_scalar("DynamicRange.scalar")
            description.append(f"There are exposure problems: the image might be too dark (UnderExposurePrevention.scalar: {under_exposure_score}) or too bright (OverExposurePrevention.scalar: {over_exposure_score}). The overall average brightness (LuminanceMean.scalar: {luminance_mean_score}) might be off leading to a loss of detail in highlights or shadows (DynamicRange.scalar: {dynamic_range_score}).")
            actionable_feedback.append("Please ensure proper exposure settings for balanced brightness and detail preservation.")

        elif contrast_element in ["la_1", "la_2"]:
            # QFIQ assessment does not incclude frontal gaze evaluation, so no scalar values will be used for this
            direction_text = "away to the right" if contrast_element == "la_1" else "away to the left"
            description.append(f"The subject is looking {direction_text}. This deviation from a straight gaze is non-compliant.")
            actionable_feedback.append("Please ensure the subject is looking straight ahead at the camera with a frontal head pose.")

        elif contrast_element == "light":
            score = get_scalar("IlluminationUniformity.scalar")
            dynamic_range_score = get_scalar("DynamicRange.scalar")
            description.append(f"The face illumination is not uniform (score: {score}). This low score indicates uneven lighting, causing harsh shadows or bright spots that obscure facial features. The DynamicRange.scalar score of {dynamic_range_score} suggests insufficient dynamic range, which can further affect the visibility of details.")
            actionable_feedback.append("Please ensure even and consistent illumination across the entire face.")

        elif contrast_element == "mkup":
            # Not all the TONO data captures makeup, but if it does, we can use the FaceOcclusionPrevention scalar
            occlusion_score = get_scalar("FaceOcclusionPrevention.scalar") # Heavy makeup can cause occlusion
            description.append(f"The subject is wearing makeup, which may obscure facial features (FaceOcclusionPrevention.scalar: {occlusion_score}). This can impact biometric matching.")
            actionable_feedback.append("Please ensure the subject is wearing minimal or no makeup for biometric capture.")

        elif contrast_element == "oof":
            score = get_scalar("Sharpness.scalar")
            description.append(f"The image is not sharp (score: {score}), causing blurriness across facial features. This severely impacts recognition accuracy.")
            actionable_feedback.append("Please ensure proper focus to achieve high sharpness in the image.")

        elif contrast_element == "pixel":
            # No specific scalar for pixelation
            description.append(f"The image exhibits pixelation or low resolution artifacts. This impacts overall clarity and detail.")
            actionable_feedback.append("Please ensure high resolution and minimal compression to avoid these distortions.")

        elif contrast_element == "poster":
            # No specific scalar for posterization
            description.append(f"The image has posterization, indicating a loss of color depth and smooth tonal gradients. This can be caused by insufficient dynamic range capture.")
            actionable_feedback.append("Please ensure sufficient color depth and dynamic range in the capture settings.")

        elif contrast_element == "sat":
            natural_colour_score = get_scalar("NaturalColour.scalar")
            description.append(f"The image colors are unnatural (NaturalColour.scalar: {natural_colour_score}, appearing either too vivid or washed out.")
            actionable_feedback.append("Please ensure balanced color saturation and natural color reproduction.")

        elif contrast_element == "sm":
            expression_score = get_scalar("ExpressionNeutrality.scalar")
            mouth_closed_score = get_scalar("MouthClosed.scalar")
            description.append(f"The subject does not have a neutral facial expression (ExpressionNeutrality.scalar: {expression_score}), (MouthClosed.scalar: {mouth_closed_score}). Expressions like smiling, frowning, or grimacing alter facial landmarks.")
            actionable_feedback.append("Please ensure the subject maintains a neutral expression with a closed mouth.")

        elif contrast_element == "sun":
            occlusion_score = get_scalar("FaceOcclusionPrevention.scalar")
            eyes_visible_score = get_scalar("EyesVisible.scalar")
            description.append(f"The subject's face is occluded by sunglasses (Occlusion.scalar: {occlusion_score}), specifically obscuring the eyes (EyesVisible.scalar: {eyes_visible_score}). This prevents biometric analysis of key features.")
            actionable_feedback.append("Please ensure the subject is not wearing sunglasses and eyes are fully visible.")

        elif contrast_element == "tq":
            inter_eye_distance = ("InterEyeDistance.scalar")
            yaw_score = get_scalar("HeadPoseYaw.scalar")
            pitch_score = get_scalar("HeadPosePitch.scalar")
            roll_score = get_scalar("HeadPoseRoll.scalar")
            description.append(f"The subject's head pose is incorrect, with significant deviation (HeadPoseYaw.scalar: {yaw_score}, HeadPosePitch.scalar: {pitch_score}, HeadPoseRoll.scalar: {roll_score}, InterEyeDistance.scalar: {inter_eye_distance}). This non-frontal pose is non-compliant.")
            actionable_feedback.append("Please ensure the subject is facing the capturing device directly and is upright.")

        elif contrast_element == "zoom":
            head_size_score = get_scalar("HeadSize.scalar")
            left_crop_score = get_scalar("LeftwardCropOfTheFaceImage.scalar")
            right_crop_score = get_scalar("RightwardCropOfTheFaceImage.scalar")
            above_margin_score = get_scalar("MarginAboveOfTheFaceImage.scalar")
            sharpness_score = get_scalar("Sharpness.scalar") # Can be low if zoomed in too much

            # Handle HeadSize and Crop based on optimal range/extreme values
            if head_size_score > 75: # Example threshold for "too large"
                description.append(f"The subject's head size is too large (score: {head_size_score}), indicating the image is excessively zoomed in. This often results in the face being cropped too tightly.")
                actionable_feedback.append("Please ensure the subject's entire head and shoulders are visible within the frame, with appropriate margins.")
            else: # Head size is within an acceptable range, now check crops/margins
                description.append(f"The head size is complaint (HeadSize.scalar: {head_size_score}).")
                crop_issues = []
                if left_crop_score < 90: # Example threshold for insufficient margin
                    crop_issues.append(f"insufficient left margin (LeftwardCropOfTheFaceImage.scalar: {left_crop_score})")
                if right_crop_score < 90:
                    crop_issues.append(f"insufficient right margin (RightwardCropOfTheFaceImage.scalar: {right_crop_score})")
                if above_margin_score < 1:
                    crop_issues.append(f"insufficient top margin (MarginAboveOfTheFaceImage.scalar: {above_margin_score})")

                if crop_issues:
                    description.append("However, the image is poorly cropped due to " + ", ".join(crop_issues) + ".")
                    actionable_feedback.append("Please ensure sufficient margins around the face to avoid cutting off parts of the head or face.")
                else:
                    description.append("The framing is excellent, with optimal head size and margins.")

    full_desc = " ".join(description)

    if actionable_feedback:
        full_desc += " Actionable Feedback: " + ", ".join(actionable_feedback)
    
    return full_desc.strip()