"""
Ref: https://github.com/PySeez/middle-finger-detection
"""
import os
import sys
import pandas as pd
import time
import logging

from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(str(Path(os.getcwd())))
from src.utils.finger_detector import FingerDetector


class MiddleFingerDetect:
    def __init__(
        self,
    ):
        self.logger = logging.getLogger("ray")
        try:
            self.logger.setLevel("INFO")
            self.logger.info("Initializing middle finger detection model.")
            self.detector = FingerDetector()
            self.logger.info("Middle finger detection model loaded.")
        except Exception as e:
            self.logger.error(
                "Error initializing middle finger detection model.",
                e,
                "\n Traceback: \n{}".format(traceback.format_exc()),
            )
            assert False  # force quit the script

    def reset(self):
        pass

    def classify_clip_with_key_frames(self, key_frames, clip_path, kinesis_event):
        """Run middle finger detect on key frames.

        Args:
            key_frames (list): list of np.ndarray which are key frames
            clip_path (string): file path to downloaded lomotif
            kinesis_event (dict): event payload

        Returns:
            dict: output from middle finger detection model
        """
        user_id = kinesis_event["user_id"]

        output_dict = {}
        output_dict["USER_ID"] = user_id
        start_time = time.time()

        try:
            self.reset()
            self.logger.debug(
                "[{}] Moderating clip for middle finger gestures.".format(user_id)
            )

            to_be_moderated = False
            results = {
                i: x[0][1].name
                for i, x in enumerate(
                    [self.detector.detect(image) for image in key_frames]
                )
                if len(x) != 0
            }
            if "Mid" in results.values():
                to_be_moderated = True

            self.logger.debug(
                "[{}] Completed moderation for middle finger gestures.".format(user_id)
            )

            output_dict["MFD_DURATION"] = round(time.time() - start_time, 3)
            output_dict["MFD_TO_BE_MODERATED"] = to_be_moderated
            output_dict["MFD_PREDICTION_SUCCESS"] = True
            output_dict["MFD_STATUS"] = 0

        except Exception as e:
            output_dict["MFD_DURATION"] = 0
            output_dict["MFD_TO_BE_MODERATED"] = True
            output_dict["MFD_PREDICTION_SUCCESS"] = False
            output_dict["MFD_STATUS"] = 4
            self.logger.error(
                "[{}] Profile could not be processed due to: {}. \nTraceback: {}".format(
                    user_id, str(e), traceback.format_exc()
                )
            )

        return output_dict
