import os
import sys
import pandas as pd
import numpy as np
import logging
import traceback

import requests
import shutil
import time
import ray
import json
import boto3
import gc
import psutil

from ray import serve
from PIL import Image
from pathlib import Path

sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(str(Path(os.getcwd())))

from src.utils.download import LomotifDownloader
from src.utils.data_processing import get_mime
from src.utils.download_models import download_models_helper
from src.tag_and_moderate import ContentTagAndModerate
from src.tag_nudity import NsfwDetect
from src.tag_middle_finger import MiddleFingerDetect
from src.utils.generate_outputs import output_template

# from dotenv import load_dotenv

# load_dotenv("./.env")

# loggers
logger = logging.getLogger("ray")
logger.setLevel("INFO")

nn_use_gpu = False
if float(os.environ["ClipNumGPUPerReplica"]) != 0:
    clip_use_gpu = True
else:
    clip_use_gpu = False


@serve.deployment(
    route_prefix="/download_image",
    max_concurrent_queries=os.environ["DownloadMaxCon"],
    num_replicas=os.environ["DownloadNumReplicas"],
    ray_actor_options={
        "num_cpus": float(os.environ["DownloadNumCPUPerReplica"]),
    },
)
class ImageDownloaderServe:
    def __init__(
        self,
    ):
        """Rayserve module for downloading profile pictures with at most 5 retries \
        and backoff of 30 seconds in between tries.
        """
        try:
            self.downloader = LomotifDownloader(
                save_folder_directory="./downloaded_profile_images"
            )
            self.num_retries = 5
            self.delay_between_retries = 30  # in seconds
            # logging.basicConfig(level=logging.INFO)
        except Exception as e:
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def __call__(self, image_url, user_id, vmem_clear_ref):
        """Rayserve __call__ definition.

        Args:
            image_url (string): URL to the profile image
            user_id (string): uid of profile
            vmem_clear_ref (bool): True to clear cache with garbage collector, False otherwise

        Returns:
            tuple: a 5-element tuple where the first element is a boolean variable that \
            is True if the image has been successfully downloaded else False, second variable \
            is the object type of the file (str), third is filepath to the downloaded image (str), \
            fourth is RayObjectRef of the image array (RayObjectRef), and lastly the number of \
            key frames (int)
        """
        if vmem_clear_ref:
            gc.collect()

        start = time.time()
        img_arr = []
        for retry_number in range(self.num_retries):
            logger.info(
                "[{}] Download retry {}/{}...".format(
                    user_id, retry_number, self.num_retries
                )
            )
            result, save_file_name = self.downloader.download(image_url, user_id)
            mime = get_mime(save_file_name)

            if mime in ["image"]:
                img_arr = [np.array(Image.open(save_file_name))]
                save_file_name_ref = ray.put(img_arr)
            else:
                mime = None
                logger.warning(
                    "[{}] File is not an image. File not processed and defaults to to-be-moderated.".format(
                        user_id
                    )
                )
            if os.path.exists(save_file_name):
                os.remove(save_file_name)
            if result:
                end = time.time()
                logger.info(
                    "[{}] Download complete, filename: {}, duration: {}".format(
                        user_id, save_file_name, end - start
                    )
                )
                break
            else:
                time.sleep(self.delay_between_retries)

        len_key_frames = len(img_arr)
        return result, mime, save_file_name, save_file_name_ref, len_key_frames


@serve.deployment(
    route_prefix="/tagging_with_coopclip",
    max_concurrent_queries=os.environ["ClipMaxCon"],
    num_replicas=os.environ["ClipNumReplicas"],
    ray_actor_options={
        "num_cpus": float(os.environ["ClipNumCPUPerReplica"]),
        "num_gpus": float(os.environ["ClipNumGPUPerReplica"]),
    },
)
class ContentModTagCoopClipServe:
    def __init__(
        self,
    ):
        """Predicts primary and secondary categories of lomotif."""
        try:
            logger.info("Downloading coopclip models from S3...")
            download_models_helper(model_name="coopclip", root="./models")
            logger.info("All coopclip model files downloaded.")
            self.coopclip_model = ContentTagAndModerate(use_gpu=clip_use_gpu)
        except Exception as e:
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def __call__(self, key_frames, save_file_name, kinesis_event, vmem_clear_ref):
        """Rayserve __call__ definition.

        Args:
            key_frames (ObjectRef): ObjectRef of list of arrays that are key frames
            save_file_name (string): filepath to the downloaded image
            kinesis_event (dict): kinesis event payload
            vmem_clear_ref (bool): True to clear cache with garbage collector, False otherwise

        Returns:
            dict: results of the model
        """
        if vmem_clear_ref:
            gc.collect()

        start = time.time()
        self.coopclip_model.reset()
        clip_results = self.coopclip_model.run_service_with_key_frames(
            key_frames_list=key_frames,
            kinesis_event=kinesis_event,
        )
        end = time.time()
        logger.info(
            "[{}] ContentModTagClipServe complete, save_file_name: {}, duration: {}".format(
                kinesis_event["user_id"], save_file_name, end - start
            )
        )
        return clip_results


@serve.deployment(
    route_prefix="/moderation_with_nudenet",
    max_concurrent_queries=os.environ["NudenetMaxCon"],
    num_replicas=os.environ["NudenetNumReplicas"],
    ray_actor_options={
        "num_cpus": float(os.environ["NudenetNumCPUPerReplica"]),
    },
)
class ContentModNudenetServe:
    def __init__(
        self,
    ):
        """Run nudity detection on key frames."""
        try:
            logger.info("Downloading nudenet models from S3...")
            download_models_helper(model_name="nudenet", root="./models")
            logger.info("All nudenet model files downloaded.")
            self.nudenet_model = NsfwDetect(
                model_path="./models/classifier_lite.onnx", use_gpu=nn_use_gpu
            )
        except Exception as e:
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def __call__(self, key_frames, save_file_name, kinesis_event, vmem_clear_ref):
        """Rayserve __call__ definition.

        Args:
            key_frames (ObjectRef): ObjectRef of list of arrays that are key frames
            save_file_name (string): filepath to the downloaded image
            kinesis_event (dict): kinesis event payload
            vmem_clear_ref (bool): True to clear cache with garbage collector, False otherwise

        Returns:
            dict: results of the model
        """
        if vmem_clear_ref:
            gc.collect()

        start = time.time()
        self.nudenet_model.reset()
        nn_results = self.nudenet_model.classify_clip_with_key_frames(
            key_frames=key_frames,
            clip_path=save_file_name,
            kinesis_event=kinesis_event,
        )
        end = time.time()
        logger.info(
            "[{}] ContentModNudenetServe complete, save_file_name: {}, duration: {}".format(
                kinesis_event["user_id"], save_file_name, end - start
            )
        )
        return nn_results


@serve.deployment(
    route_prefix="/moderation_with_middle_finger_detect",
    max_concurrent_queries=os.environ["MFDMaxCon"],
    num_replicas=os.environ["MFDNumReplicas"],
    ray_actor_options={
        "num_cpus": float(os.environ["MFDNumCPUPerReplica"]),
    },
)
class ContentModMiddleFingerDetectServe:
    """Run middle finger detection on key frames."""

    def __init__(
        self,
    ):
        try:
            self.mfd_model = MiddleFingerDetect()
        except Exception as e:
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    def __call__(self, key_frames, save_file_name, kinesis_event, vmem_clear_ref):
        """Rayserve __call__ definition.

        Args:
            key_frames (ObjectRef): ObjectRef of list of arrays that are key frames
            save_file_name (string): filepath to the downloaded image
            kinesis_event (dict): kinesis event payload
            vmem_clear_ref (bool): True to clear cache with garbage collector, False otherwise

        Returns:
            dict: results of the model
        """
        if vmem_clear_ref:
            gc.collect()

        start = time.time()
        self.mfd_model.reset()
        mfd_results = self.mfd_model.classify_clip_with_key_frames(
            key_frames=key_frames,
            clip_path=save_file_name,
            kinesis_event=kinesis_event,
        )
        end = time.time()
        logger.info(
            "[{}] ContentModMiddleFingerDetectServe complete, save_file_name: {}, duration: {}".format(
                kinesis_event["user_id"], save_file_name, end - start
            )
        )
        return mfd_results


@serve.deployment(
    route_prefix="/composed",
    max_concurrent_queries=os.environ["ComposedMaxCon"],
    num_replicas=os.environ["ComposedNumReplicas"],
    ray_actor_options={
        "num_cpus": float(os.environ["ComposedNumCPUPerReplica"]),
    },
)
class ComposedModel:
    """Composition of the whole pipeline starting from downloading \
    profile image to generating all outputs and putting them to SQS.
    """

    def __init__(self):
        try:
            self.download_engine = ImageDownloaderServe.get_handle(sync=False)
            self.model_nudenet = ContentModNudenetServe.get_handle(sync=False)
            self.model_coopclip = ContentModTagCoopClipServe.get_handle(sync=False)
            self.model_mfd = ContentModMiddleFingerDetectServe.get_handle(sync=False)
            self.sqs_client = boto3.client("sqs")
        except Exception as e:
            logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
            assert False  # force quit the script

    async def __call__(self, starlette_request):
        """Rayserve __call__ definition.

        Args:
            starlette_request (bytes): incoming request payload

        Returns:
            dict: outputs that will be written to a snowflake table
        """
        vmem = psutil.virtual_memory().percent
        logger.info(
            "% Virtual memory used: {}, gc items: {}".format(vmem, gc.get_count())
        )
        vmem_clear = vmem > 50.0
        vmem_clear_ref = ray.put(vmem_clear)

        if vmem_clear:
            gc.collect()

        start = time.time()
        kinesis_event = await starlette_request.json()
        kinesis_event_ref = ray.put(kinesis_event)
        message_receive_time = str(pd.Timestamp.utcnow())
        user_id = kinesis_event["user_id"]
        image_url = kinesis_event["event_properties"]["target_user_image_url"]
        logger.info("Message received: {}".format(user_id))

        if len(image_url) > 0:

            output_dict = output_template(
                kinesis_event,
                message_receive_time,
            )

            (
                download_result,
                mime,
                save_file_name,
                save_file_name_ref,
                len_key_frames,
            ) = await (
                await self.download_engine.remote(image_url, user_id, vmem_clear_ref)
            )

            if download_result:
                if mime is not None:
                    if len_key_frames == 0:
                        output_dict["MODEL_ATTRIBUTES"]["NN_STATUS"] = 5
                        output_dict["MODEL_ATTRIBUTES"]["CLIP_STATUS"] = 5
                        output_dict["MODEL_ATTRIBUTES"]["COOP_STATUS"] = 5
                        output_dict["MODEL_ATTRIBUTES"]["MFD_STATUS"] = 5
                        logger.info("[{}] No key frames generated.".format(user_id))
                    else:
                        logger.info(
                            "[{}] Sending requests to all models.".format(user_id)
                        )
                        mfd_results_ref = await self.model_mfd.remote(
                            save_file_name_ref,
                            save_file_name,
                            kinesis_event_ref,
                            vmem_clear_ref,
                        )
                        nn_results_ref = await self.model_nudenet.remote(
                            save_file_name_ref,
                            save_file_name,
                            kinesis_event_ref,
                            vmem_clear_ref,
                        )
                        clip_results_ref = await self.model_coopclip.remote(
                            save_file_name_ref,
                            save_file_name,
                            kinesis_event_ref,
                            vmem_clear_ref,
                        )
                        mfd_results = await (mfd_results_ref)
                        nn_results = await (nn_results_ref)
                        clip_results = await (clip_results_ref)
                        logger.info("[{}] Getting MFD results.".format(user_id))
                        logger.info("[{}] Getting Nudenet results.".format(user_id))
                        logger.info("[{}] Getting CLIP results.".format(user_id))

                        logger.info("[{}] Aggregating results...".format(user_id))
                        for k, v in mfd_results.items():
                            output_dict["MODEL_ATTRIBUTES"][k] = v
                        for k, v in nn_results.items():
                            output_dict["MODEL_ATTRIBUTES"][k] = v
                        for k, v in clip_results.items():
                            if k in [
                                "PREDICTED_PRIMARY_CATEGORY",
                                "PREDICTED_SECONDARY_CATEGORY",
                            ]:
                                output_dict[k] = v
                            else:
                                output_dict["MODEL_ATTRIBUTES"][k] = v
                        logger.info("[{}] Results aggregated.".format(user_id))
                else:
                    output_dict["MODEL_ATTRIBUTES"]["NN_STATUS"] = 1
                    output_dict["MODEL_ATTRIBUTES"]["CLIP_STATUS"] = 1
                    output_dict["MODEL_ATTRIBUTES"]["COOP_STATUS"] = 1
                    output_dict["MODEL_ATTRIBUTES"]["MFD_STATUS"] = 1
                    logger.info("[{}] Mime is None.".format(user_id))

                del save_file_name_ref

            else:
                output_dict["MODEL_ATTRIBUTES"]["NN_STATUS"] = 403
                output_dict["MODEL_ATTRIBUTES"]["CLIP_STATUS"] = 403
                output_dict["MODEL_ATTRIBUTES"]["COOP_STATUS"] = 403
                output_dict["MODEL_ATTRIBUTES"]["MFD_STATUS"] = 403
                logger.info(
                    "[{}] Image file does not exist or download has failed.".format(
                        user_id
                    )
                )

            del vmem_clear_ref
            del kinesis_event_ref

            if (
                output_dict["MODEL_ATTRIBUTES"]["NN_TO_BE_MODERATED"]
                or output_dict["MODEL_ATTRIBUTES"]["CLIP_TO_BE_MODERATED"]
                or output_dict["MODEL_ATTRIBUTES"]["MFD_TO_BE_MODERATED"]
            ):
                output_dict["TO_BE_MODERATED"] = True
            else:
                output_dict["TO_BE_MODERATED"] = False

            total_duration = round(time.time() - start, 3)
            output_dict["TOTAL_DURATION"] = total_duration
            logger.info("[{}] {}".format(user_id, output_dict))

            try:
                # Send message to SQS queue
                logger.info(
                    "[{}] Attempting to send output to SQS: {}.".format(
                        user_id, os.environ["SnowflakeResultsQueue"]
                    )
                )
                msg = json.dumps(output_dict)
                response = self.sqs_client.send_message(
                    QueueUrl=os.environ["SnowflakeResultsQueue"],
                    DelaySeconds=0,
                    MessageBody=msg,
                )
                logger.info(
                    "[{}] Sent outputs to SQS: {}.".format(
                        user_id, os.environ["SnowflakeResultsQueue"]
                    )
                )
                logger.info(
                    "[{}] ComposedModel complete, save_file_name: {}, duration: {}".format(
                        user_id, save_file_name, total_duration
                    )
                )
                return output_dict

            except Exception as e:
                vmem_clear_ref = None
                kinesis_event_ref = None
                save_file_name_ref = None
                del save_file_name_ref
                del vmem_clear_ref
                del kinesis_event_ref
                logger.error(e, "\n Traceback: \n{}".format(traceback.format_exc()))
                assert False  # force quit the script
        else:
            logger.warning("[{}] Image url length 0.".format(user_id))
            return {}


if __name__ == "__main__":
    env_vars = {
        "AWS_ROLE_ARN": os.environ.get("AWS_ROLE_ARN"),
        "AWS_WEB_IDENTITY_TOKEN_FILE": os.environ.get("AWS_WEB_IDENTITY_TOKEN_FILE"),
        "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION"),
        "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY"),
        "AiModelBucket": os.environ.get("AiModelBucket"),
        "SnowflakeResultsQueue": os.environ["SnowflakeResultsQueue"],
    }
    runtime_env = {"env_vars": {}}

    for key, value in env_vars.items():
        if value is not None:
            runtime_env["env_vars"][key] = value

    ray.init(address="auto", namespace="serve", runtime_env=runtime_env)
    serve.start(detached=True, http_options={"host": "0.0.0.0"})

    logger.info("The environment variables in rayserve are: {}".format(runtime_env))
    logger.info("All variables are: {}".format(env_vars))

    logger.info("Starting rayserve server.")
    logger.info("Deploying modules.")

    ImageDownloaderServe.deploy()
    ContentModNudenetServe.deploy()
    ContentModMiddleFingerDetectServe.deploy()
    ContentModTagCoopClipServe.deploy()
    ComposedModel.deploy()

    logger.info("Deployment completed.")
    logger.info("Waiting for requests...")
