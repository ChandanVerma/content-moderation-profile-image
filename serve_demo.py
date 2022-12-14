import os
import sys
import requests
import logging
import json
import pickle
import traceback
import tracemalloc

tracemalloc.start()

from pathlib import Path
from pandas import Timestamp

sys.path.append(str(Path(os.getcwd()).parent))
sys.path.append(str(Path(os.getcwd())))

# loggers
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ray")
logger.setLevel("INFO")

if __name__ == "__main__":
    kinesis_event = {
        "app_version_id": "2.17.0",
        "country": "co",
        "device_id": "32462456-a204-e251-d25e-2096652e993e",
        "event_properties": {
            "action_time": "2022-06-10 01:21:01",
            "app_version_id": "2.17.0",
            "target_user_caption": "",
            "target_user_image_url": "https://cdn.lomotif.com/user/profile/7c754bea1aa6672b/5e459c67de6fec5d.png",
            "target_user_followers": 0,
            "target_user_following": 0,
            "target_user_country": None,
            "target_user_date_joined": "2022-06-10 01:20:32",
            "target_user_gender": "unknown",
            "target_user_id": 37217839,
            "target_user_username": "danysantynener07",
            "target_user_name": "danysantynener07",
            "target_user_locale": "en",
            "uid": 37217839,
        },
        "event_time": "2022-06-10 01:21:01",
        "os_name": "android",
        "os_version": "11",
        "platform": "android",
        "session_id": None,
        "user_id": "37217839",
        "process_time": "2022-06-10 01:21:01.521300",
        "hour": "1",
    }
    # url = "https://lomotif-prod.s3.amazonaws.com/user/profile/06e784fb46fe7487/3257bf71682d8567.png"
    try:

        resp = requests.get(
            "http://0.0.0.0:8000/composed", json=kinesis_event, timeout=5
        )
        if resp.status_code == 200:
            output = resp.json()
            logger.info("Rayserve tasks successful. Output: {}".format(output))

        else:
            logger.error(
                "Error in rayserve tasks. Status code: {} \nTraceback: {}".format(
                    resp.status_code, resp.text
                )
            )
    except:
        assert False, logger.error(
            "[{}] Lomotif could not be processed due to: {}. \nTraceback: {}".format(
                kinesis_event["user_id"], traceback.format_exc()
            )
        )
