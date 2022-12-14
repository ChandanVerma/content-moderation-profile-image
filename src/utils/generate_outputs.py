def output_template(kinesis_event, message_receive_time):
    """Generates an output dictionary to be sent to SQS and \
    eventually a Snowflake table.

    Args:
        kinesis_event (dict): event payload
        message_receive_time (string): UTC time when the message is \
        received by the endpoint.

    Returns:
        dict: template to store outputs
    """

    # general outputs
    output_dict = {}
    output_dict["UID"] = kinesis_event["user_id"]
    output_dict["USERNAME"] = kinesis_event["event_properties"]['target_user_name']
    output_dict["IMAGE_URL"] = kinesis_event["event_properties"]['target_user_image_url']
    output_dict["COUNTRY"] = kinesis_event["country"]
    output_dict["EVENT_TIME"] = kinesis_event["event_time"]
    output_dict["MESSAGE_RECEIVE_TIME"] = message_receive_time
    output_dict["TOTAL_DURATION"] = 0

    # nudenet outputs
    output_dict["MODEL_ATTRIBUTES"] = {}
    output_dict["MODEL_ATTRIBUTES"]["NN_DURATION"] = 0
    output_dict["MODEL_ATTRIBUTES"]["NN_SAFE_SCORES"] = ""
    output_dict["MODEL_ATTRIBUTES"]["NN_UNSAFE_SCORES"] = ""
    output_dict["MODEL_ATTRIBUTES"]["NN_TO_BE_MODERATED"] = True
    output_dict["MODEL_ATTRIBUTES"]["NN_PREDICTION_SUCCESS"] = False
    output_dict["MODEL_ATTRIBUTES"]["NN_STATUS"] = -1

    # clip outputs
    output_dict["MODEL_ATTRIBUTES"]["CLIP_DURATION"] = 0
    output_dict["MODEL_ATTRIBUTES"]["CLIP_PREDICTION_SUCCESS"] = False
    output_dict["MODEL_ATTRIBUTES"]["CLIP_TO_BE_MODERATED"] = True
    output_dict["MODEL_ATTRIBUTES"]["CLIP_STATUS"] = -1

    # coop outputs
    output_dict["MODEL_ATTRIBUTES"]["COOP_DURATION"] = 0
    output_dict["MODEL_ATTRIBUTES"]["COOP_PREDICTION_SUCCESS"] = False
    output_dict["MODEL_ATTRIBUTES"]["COOP_STATUS"] = -1

    # coop-clip outputs
    output_dict["PREDICTED_PRIMARY_CATEGORY"] = ""
    output_dict["PREDICTED_SECONDARY_CATEGORY"] = ""
    output_dict["MODEL_ATTRIBUTES"]["PREDICTED_TOP3_PRIMARY_CATEGORY"] = ""
    output_dict["MODEL_ATTRIBUTES"]["PREDICTED_TOP3_SECONDARY_CATEGORY"] = ""

    # middle finger outputs
    output_dict["MODEL_ATTRIBUTES"]["MFD_DURATION"] = 0
    output_dict["MODEL_ATTRIBUTES"]["MFD_TO_BE_MODERATED"] = True
    output_dict["MODEL_ATTRIBUTES"]["MFD_PREDICTION_SUCCESS"] = False
    output_dict["MODEL_ATTRIBUTES"]["MFD_STATUS"] = -1

    # other attributes
    output_dict["MODEL_VERSION"] = "1.1.0"

    return output_dict
