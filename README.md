# Content Moderation for Profile Images
This repo does content moderation for profile images/ image data in general. If any questions, please reach out to Data Science team (Sze Chi, Thulasiram, Chandan).

# Set-up .env file for testing in local
There needs to be a `.env` file with following parameters.
```
DownloadNumCPUPerReplica=0.2
DownloadNumReplicas=1
DownloadMaxCon=100


NudenetNumCPUPerReplica=0.8
NudenetNumReplicas=1
NudenetMaxCon=100

MFDNumCPUPerReplica=0.8
MFDNumReplicas=1
MFDMaxCon=100

ClipNumCPUPerReplica=0.1
ClipNumGPUPerReplica=0.14
ClipNumReplicas=1
ClipMaxCon=100


ComposedNumCPUPerReplica=0.1
ComposedNumReplicas=1
ComposedMaxCon=100

SnowflakeResultsQueue=content_moderation_profile-results_dev
AiModelBucket=lomotif-datalake-dev
```

# Additional variables for internal testing
For DS Team internal testing, we also need to add the following env vars to the `.env` file:
```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=us-east-2
```
and uncomment these lines in `tasks.py`:
```
# from dotenv import load_dotenv

# load_dotenv("./.env")
```
To prepare the conda environment to test the script:
```
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git --no-deps
pip install -U "ray[default]==1.11.1"
pip install "ray[serve]"
pip install pytest
```

# For use in `g4dn.2xlarge` instance, use the following variables instead
```
DownloadNumCPUPerReplica=0.2
DownloadNumReplicas=1
DownloadMaxCon=1000
NudenetNumCPUPerReplica=0.7
NudenetNumReplicas=3
NudenetMaxCon=1000
MFDNumCPUPerReplica=0.7
MFDNumReplicas=3
MFDMaxCon=1000
ClipNumCPUPerReplica=0.1
ClipNumGPUPerReplica=0.17
ClipNumReplicas=4
ClipMaxCon=1000
ComposedNumCPUPerReplica=0.1
ComposedNumReplicas=1
ComposedMaxCon=1000
SnowflakeResultsQueue=content_moderation_profile-results_dev
AiModelBucket=lomotif-datalake-dev
RAY_LOG_TO_STDERR=1
```

# Instructions (Docker)
1) Ensure there are environment variables or `.env` file, see section above for environment variables.
2) Ensure GPU for docker is enabled. See section below.
3) Once the container is able to detect the GPU, we can follow the normal process of

```
docker-compose build
docker-compose up
```

# Enabling GPU for Docker
To enable the GPU for Docker, make sure Nvidia drivers for the system are installed. [Refer link for details](https://linuxconfig.org/how-to-install-the-nvidia-drivers-on-ubuntu-18-04-bionic-beaver-linux)

Commands which can help install Nvidia drivers are:
```
unbuntu-drivers devices
sudo ubuntu-drivers autoinstall
```

Then nvidia-docker2 tools needs to be installed.
To install follow the below instructions.
[Refer link for details](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```
curl https://get.docker.com | sh   && sudo systemctl --now enable docker
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)    && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -    && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

# Testing the code locally
1) Test if the code is working as expected. Firstly on terminal, do:
```bash
ray start --head --port=6300
```
2) Then, deploy the ray services:
```bash
python serve_tasks/tasks.py
```
3) Run this:
```bash
python serve_demo.py
```

# More details about the output
<!-- The output will be written to this table on Snowflake: `DS_CONTENT_MODERATION_TAGGING_1ST_LAYER` (In production). -->
Example output upon sending a request to the deployment service:
```python
{'UID': '37217839', 'USERNAME': 'danysantynener07', 'IMAGE_URL': 'https://cdn.lomotif.com/user/profile/7c754bea1aa6672b/5e459c67de6fec5d.png', 'COUNTRY': 'co', 'EVENT_TIME': '2022-06-10 01:21:01', 'MESSAGE_RECEIVE_TIME': '2022-06-21 02:18:04.776753+00:00', 'TOTAL_DURATION': 1.208, 'MODEL_ATTRIBUTES': {'NN_DURATION': 0.053, 'NN_SAFE_SCORES': '0.9127', 'NN_UNSAFE_SCORES': '0.0873', 'NN_TO_BE_MODERATED': False, 'NN_PREDICTION_SUCCESS': True, 'NN_STATUS': 0, 'CLIP_DURATION': 0.048, 'CLIP_PREDICTION_SUCCESS': True, 'CLIP_TO_BE_MODERATED': False, 'CLIP_STATUS': 0, 'COOP_DURATION': 0.91, 'COOP_PREDICTION_SUCCESS': True, 'COOP_STATUS': 0, 'PREDICTED_TOP3_PRIMARY_CATEGORY': 'selfies, life-style, beauty-and-grooming', 'PREDICTED_TOP3_SECONDARY_CATEGORY': 'selfie, fashion, make-up', 'MFD_DURATION': 0.066, 'MFD_TO_BE_MODERATED': False, 'MFD_PREDICTION_SUCCESS': True, 'MFD_STATUS': 0, 'USER_ID': '37217839'}, 'PREDICTED_PRIMARY_CATEGORY': 'selfies, life-style', 'PREDICTED_SECONDARY_CATEGORY': 'selfie, fashion', 'MODEL_VERSION': '1.0.0', 'TO_BE_MODERATED': False}
```
- UID, USERNAME, IMAGE_URL, COUNTRY, EVENT_TIME: As per event_type=modify_user definition.
- MESSAGE_RECEIVE_TIME: UTC time where kinesis message is received by the deployment service.
- (NN/CLIP/COOP/MFD/TOTAL)_DURATION: Time in seconds taken by respective models.
- NN_SAFE_SCORES: Nudenet safe scores per key frame.
- NN_UNSAFE_SCORES: Nudenet unsafe scores per key frame.
- (NN/CLIP/COOP/MFD)_TO_BE_MODERARED: True if needs to be moderated. Otherwise False.
- PREDICTED_PRIMARY_CATEGORY: primary category prediction.
- PREDICTED_SECONDARY_CATEGORY: secondary category prediction.
- (NN/CLIP/COOP/MFD)_PREDICTION_SUCCESS: True if STATUS is 0. Otherwise False.
- (NN/CLIP/COOP/MFD)_STATUS: 
    - 0: Prediction successful. 
    - 1: Not a video or image, prediction unsuccesful. 
    - 403: Video clip file not found, prediction unsuccessful. Or Lomotif does not exist on S3, cannot be downloaded after retries, prediction unsuccessful.
    - 4: Some unknown error in the model that was caught by the try...except... loop. Prediction unsucessful.
    - 5: No key frames selected. Prediction unsucessful.




