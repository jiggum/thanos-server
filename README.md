# Thanos Server

## Before you start
#### Locate [inpaint](https://drive.google.com/open?id=1bgNv9wtk_ExRgJBhWv1RtaAqqCP1ypGS) directory under the *model_logs* directory


## Run

#### 1. Run your virtualenv with python3

#### 2. Install dependency with `pip install -r requirements.txt`

#### 3. Start Flask server with `S3_BUCKET_NAME='YOUR_BUCKET_NAME' python app.py`


## Project Structure
#### app.py
Contain Flask's endpoint handler. Integrating and Image save process are located.

#### segmentation.py
Contain instant segmentation logic with torchvision. This module generate four image.
1. Persons erased image
2. Whole persons mask
3. Half of persons mask
4. Another half of persons mask

#### inpaint.py
Contain [inpainting](https://github.com/JiahuiYu/generative_inpainting) logic with tensorflow.
From the segmentation module's '1. Persons erased image' and '2. whole persons mask', process inpainting logic and return output image.
