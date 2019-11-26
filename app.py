from flask import Flask, jsonify, request
import os
import cv2
import boto3
import uuid

from inpaint import inpainting_api
from segmentation import instance_segmentation_api
from util import get_transparent_img

app = Flask(__name__)
app.config.from_pyfile('config.py')

@app.route('/thanos', methods=['POST'])
def thanos():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        input_img, mask_img, persons_img = instance_segmentation_api('origin.jpeg')
        background_img = inpainting_api(input_img, mask_img, 'output.png')

        background_file_name = '{}-{}.png'.format(uuid.uuid4(), 'background')
        persons_file_name = '{}-{}.png'.format(uuid.uuid4(), 'persons')
        cv2.imwrite(background_file_name, background_img)
        cv2.imwrite(persons_file_name, get_transparent_img(persons_img))

        s3 = boto3.client('s3', region_name='ap-northeast-2')
        s3.upload_file(background_file_name, app.config['S3_BUCKET'], background_file_name)
        s3.upload_file(persons_file_name, app.config['S3_BUCKET'], persons_file_name)
        os.remove(background_file_name)
        os.remove(persons_file_name)
        return jsonify({
            'background': "{}{}".format(app.config['S3_LOCATION'], background_file_name),
            'persons': "{}{}".format(app.config['S3_LOCATION'], persons_file_name),
        })

if __name__ == '__main__':
   app.run(debug = True)
