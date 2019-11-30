from flask import Flask, jsonify, request, abort
from flask_cors import cross_origin
import os
import cv2
import boto3
import uuid

from inpaint import inpainting_api
from segmentation import instance_segmentation_api
from util import get_transparent_img

app = Flask(__name__)
app.config.from_pyfile('config.py')

@app.route('/')
def health_check():
    return 'Alive!'

@app.route('/thanos', methods=['POST'])
@cross_origin()
def thanos():
    uid = uuid.uuid4()
    background_file_name = '{}-{}.png'.format(uid, 'background')
    persons_file_name = '{}-{}.png'.format(uid, 'persons')
    image = request.files['image']
    input_file_name = '{}-{}'.format(uid, image.filename)
    try:
        with open(input_file_name, 'wb') as f:
            f.write(image.read())

        input_img, mask_img, persons_img = instance_segmentation_api(input_file_name)
        background_img = inpainting_api(input_img, mask_img)


        cv2.imwrite(background_file_name, background_img)
        cv2.imwrite(persons_file_name, get_transparent_img(persons_img))

        s3 = boto3.client('s3', region_name='ap-northeast-2')
        s3.upload_file(background_file_name, app.config['S3_BUCKET'], background_file_name)
        s3.upload_file(persons_file_name, app.config['S3_BUCKET'], persons_file_name)
        os.remove(background_file_name)
        os.remove(persons_file_name)
        os.remove(input_file_name)
        return jsonify({
            'background': "{}{}".format(app.config['S3_LOCATION'], background_file_name),
            'persons': "{}{}".format(app.config['S3_LOCATION'], persons_file_name),
        })
    except Exception as e:
        print(e)
        try:
            os.remove(background_file_name)
            os.remove(persons_file_name)
            os.remove(input_file_name)
        except:
            pass
    return abort(500)

if __name__ == '__main__':
   app.run(host='0.0.0.0', debug = True)
