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
    persons1_file_name = '{}-{}.png'.format(uid, 'persons1')
    persons2_file_name = '{}-{}.png'.format(uid, 'persons2')
    s3_directory_path = 'images/{}'
    image = request.files['image']
    input_file_name = '{}-{}'.format(uid, image.filename)
    try:
        with open(input_file_name, 'wb') as f:
            f.write(image.read())

        input_img, mask_img, persons1_img, persons2_img = instance_segmentation_api(input_file_name)
        if (mask_img is None):
            return jsonify({
                'code': 'error',
                'message': 'No Persons on Image'
            })
        background_img = inpainting_api(input_img, mask_img)
        cv2.imwrite(background_file_name, background_img)
        if (persons1_img is not None):
            cv2.imwrite(persons1_file_name, get_transparent_img(persons1_img))
        if (persons2_img is not None):
            cv2.imwrite(persons2_file_name, get_transparent_img(persons2_img))

        s3 = boto3.client('s3', region_name='ap-northeast-2')
        s3.upload_file(background_file_name, app.config['S3_BUCKET'], s3_directory_path.format(background_file_name))
        if (persons1_img is not None):
            s3.upload_file(persons1_file_name, app.config['S3_BUCKET'], s3_directory_path.format(persons1_file_name))
        if (persons2_img is not None):
            s3.upload_file(persons2_file_name, app.config['S3_BUCKET'], s3_directory_path.format(persons2_file_name))
        os.remove(input_file_name)
        os.remove(background_file_name)
        if (persons1_img is not None):
            os.remove(persons1_file_name)
        if (persons2_img is not None):
            os.remove(persons2_file_name)

        data = {
            'background': "{}{}".format(app.config['S3_LOCATION'], s3_directory_path.format(background_file_name)),
        }
        if (persons1_img is not None):
            data['persons1'] = "{}{}".format(app.config['S3_LOCATION'], s3_directory_path.format(persons1_file_name))
        if (persons2_img is not None):
            data['persons2'] = "{}{}".format(app.config['S3_LOCATION'], s3_directory_path.format(persons2_file_name))
        return jsonify({
            'code': 'ok',
            'data': data
        })
    except Exception as e:
        print(e)
        try:
            os.remove(input_file_name)
            os.remove(background_file_name)
            os.remove(persons1_file_name)
            os.remove(persons2_file_name)
        except:
            pass
    return abort(500)

if __name__ == '__main__':
   app.run(host='0.0.0.0', debug = True)
