import os
S3_BUCKET = os.environ.get("S3_BUCKET_NAME")
S3_LOCATION = 'http://{}.s3.amazonaws.com/'.format(S3_BUCKET)
