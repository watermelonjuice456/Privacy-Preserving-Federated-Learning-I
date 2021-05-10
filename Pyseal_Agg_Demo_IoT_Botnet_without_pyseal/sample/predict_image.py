from pprint import pprint

import requests

url = "http://localhost:7000/predict_image"

img_name = "test_cropped.png"
with open(img_name, 'rb') as img:
    files = {'image': (img_name, img, 'multipart/form-data', {'Expires': '0'})}
    with requests.Session() as s:
        r = s.post(url, files=files)
        print(r.status_code)
        print(r.content)
        pprint(r.json())

