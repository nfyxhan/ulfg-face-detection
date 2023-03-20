# -*- coding: utf-8 -*-
import urllib3
import json
from PIL import Image
import io
import base64
import zlib

http = urllib3.PoolManager()


def pil_to_b64(img):
    img_byte = io.BytesIO()
    img.save(img_byte, format="png")
    img_byte = img_byte.getvalue()
    img_b64 = base64.b64encode(img_byte).decode()
    return img_b64

def b64_to_pil(img_b64):
    raw_image = base64.b64decode(img_b64.encode('utf8'))
    return Image.open(io.BytesIO(raw_image))


def getFace(url, img, draw=False):
    img_b64 = pil_to_b64(img)
    body ={ 
            "img":  img_b64 ,
            "draw": draw,
            }
    body = json.dumps(body, ensure_ascii=False)
    method = 'POST'
    r = http.request(method, url, body=body, headers={'Content-Type': 'application/json'})
    data = json.loads(r.data.decode('utf-8')) #, ensure_ascii=False))
    if draw:
        img_b64 = data['draw_img']
        img_pil = b64_to_pil(img_b64)
        img_pil.save('/tmp/img.png')
        data['draw_img'] = True
    return data

if __name__ == "__main__":
    url = "http://localhost:8090/api/face-detection"
    img_path =  "imgs/1.jpg"
    img = Image.open(img_path)
    draw = True
    data = getFace(url, img, draw)
    print(data)
