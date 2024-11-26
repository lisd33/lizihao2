import base64
import json

import cv2
import numpy as np
from flask import Flask, request
from ultralytics import YOLO

import yaml


MODEL_CLS_2_REAL_ID = {
    0: 1,
    1: 4,
    2: 3,
    3: 2
}


config_file = open('./config.yaml')
config = yaml.load(config_file, yaml.loader.SafeLoader)
config_file.close()

app = Flask(__name__)
model = YOLO(model=config['infer_weights'])


@app.route('/infer', methods=['get', 'post'])
def infer():
    ret = {"success": False}
    # 获取图片文件
    img = request.files.get('img')

    if img is None:
        ret = {'code': -1, 'message': '文件为空'}
        print("图片文件未上传")
        return json.dumps(ret, ensure_ascii=False)

    try:
        # 得到客户端传输的图像
        input_image = img.read()
        print(f"Image size: {len(input_image)} bytes")
        
        # 确保图像数据非空
        if len(input_image) == 0:
            ret = {'code': -2, 'message': '图像数据为空'}
            print("图像数据为空")
            return json.dumps(ret, ensure_ascii=False)

        imBytes = np.frombuffer(input_image, np.uint8)
        in_img = cv2.imdecode(imBytes, cv2.IMREAD_COLOR)

        # 确保图像成功解码
        if in_img is None:
            ret = {'code': -3, 'message': '图像解码失败'}
            print("图像解码失败")
            return json.dumps(ret, ensure_ascii=False)

        print("图像成功解码，开始推理...")

        # 进行模型推理
        result = model.predict(source=in_img, save=True, imgsz=640, device='cuda')[0]
        results = []

        for box in result.boxes:
            cls_id = int(box.cls.cpu().numpy()[0])
            ret_id = MODEL_CLS_2_REAL_ID.get(cls_id, -1)
            x, y, w, h = box.xywh.cpu().numpy()[0]

            results.append({
                'cls': ret_id,
                'xywh': [float(x), float(y), float(w), float(h)]
            })

        ret = {
            'code': 200,
            'message': '成功',
            'data': results
        }
    except Exception as e:
        print(f"发生异常：{e}")
        ret = {'code': -4, 'message': f'推理过程出错: {str(e)}'}

    return json.dumps(ret, ensure_ascii=False)




if __name__ == "__main__":
    port = config['port']
    app.run(debug=False, host='0.0.0.0', port=5000)
