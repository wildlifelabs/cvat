import base64
import io
import json

import yaml
from model_handler import ModelHandler
from PIL import Image


def init_context(context):
    try:
        context.logger.info("Start Init")
        with open("/opt/nuclio/function.yaml", "rb") as function_file:
            functionconfig = yaml.safe_load(function_file)
        labels_spec = functionconfig["metadata"]["annotations"]["spec"]
        labels = {item["id"]: item["name"] for item in json.loads(labels_spec)}
        model = ModelHandler(labels)
        context.user_data.model = model
        context.logger.info("End Init")
    except Exception as ex:
        raise Exception(f"Cannot init context: {ex}")

def handler(context, event):
    try:
        context.logger.info("Run YoloV7 ONNX model")
        data = event.body
        buf = io.BytesIO(base64.b64decode(data["image"]))
        threshold = float(data.get("threshold", 0.5))
        image = Image.open(buf).convert("RGB")
        shapes = data.get("shapes")
        states = data.get("states")
        context.logger.error("in --> shapes")
        context.logger.error(shapes)
        context.logger.error("in --> states")
        context.logger.error(states)
        results = context.user_data.model.infer(image, threshold, shapes, states)
        context.logger.error("out --> results JSON")
        context.logger.error(json.dumps(results))
        return context.Response(
            body=json.dumps(results), headers={}, content_type="application/json", status_code=200
        )
    except Exception as ex:
        raise Exception(f"Cannot handle context: {ex}")
