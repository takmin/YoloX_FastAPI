from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
import base64
import os
import configparser
import json
from detector import DetectorYolo, backend_target_pairs
import logging
import datetime
import time
import asyncio
import cv2 as cv
import numpy as np


def LoadConfigFile(filename):
    '''Load Config File
    '''
    params = dict()

    config = configparser.ConfigParser()
    config.read(filename)
    if(not "Yolo" in config):
        print("[Yolo] is needed in config.ini")
        return None

    yolo_config = config["Yolo"]

    params["backend"] = 0
    if("backend" in yolo_config):
        params["backend"] = int(yolo_config["backend"])

    if(not "onnxFile" in yolo_config):
        print("\"onnxFile\" is needed under [Yolo] in config.ini")
        return None
    params["onnxFile"] = yolo_config["onnxFile"]

    if(not "classes" in yolo_config):
        print("\"classes\" is needed under [Yolo] in config.ini")
        return None
    params["classes"] = json.loads(yolo_config["classes"])

    
    params["confidence"] = 0.5
    if("confidence" in yolo_config):
        params["confidence"] = float(yolo_config["confidence"])
    
    params["nmsIoU"] = 0.5
    if("nmsIoU" in yolo_config):
        params["nmsIoU"] = float(yolo_config["nmsIoU"])

    params["threshold"] = 0.5
    if("threshold" in yolo_config):
        params["threshold"] = float(yolo_config["threshold"])

    params["log_path"] = "logs"
    if("Server" in config):
        server_conf = config["Server"]
        if("LogDirectory" in server_conf):
            params["log_path"] = server_conf["LogDirectory"]
        if("ImageLogDirectory" in server_conf):
            params["save_img_path"] = server_conf["ImageLogDirectory"]
    return params


def create_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO) # 出力レベルを設定
    stream_h = logging.StreamHandler()
    stream_h.setLevel(logging.DEBUG) # 出力レベルを設定
    fmt = logging.Formatter('%(name)s : %(levelname)s : %(message)s')
    stream_h.setFormatter(fmt)
    logger.addHandler(stream_h)
    return logger


def add_logfile_to_logger(logger, log_file):
    file_h = logging.FileHandler(log_file)
    fmt = logging.Formatter('%(asctime)s : %(name)s : %(levelname)s : %(message)s')
    file_h.setFormatter(fmt)
    file_h.setLevel(logging.DEBUG)
    logger.addHandler(file_h)


def log_result(img, ext, save_dir, logger, results, proctime):
    timestamp = str(time.time())
    log_msg = "{} is recognized as [{}] in {} sec"
    result_ids = [rslt["label"] for rslt in results]
    filename = timestamp + "." + ext
    log_msg = log_msg.format(filename, str(result_ids), proctime)
    logger.info(log_msg)
    if(save_dir is None):
        return
    save_img_file = os.path.join(save_dir, filename)
    with open(save_img_file, 'wb') as f:
        f.write(img)

   
###### Create FastAPI Instance ######
app = FastAPI()
####################################3

### Load Config File ###
config_file = 'config.ini'
params = LoadConfigFile(config_file)
if(params is None):
    exit()

# Prepare Log File
if(not os.path.isdir(params["log_path"])):
    os.mkdir(params["log_path"])

log_filename = str(datetime.datetime.now().date())
log_file_path = os.path.join(params["log_path"], log_filename+".log")
logger = create_logger(__name__)
add_logfile_to_logger(logger, log_file_path)

# Prepare Directory to save uploaded images
save_img_dir = None
if("save_img_path" in params):
    if(not os.path.isdir(params["save_img_path"])):
        os.mkdir(params["save_img_path"])
    save_img_dir = os.path.join(params["save_img_path"], log_filename)
    if(not os.path.isdir(save_img_dir)):
        os.mkdir(save_img_dir)

# Load Yolo Model
detector_net = None
try:
    backend_target_ids = backend_target_pairs[params["backend"]]
    backend_id = backend_target_ids[0]
    target_id = backend_target_ids[1]
    detector_net = DetectorYolo(onnxFile=params["onnxFile"],
                            backend_id=backend_id, target_id=target_id,
                            classes=params["classes"],
                            confidence=params["confidence"], nmsIoU=params["nmsIoU"], threshold=params["threshold"])
except Exception:
    logger.error("Fail to load Yolo model. config.ini may be wrong.")
    exit() 
logger.info("Yolo Model loaded.")


@app.post("/api/detect")
async def query(request: Request):
    try:
        start = time.perf_counter()

        # Parse the request body as JSON
        item = await request.json()
        logger.debug('got item')

        # Access the file and type_value from the JSON object
        file_base64 = item['image']
        logger.debug('got image')
        vis = False
        if('visualize' in item):
            vis = item['visualize']

        # Get the file format and the base64 string from the file_base64 string
        format, img_data64 = file_base64.split(';base64,')
        ext = format.split('/')[-1]

        # Decode the base64 string to get the original file content
        file_content = base64.b64decode(img_data64)

        # Convert the file content to a NumPy array
        file_array = np.frombuffer(file_content, dtype=np.uint8)

        img = cv.imdecode(file_array, flags=cv.IMREAD_COLOR)
        result = {}
        detected = detector_net.infer(img)
        if(vis):
            vis_img = detector_net.vis(detected, img)
            _, buffer = cv.imencode('.png', vis_img)
            byte_arr = buffer.tobytes()
            result["visualize"] = base64.b64encode(byte_arr).decode('utf-8')

        for det in detected:
            det["position"] = det["position"].tolist()
        result["detected"] = detected
                
        proctime = time.perf_counter() - start
        asyncio.new_event_loop().run_in_executor(None, log_result, file_content, ext, save_img_dir, logger, detected, proctime)
        return result

    except Exception as e:
        print(e)
        logger.error(e)



@app.get("/")
async def main():
    content = """
<body>
<input name="file" type="file" id="tgt_image"/>
<br/>
<input type="checkbox" id="visualize" name="visualize">visualize</input><br/>
<img id="inputImage"/><br/>
<button id="send_button" onclick='send()'>Submit</button>
<br/>
<div id="result"></div>
<img id="outputImage"/>
<script>
    const input_file = document.getElementById("tgt_image");
    const srcImage = document.querySelector('#inputImage');
    input_file.addEventListener("change", function (e) {
      const file = e.target.files[0];
      const reader = new FileReader();
      reader.addEventListener("load", function () {
        srcImage.src = reader.result;
      }, false);

      if (file) {
        reader.readAsDataURL(file);
      }
    })

    // Send the selected file to the server
    function send() {
        const visInput = document.querySelector('#visualize');

        // Create an object with the file and type_value
        const data = {
            image: srcImage.src,
            visualize: visInput.checked,
        };

        // Send an HTTP POST request with the object as the body
        fetch('http://localhost:8000/api/detect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        })
        .then((response) => response.text())
        .then((responseText) => {
            // handle the response
            response_data = JSON.parse(responseText);
            document.querySelector('#result').innerText = JSON.stringify(response_data["detected"]);
            if(response_data.hasOwnProperty("visualize")){
                document.querySelector('#outputImage').src = "data:image/png;base64," + response_data["visualize"];
            }
        });
    };
</script>
</body>
    """
    return HTMLResponse(content=content)