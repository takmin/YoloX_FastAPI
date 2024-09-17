# YoloX + FastAPI

2024/03/15 MINAGAWAA Takuya

I created simple API for object detector YOLOX.


YOLOX code and a onnx model are from OpenCV Zoo:

https://github.com/opencv/opencv_zoo/tree/main/models/object_detection_yolox


## Install
git and pip are required.

```
$ git clone https://github.com/takmin/YoloX_FastAPI.git
$ cd YoloX_FastAPI
$ pip install -r requirements.txt
```

## Configuration

Edit configuration file "config.ini" for parameters for Yolo and log file.

* backend: What backend and target processors are:
	* 0: (default) OpenCV implementation + CPU
	* 1: CUDA + GPU (CUDA)
	* 2: CUDA + GPU (CUDA FP16)
	* 3: TIM-VX + NPU
	* 4: CANN + NPU
* onnxFile: File path to YOLOX trained model file (onnx format)
* confidence: Class confidence for YoloX
* nmsIoU: NMS IoU threshold for YoloX
* threshold: Object threshold for YoloX
* LogDirectory: Directory to store log files.
* ImageLogDirectory: Directory to store images. If comment out, no image is stored.

	

## Start API

You launch api service as below:
```
$  uvicorn detect_api:app
```

To get object detection results using API, HTTP POST json to "http://<uri>:<port>/api/detect" like below:

```
{
  image: <base64 encoded image>,
  visualize: <"true" to get visualized result>
{
```

Then you will get the following response
```
{
  detected: [
    {
      "id":0,
      "label":"person",
      "score":0.8404238820075989,
      "position":[711,73,1008,879]
    },
    {
      "id":2,
      "label":"car",
      "score":0.6321859955787659,
      "position":[65,28,846,880]
    }
  ],
  visualize: <base64 encoded image>
}

```

"position" indicates coordinates of bounding box left-up and right-bottom corners: x0, y0, x1, y1. 



You can test the API by accessing the following URL in your browser.

http://localhost:8000

