from yolox import YoloX
import cv2 as cv
import numpy as np

backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]


class DetectorYolo:
    def __init__(self, onnxFile, backend_id, target_id, classes, confidence, nmsIoU, threshold) -> None:
        # Valid combinations of backends and targets
        self.classes = classes
        self.model = YoloX(modelPath=onnxFile,
                          confThreshold=confidence,
                          nmsThreshold=nmsIoU,
                          objThreshold=threshold,
                          backendId=backend_id,
                          targetId=target_id)
    
    def infer(self, img):
        input_blob, letterbox_scale = self.letterbox(img)
        preds = self.model.infer(input_blob)
        result = [self.__create_detect_info(det, letterbox_scale) for det in preds]
        return result


    def __create_detect_info(self, det, letterbox_scale):
        detect_info = {}
        box = self.unletterbox(det[:4], letterbox_scale).astype(np.int32)
        cls_id = int(det[-1])
        detect_info["id"] = cls_id
        detect_info["label"] = self.classes[cls_id]
        detect_info["score"] = det[-2]
        detect_info["position"] = box
        return detect_info


    # 入力画像を640 x 640に収まるようにリサイズして、余った領域をグレーでPadding
    def letterbox(cls, srcimg, target_size=(640, 640)):
        padded_img = np.ones((target_size[0], target_size[1], 3)).astype(np.float32) * 114.0
        ratio = min(target_size[0] / srcimg.shape[0], target_size[1] / srcimg.shape[1])
        resized_img = cv.resize(
            srcimg, (int(srcimg.shape[1] * ratio), int(srcimg.shape[0] * ratio)), interpolation=cv.INTER_LINEAR
        ).astype(np.float32)
        padded_img[: int(srcimg.shape[0] * ratio), : int(srcimg.shape[1] * ratio)] = resized_img

        return padded_img, ratio


    def unletterbox(cls, bbox, letterbox_scale):
        return bbox / letterbox_scale


    # 結果の描画
    def vis(self, dets, srcimg, fps=None):
        res_img = srcimg.copy()

        if fps is not None:
            fps_label = "FPS: %.2f" % fps
            cv.putText(res_img, fps_label, (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for det in dets:
            box = det["position"]
            score = det["score"]

            x0, y0, x1, y1 = box

            text = '{}:{:.1f}%'.format(det["label"], score * 100)
            font = cv.FONT_HERSHEY_SIMPLEX
            txt_size = cv.getTextSize(text, font, 0.4, 1)[0]
            cv.rectangle(res_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
            cv.rectangle(res_img, (x0, y0 + 1), (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])), (255, 255, 255), -1)
            cv.putText(res_img, text, (x0, y0 + txt_size[1]), font, 0.4, (0, 0, 0), thickness=1)

        return res_img

