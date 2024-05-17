
import enum
import logging
import pathlib


try:
    import coremltools

    CT_PRESENT = True
except ImportError:
    CT_PRESENT = False
    logging.warning(
        "Coremltools가 설치되지 않았습니다."
        "CoreML 저장된 모델을 사용하려는 경우 "
        "`pip install 'basic-pitch[coreml]'`로 기본 피치를 다시 설치하세요."
    )

try:
    import tflite_runtime.interpreter

    TFLITE_PRESENT = True
except ImportError:
    TFLITE_PRESENT = False
    logging.warning(
        "tflite-runtime이 설치되지 않았습니다."
        "TFLite 모델을 사용할 계획이라면,"
        "`pip install 'basic-pitch tflite-runtime'`으로 basic-pitch를 다시 설치하거나"
        "`pip 설치 '기본 피치[tf]'"
    )

try:
    import onnxruntime

    ONNX_PRESENT = True
except ImportError:
    ONNX_PRESENT = False
    logging.warning(
        "onnxruntime이 설치되지 않았습니다."
        "ONNX 모델을 사용할 계획이라면,"
        "`pip install 'basic-pitch[onnx]'`로 기본 피치를 다시 설치하세요."
    )


try:
    import tensorflow

    TF_PRESENT = True
except ImportError:
    TF_PRESENT = False
    logging.warning(
        "텐서플로우가 설치되지 않았습니다."
        "TF 저장 모델을 사용할 계획이라면,"
        "`pip install 'basic-pitch[tf]'`로 기본 피치를 다시 설치하세요."
    )


class FilenameSuffix(enum.Enum):
    tf = "nmp"
    coreml = "nmp.mlpackage"
    tflite = "nmp.tflite"
    onnx = "nmp.onnx"


if TF_PRESENT:
    _default_model_type = FilenameSuffix.tf
elif CT_PRESENT:
    _default_model_type = FilenameSuffix.coreml
elif TFLITE_PRESENT:
    _default_model_type = FilenameSuffix.tflite
elif ONNX_PRESENT:
    _default_model_type = FilenameSuffix.onnx


def build_icassp_2022_model_path(suffix: FilenameSuffix) -> pathlib.Path:
    return pathlib.Path(__file__).parent / "saved_models/icassp_2022" / suffix.value


ICASSP_2022_MODEL_PATH = build_icassp_2022_model_path(_default_model_type)
