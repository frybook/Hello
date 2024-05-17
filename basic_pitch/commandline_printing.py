# 명령어 인쇄

import os
import pathlib
import threading
from contextlib import contextmanager
from typing import Iterator, Union

TF_LOG_LEVEL_KEY = "TF_CPP_MIN_LOG_LEVEL"
TF_LOG_LEVEL_NO_WARNINGS_VALUE = "3"
s_print_lock = threading.Lock()
OUTPUT_EMOJIS = {
    "MIDI": "💅",
    "MODEL_OUTPUT_NPZ": "💁‍♀️",
    "MIDI_SONIFICATION": "🎧",
    "NOTE_EVENTS": "🌸",
}


def generating_file_message(output_type: str) -> None:
    """파일이 생성되고 있다는 메세지를 인쇄합니다.
    
    Args:
        output_type: 생성되는 파일 종류를 나타내는 문자열

    """
    print(f"\n\n  Creating {output_type.replace('_', ' ').lower()}...")


def file_saved_confirmation(output_type: str, save_path: Union[pathlib.Path, str]) -> None:
    """파일이 성공적으로 저장되었다는 확인 메세지를 인쇄합니다.

    Args:
        output_type: 생성되는 파일의 종류 입니다.
        save_path: 출력 파일의 경로입니다.

    """
    print(f"  {OUTPUT_EMOJIS[output_type]} Saved to {save_path}")


def failed_to_save(output_type: str, save_path: Union[pathlib.Path, str]) -> None:
    """Print a failure to save message

    Args:
        output_type: The kind of file that is being generated.
        save_path: The path to output file.

    """
    print(f"\n🚨 Failed to save {output_type.replace('_', ' ').lower()} to {save_path} \n")


@contextmanager
def no_tf_warnings() -> Iterator[None]:
    """
    이 컨텍스트에서 텐서플로우 경고를 억제합니다.
    """
    tf_logging_level = os.environ.get(TF_LOG_LEVEL_KEY, TF_LOG_LEVEL_NO_WARNINGS_VALUE)
    os.environ[TF_LOG_LEVEL_KEY] = TF_LOG_LEVEL_NO_WARNINGS_VALUE
    yield
    os.environ[TF_LOG_LEVEL_KEY] = tf_logging_level
