# 상수 변하지않는 변수를 뜻함

import numpy as np

FFT_HOP = 256 #  주파수 변환 특징
N_FFT = 8 * FFT_HOP

NOTES_BINS_PER_SEMITONE = 1  # 반음 당 음표의 수
CONTOURS_BINS_PER_SEMITONE = 3 
# 신호의 주파수 또는 시간 영역 특성의 특정 측면을 캡처한다는 것을 의미
# 첫 번째 반음의 central bin의 기본 주파수
# second bin if annotations_bins_per_semitone is 3)
ANNOTATIONS_BASE_FREQUENCY = 27.5  # 피아노의 가장 낮은 건반
ANNOTATIONS_N_SEMITONES = 88  # 피아노 건반 수
AUDIO_SAMPLE_RATE = 22050     # 오디오 샘플 속도
AUDIO_N_CHANNELS = 1
N_FREQ_BINS_NOTES = ANNOTATIONS_N_SEMITONES * NOTES_BINS_PER_SEMITONE
# 음찾을때 기준점 피아노 건반에 반음 당 음표의 수
N_FREQ_BINS_CONTOURS = ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE
# 스펙트럼 분석, 피치감지 등에 쓰임
AUDIO_WINDOW_LENGTH = 2  # 훈련 데이터셋에서 첫 번째 원본에 속하는 훈련 예제의 시간(초)을 의미

ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP
ANNOTATION_HOP = 1.0 / ANNOTATIONS_FPS

# 우리가 계산하는 시간-주파수 표현의 프레임 수입니다
ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH

# AUDIO_N_SAMPLES는 모델에 대한 입력으로 사용하는 (클리핑된) 오디오의 샘플 수
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP

DATASET_SAMPLING_FREQUENCY = {
    "MAESTRO": 5,
    "GuitarSet": 2,
    "MedleyDB-Pitch": 2,
    "iKala": 2,
    "slakh": 2,
}


def _freq_bins(bins_per_semitone: int, base_frequency: float, n_semitones: int) -> np.array:
    d = 2.0 ** (1.0 / (12 * bins_per_semitone))
    bin_freqs = base_frequency * d ** np.arange(bins_per_semitone * n_semitones)
    return bin_freqs


FREQ_BINS_NOTES = _freq_bins(NOTES_BINS_PER_SEMITONE, ANNOTATIONS_BASE_FREQUENCY, ANNOTATIONS_N_SEMITONES)
FREQ_BINS_CONTOURS = _freq_bins(CONTOURS_BINS_PER_SEMITONE, ANNOTATIONS_BASE_FREQUENCY, ANNOTATIONS_N_SEMITONES)
