import os
from pathlib import Path

FFMPEG_EXE = r"C:\dev\vcpkg\installed\x64-windows\tools\ffmpeg\ffmpeg.exe"
FFPROBE_EXE = r"C:\dev\vcpkg\installed\x64-windows\tools\ffmpeg\ffprobe.exe"

# torchcodec (used by transformers/sentence-transformers to decode clip .mp4 files for the
# §8 motion index) dynamically links libtorchcodec_core*.dll against the vcpkg FFmpeg shared
# libraries and, because that FFmpeg build was compiled with --enable-libnpp, transitively
# against the CUDA Toolkit's NPP DLLs. Since Python 3.8, ctypes/torch.ops.load_library no
# longer searches PATH for a DLL's own dependencies on Windows, so these must be registered
# via os.add_dll_directory() before any video is decoded. Missing directories are skipped
# silently (e.g. on a machine without this exact CUDA Toolkit version) rather than raising,
# since only the video/motion path needs them — image-only usage must keep working.
_DLL_SEARCH_DIRS = [
    r"C:\dev\vcpkg\installed\x64-windows\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
]
if hasattr(os, "add_dll_directory"):
    for _dir in _DLL_SEARCH_DIRS:
        if os.path.isdir(_dir):
            os.add_dll_directory(_dir)

EMBEDDING_MODEL_ID = "Qwen/Qwen3-VL-Embedding-2B"
RERANKER_MODEL_ID = "Qwen/Qwen3-VL-Reranker-2B"
EXPLAIN_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

DEFAULT_FPS = 1.0
DEFAULT_FRAME_WIDTH = 896
DEFAULT_THUMB_WIDTH = 320
DEFAULT_JPEG_QUALITY = 3  # ffmpeg -q:v scale (2=high .. 31=low)

DEFAULT_SCENE_THRESHOLD = 0.08

DEFAULT_RECALL_TOP_M = 150
DEFAULT_SEGMENT_GAP_SEC = 2.0
DEFAULT_RERANK_TOP_SEGMENTS = 20

DEFAULT_CLIP_WINDOW_SEC = 4.0
DEFAULT_CLIP_STRIDE_SEC = 2.0
DEFAULT_CLIP_WIDTH = 640
DEFAULT_CLIP_BATCH_SIZE = 2
MIN_CLIP_TAIL_SEC = 1.0  # skip a trailing window shorter than this

MIN_PIXELS = 28 * 28 * 64
MAX_PIXELS = 28 * 28 * 640

DEFAULT_EXPLAIN_MAX_FRAMES = 6
DEFAULT_EXPLAIN_MAX_NEW_TOKENS = 512  # open-ended Korean answers routinely passed 256

# Applied on top of greedy decoding (see explain.Explainer.ask). 1.05 was enough to keep the
# degenerate runs away; higher starts penalizing words an answer legitimately repeats.
EXPLAIN_REPETITION_PENALTY = 1.05

# One-line summary written into the manifest at index time and shown as the dropdown tooltip.
DEFAULT_DESCRIBE_MAX_FRAMES = 8
DEFAULT_DESCRIBE_MAX_NEW_TOKENS = 160
DESCRIBE_PROMPT = (
    "이 이미지들은 한 영상에서 시간 순서대로 고르게 뽑은 프레임이다. "
    "이 영상이 어떤 영상인지 두 문장 이내로 간결하게 설명해줘. "
    "장소, 등장하는 사람/사물, 주요 활동을 포함하고, 프레임 번호나 시각은 언급하지 마. "
    # Without this the model sometimes answers a Korean prompt in Chinese.
    "반드시 한국어로 답해줘."
)
