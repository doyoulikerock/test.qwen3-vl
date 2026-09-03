# 설치 가이드 (새 PC에서 시작하기)

이 저장소를 처음 clone한 Windows PC에서 `vidsearch`를 동작시키기 위한 전체 절차.
사용법(명령어 옵션)은 [README.md](README.md), 설계 배경은 [PLAN.md](PLAN.md) 참고.

**현재 Windows 전용이다.** `config.py`에 이 프로젝트가 개발된 머신 경로가 하드코딩돼 있어서
(아래 3단계), 다른 PC로 옮기면 반드시 고쳐야 한다. Linux/macOS는 아직 검증되지 않았다
(`os.add_dll_directory` 같은 Windows 전용 API를 쓰지만 `hasattr` 가드가 있어 최소한 에러 없이 무시는 된다).

## 0. 요구 사항

| 항목 | 최소 조건 | 비고 |
|---|---|---|
| GPU | NVIDIA, VRAM 10GB 이상 | 2B 모델 2개(임베더/리랭커)를 **순차로만** 로드하므로 동시에는 8GB 정도만 있으면 됨. 12GB 카드 기준 디스플레이가 이미 1.5~2GB를 점유하니 실가용은 표기 용량보다 낮게 잡을 것 (PLAN.md §"조사로 확인된 사실" 참고) |
| CUDA 드라이버 | torch 2.14+cu130이 요구하는 최신 드라이버 | 오래된 드라이버면 cu126 빌드로 대체 (2단계) |
| CUDA Toolkit | (선택, `--with-motion` 쓸 경우만) v12.x 아무 버전이나 설치돼 있어야 함 | NPP 라이브러리 경로 필요 — 3단계 참고 |
| OS | Windows 10/11 | |
| Python | 3.12 | |
| [uv](https://github.com/astral-sh/uv) | 최신 | `pip install uv` 또는 공식 설치 스크립트 |
| ffmpeg | shared 빌드 (DLL 포함) | vcpkg 권장 (아래) — 시스템 PATH에 안 넣어도 됨, 경로만 설정 |
| git | | |
| 디스크 여유 공간 | 15GB+ | torch/transformers 등 venv 패키지(~4GB) + Qwen3-VL 모델 2개(각 ~4.5GB, 최초 실행 시 자동 다운로드) |

## 1. 저장소 clone

```powershell
git clone https://github.com/doyoulikerock/test.qwen3-vl.git vidsearch
cd vidsearch
```

## 2. Python 가상환경 + 패키지 설치

```powershell
uv venv --python 3.12
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install "transformers>=4.57" "sentence-transformers>=5.4" qwen-vl-utils accelerate pillow numpy
```

- 드라이버가 오래돼 cu130 설치/구동이 안 되면 `--index-url https://download.pytorch.org/whl/cu126`로 대체.
- 설치 후 GPU 인식 확인:

  ```powershell
  .venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
  ```

  `True <GPU 이름>`이 안 나오면 이후 단계로 넘어가지 말고 드라이버/CUDA 버전부터 맞출 것.

**동작(모션) 질의(`--with-motion`)를 쓸 계획이면** 추가로:

```powershell
uv pip install torchcodec
```

## 3. ffmpeg 설치 + `config.py` 경로 수정 (필수)

`config.py`는 이 프로젝트가 개발된 머신의 절대경로를 하드코딩하고 있다. **clone한 그대로 실행하면
십중팔구 ffmpeg 경로부터 틀려서 바로 실패한다.** 아래 두 군데를 확인·수정해야 한다.

### 3-1. ffmpeg 자체 설치

가장 간단한 방법은 [vcpkg](https://github.com/microsoft/vcpkg)로 shared 빌드를 받는 것(개발 머신도 이 방식):

```powershell
git clone https://github.com/microsoft/vcpkg.git C:\dev\vcpkg
C:\dev\vcpkg\bootstrap-vcpkg.bat
C:\dev\vcpkg\vcpkg.exe install ffmpeg:x64-windows
```

vcpkg가 아니어도 상관없다 — **DLL을 포함한 shared 빌드**(`ffmpeg.exe`/`ffprobe.exe`와 `avcodec-*.dll` 등이
함께 있는 배포판)이기만 하면 된다. static 빌드(exe 하나만 있는 것)는 `--with-motion`을 쓸 경우 3-3단계에서 문제가 된다.

### 3-2. `config.py`의 exe 경로 수정

[vidsearch/config.py](vidsearch/config.py) 4~5번째 줄:

```python
FFMPEG_EXE = r"C:\dev\vcpkg\installed\x64-windows\tools\ffmpeg\ffmpeg.exe"
FFPROBE_EXE = r"C:\dev\vcpkg\installed\x64-windows\tools\ffmpeg\ffprobe.exe"
```

→ 실제 설치한 `ffmpeg.exe`/`ffprobe.exe` 경로로 바꾼다. vcpkg로 설치했다면 그대로 두면 된다.

### 3-3. (선택) `--with-motion`용 DLL 경로 수정

클립(동작 질의) 인덱싱은 `torchcodec`으로 비디오를 디코딩하는데, 이게 ffmpeg 공유 라이브러리에,
그리고 (이 vcpkg ffmpeg 빌드는 `--enable-libnpp`로 컴파일돼 있어서) **CUDA Toolkit의 NPP DLL에도** 의존한다.
Windows + Python 3.8+ 에서는 PATH에 넣는 것만으론 부족해서 `os.add_dll_directory()`로 명시 등록해야 하는데,
그 경로가 [vidsearch/config.py](vidsearch/config.py) 15~18번째 줄에 하드코딩돼 있다:

```python
_DLL_SEARCH_DIRS = [
    r"C:\dev\vcpkg\installed\x64-windows\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
]
```

→ 첫 줄은 vcpkg ffmpeg의 `bin` 폴더(3-1에서 설치한 경로 + `\installed\x64-windows\bin`)로,
둘째 줄은 설치된 CUDA Toolkit 버전(`v12.8`이 아닐 수 있음 — `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\`
아래 실제 폴더명 확인)에 맞게 바꾼다.

**주의**: 이 경로가 틀려도 에러가 나지 않는다(존재하지 않는 디렉토리는 조용히 건너뛴다) — 대신 `--with-motion`으로
클립을 인코딩하려 할 때 `torchcodec`이 DLL을 못 찾아 `ImportError`/`OSError`가 난다. 프레임 기반 검색(기본 기능)은
이 설정과 무관하게 동작하므로, 동작 질의를 안 쓸 거면 이 단계는 건너뛰어도 된다.

CUDA Toolkit 자체가 없다면 [NVIDIA 사이트](https://developer.nvidia.com/cuda-downloads)에서 설치(수 GB, 시간이 걸림).
이미 다른 목적으로 설치돼 있다면 새로 설치할 필요 없이 경로만 맞추면 된다.

## 4. 동작 확인

```powershell
.venv\Scripts\python.exe -m vidsearch index "c:\경로\아무_영상.mp4"
```

정상이면 순서대로 `Probing → Extracting frames → Detecting scene boundaries → Loading ... encoding frames`가
출력되고, 최초 실행이라 Qwen3-VL-Embedding-2B(~4.5GB) 다운로드가 먼저 일어난다(네트워크 필요, 수 분 소요 가능).
완료되면 `data\<영상이름>\`에 `frames/`, `thumbs/`, `embeddings.npy`, `meta.jsonl`, `manifest.json`이 생긴다.

이어서 검색:

```powershell
.venv\Scripts\python.exe -m vidsearch search "c:\경로\아무_영상.mp4" "찾고 싶은 장면을 한국어로"
```

리랭커(Qwen3-VL-Reranker-2B)가 이때 처음 다운로드된다(~4.5GB, 첫 검색만 느림).
결과 표가 뜨고 `ffplay -ss <초> "<영상>"` 명령이 같이 출력되면 정상.

**`--with-motion`까지 확인하려면**:

```powershell
.venv\Scripts\python.exe -m vidsearch index "c:\경로\아무_영상.mp4" --with-motion
```

3-3단계 경로가 맞다면 `Extracting clips → ... encoding clips`까지 에러 없이 끝난다.

## 5. 자주 겪는 문제

| 증상 | 원인 | 해결 |
|---|---|---|
| `index` 실행하자마자 ffmpeg 관련 에러 | `config.py`의 `FFMPEG_EXE`/`FFPROBE_EXE` 경로가 이 PC에 없음 | 3-2단계 |
| `torch.cuda.is_available()`가 `False` | 드라이버가 cu130이 요구하는 버전보다 낮음 | cu126로 재설치(2단계), 그래도 안 되면 드라이버 업데이트 |
| `Warning: You are sending unauthenticated requests to the HF Hub` | HF 계정 미로그인 상태에서 Hub API 호출 — 정상, 무시 가능 | 캐시가 다 받아진 뒤로는 자동 오프라인 전환(`hf_cache.py`)되어 이 경고도 사라짐 |
| `--with-motion`에서 `ImportError: torchcodec is not installed` | 2단계의 `torchcodec` 설치를 건너뜀 | `uv pip install torchcodec` |
| `--with-motion`에서 `Could not load libtorchcodec...` / DLL 관련 `OSError` | 3-3단계의 `_DLL_SEARCH_DIRS` 경로가 이 PC와 안 맞음 | 실제 vcpkg ffmpeg `bin` 폴더, 실제 CUDA Toolkit 버전 폴더로 수정 |
| GPU 메모리 부족(OOM) | 임베더/리랭커를 동시에 올렸거나 배치 크기가 큼 | 코드상 두 모델은 항상 순차 로드/해제되므로 보통 문제 없음. 계속되면 `index --batch-size 2`(또는 1)로 낮추기 |
| 콘솔에 한글이 깨져 보임(`???` 등) | 터미널 코드페이지가 UTF-8이 아님 | CLI 자체는 UTF-8로 출력하도록 처리돼 있음(`cli.py`); 그래도 깨지면 Windows Terminal 등 UTF-8을 지원하는 터미널 사용 |

## 6. 다음에 볼 문서

- 명령어 전체 옵션, 동작 원리 요약 → [README.md](README.md)
- 왜 2B 모델을 골랐는지, 리랭커/클립 임베딩의 실측 검증 결과, 겪었던 이슈들의 원인 분석 → [PLAN.md](PLAN.md)
