# vidsearch

Qwen3-VL(`Embedding-2B` + `Reranker-2B`)로 로컬 영상 속 장면을 자연어(한국어/영어)로 검색하는 CLI 프로토타입.
전부 로컬 GPU에서 돌아가며, 영상 파일이 외부로 나가지 않는다.

설계 배경과 실측 검증 과정(왜 2B를 골랐는지, 리랭커가 왜 필요한지, 클립 임베딩으로 동작 질의를 어떻게 잡았는지 등)은 [PLAN.md](PLAN.md)에 자세히 기록돼 있다.

## 요구 사항

- NVIDIA GPU (VRAM 10GB 이상 권장 — 2B 모델 2개를 순차 로드)
- Python 3.12, [uv](https://github.com/astral-sh/uv)
- ffmpeg (`config.py`의 `FFMPEG_EXE`/`FFPROBE_EXE` 경로를 실제 설치 위치에 맞게 수정)
- (동작 질의용 클립 인덱싱을 쓸 경우) `torchcodec` + FFmpeg 공유 라이브러리를 OS가 찾을 수 있어야 함.
  Windows에서는 `config.py`의 `_DLL_SEARCH_DIRS`에 vcpkg FFmpeg `bin` 폴더와 CUDA Toolkit `bin` 폴더 경로가 하드코딩돼 있다 — 다른 머신에서는 이 경로를 맞게 바꿔야 한다.

## 설치

```powershell
uv venv --python 3.12
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install "transformers>=4.57" "sentence-transformers>=5.4" qwen-vl-utils accelerate pillow numpy
uv pip install torchcodec   # --with-motion(동작 질의)을 쓸 경우에만 필요
```

최초 실행 시 Qwen3-VL-Embedding-2B / Qwen3-VL-Reranker-2B 가중치가 Hugging Face Hub에서 자동 다운로드된다(각 ~4.5GB).
이후 실행부터는 로컬 캐시가 감지되면 자동으로 오프라인 모드로 전환돼 네트워크를 타지 않는다(`hf_cache.py`).

## 사용법

### 1. 인덱싱 — `vidsearch index <영상>`

영상을 검색 가능한 형태로 미리 가공해 `data/<영상이름>/`에 저장하는 1회성 작업.

```powershell
.venv\Scripts\python.exe -m vidsearch index "c:\videos\sample.mp4"
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--fps` | 1.0 | 프레임 샘플링 간격(초당) |
| `--width` | 896 | 인덱싱용 프레임 폭(px) |
| `--batch-size` | 4 | 프레임 인코딩 배치 크기 |
| `--force` | off | 이미 인덱스가 있어도 다시 만듦 |
| `--with-motion` | off | **동작/모션 질의**("뛰어가는 사람들" 등)를 위한 클립(짧은 비디오) 임베딩 인덱스도 함께 생성. 이미 프레임 인덱스가 있는 영상에 이 옵션만 추가로 다시 실행하면 프레임은 재사용하고 클립 인덱스만 추가된다 |
| `--clip-window` | 4.0 | 클립 윈도우 길이(초) |
| `--clip-stride` | 2.0 | 클립 슬라이딩 간격(초, 50% 오버랩이 기본) |
| `--clip-width` | 640 | 클립 프레임 폭(px) |
| `--clip-batch-size` | 2 | 클립 인코딩 배치 크기 |

### 2. 검색 — `vidsearch search <영상> <질의>`

```powershell
.venv\Scripts\python.exe -m vidsearch search "c:\videos\sample.mp4" "파란 옷을 입은 아이가 뛰어간다"
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--query-en` | - | 질의의 영어 번역(`--bilingual`과 함께 사용) |
| `--bilingual` | off | 한국어+영어 질의를 각각 인코딩해 프레임별 최고 점수 사용 |
| `--recall` | 150 | 임베딩 회수 단계에서 추릴 상위 후보 수 |
| `--gap` | 2.0 | 구간 병합 시 허용 간격(초) |
| `--rerank-top` | 20 | 리랭킹 대상 상위 구간 수 |
| `--no-rerank` | off | 리랭커 생략(빠르지만 부정확) |
| `--no-motion` | off | 클립 인덱스가 있어도 동작 채널 무시 |
| `--top` | 10 | 출력할 결과 개수 |
| `--json` | off | 표 대신 JSON 출력 |

출력 예시:

```
          time     score  thumb
  00:03:22.000    0.3750  C:\...\data\sample\thumbs\t_00203.jpg

ffplay -ss 202.00 "c:\videos\sample.mp4"
```

- **time**: 영상 내 타임코드
- **score**: 관련도 점수(리랭커 사용 시 대략 -1~1, 0 이상이면 유의미한 매치)
- **thumb**: 해당 시점 썸네일 이미지 경로 — VSCode나 탐색기로 바로 열어 확인 가능

### 3. 실제 장면 확인

**A. 썸네일을 직접 열기** — `thumb` 경로 또는 `data/<영상이름>/thumbs/` 폴더 안 썸네일을 직접 열람.

**B. 영상에서 그 지점 재생** — 출력 맨 아래 `ffplay ...` 명령을 실행. `ffplay`가 PATH에 없으면 전체 경로로 실행:

```powershell
& "C:\dev\vcpkg\installed\x64-windows\tools\ffmpeg\ffplay.exe" -ss 202.00 "c:\videos\sample.mp4"
```

## 동작 원리 요약

1. **인덱싱**: ffmpeg로 프레임(1fps)을 추출하고 Qwen3-VL-Embedding-2B로 벡터화해 저장 — 검색 때마다 영상을 다시 처리하지 않기 위함.
2. **회수(recall)**: 질의를 벡터화해 저장된 프레임 벡터와 코사인 유사도 비교, 상위 후보 추출. 빠르지만 판단이 거칠다(모달리티 갭).
3. **구간 병합**: 시간상 가까운 후보 프레임들을 하나의 "장면 구간"으로 묶는다.
4. **리랭킹**: 상위 구간의 대표 프레임(또는 그 구간이 클립 채널에서 왔다면 클립 자체)을 Qwen3-VL-Reranker-2B 크로스 인코더에 넣어 정밀 재채점 — "빠르지만 대충인 1차 필터 → 느리지만 정확한 2차 검증" 구조.
5. **(선택) 동작 채널**: `--with-motion`으로 만든 클립 임베딩 인덱스가 있으면, 정지 프레임만으로는 판단하기 어려운 동작 질의("뛰어가는", "넘어지는")도 다중 프레임 클립으로 인식해 회수 단계에 함께 반영한다.

자세한 설계 근거, 실측 수치, 겪었던 이슈와 해결 과정은 [PLAN.md](PLAN.md) 참고.
