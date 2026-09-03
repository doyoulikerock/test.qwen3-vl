# vidsearch

Qwen3-VL(`Embedding-2B` + `Reranker-2B`)로 로컬 영상 속 장면을 자연어(한국어/영어)로 검색하고,
`4B-Instruct`로 특정 구간에 대해 자유롭게 질문하는 CLI 프로토타입.
전부 로컬 GPU에서 돌아가며, 영상 파일이 외부로 나가지 않는다.

설계 배경과 실측 검증 과정(왜 2B를 골랐는지, 리랭커가 왜 필요한지, 클립 임베딩으로 동작 질의를 어떻게 잡았는지 등)은 [PLAN.md](PLAN.md)에 자세히 기록돼 있다.
**새 PC에서 처음 설치한다면 [SETUP.md](SETUP.md)부터 볼 것** — 이 프로젝트는 개발 머신 경로가 일부 하드코딩돼 있어 clone만으로는 바로 안 돌아간다.

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
`ask` 커맨드를 처음 쓰면 Qwen3-VL-4B-Instruct(~8GB)가 추가로 내려받아진다.
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

### 3. 질의응답 — `vidsearch ask <영상> [--start <t> --end <t>]`

이미 인덱싱된 영상에 대해 Qwen3-VL-4B-Instruct에게 자연어로 아무거나 물어본다.
`search`가 "이 장면이 어디 있나"(검색)를 답한다면, `ask`는 "여기서 무슨 일이 벌어지나"(질의응답)를 답한다.
대상 구간의 인덱싱된 프레임 중 최대 `--max-frames`개를 균등 샘플링해 **한 번에 함께** 모델에 넣으므로, 프레임 사이를 비교하는 질문("최대 몇 명")도 가능하다.

```powershell
# 구간 지정 없이 — 영상 전체에서 고르게 샘플링
.venv\Scripts\python.exe -m vidsearch ask "c:\videos\sample.mp4" --question "이 영상을 한 문장으로 요약해줘."

# 특정 구간만 (기본 질문 = 사람 수 세기)
.venv\Scripts\python.exe -m vidsearch ask "c:\videos\sample.mp4" --start 190 --end 198

# HH:MM:SS 형식도 그대로 사용 가능. 한쪽만 줘도 된다(여기부터 끝까지)
.venv\Scripts\python.exe -m vidsearch ask "c:\videos\sample.mp4" --start 00:03:10 --question "이 사람들이 뭘 하고 있어?"
```

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--start` | 영상 시작 | 구간 시작 — 초(`190`) 또는 `HH:MM:SS(.ms)` / `MM:SS`. `search` 출력의 타임코드를 그대로 붙여넣어도 된다 |
| `--end` | 영상 끝 | 구간 끝 (같은 형식). `--start`/`--end` 둘 다 생략하면 **영상 전체**가 대상 |
| `--question` | 사람 수 세기 질문 | 모델에게 던질 질문. **아무 질문이나 가능** — 기본값이 사람 수일 뿐 카운팅 전용 기능이 아니다 |
| `--max-frames` | 6 | 구간에서 샘플링할 최대 프레임 수(많을수록 정확하지만 느리고 VRAM을 더 씀) |
| `--json` | off | 답변 대신 JSON(구간·샘플링 시각·질문·답변) 출력 |

출력 예시:

```
Sampling 6 frame(s) from 190.00s-198.00s: ['190.0s', '191.0s', '192.0s', '193.0s', '194.0s', '195.0s']

7명

이 프레임들에서 동시에 보이는 사람들의 최대 수는 7명이며, ...
```

> **주의 — 이건 객체 탐지가 아니다.** VLM에게 "몇 명이야?"라고 텍스트로 물어보는 방식이라, 사람이 겹치거나 군중이 많아질수록 카운트가 부정확해진다.
> 정확한 인원 계수가 필요하면 별도 객체 탐지 모델(YOLO 등)을 붙이는 편이 신뢰도가 높다.
> 또한 이 커맨드는 세 번째 모델(4B-Instruct, ~8GB)을 쓰므로 **최초 실행 시 별도 다운로드**가 발생한다. 다른 모델과 동시에 올라가지 않도록 순차 로드/해제한다.

### 4. 실제 장면 확인

**A. 썸네일을 직접 열기** — `thumb` 경로 또는 `data/<영상이름>/thumbs/` 폴더 안 썸네일을 직접 열람.
검색 없이 특정 시점을 바로 찾아보고 싶다면, 파일명의 5자리 인덱스(`t_00203.jpg` → 203)로 계산할 수 있다:
`t_sec = (인덱스 - 1) / fps` — 기본값(`--fps 1.0`)이면 `t_00203.jpg` = 202초. `frames/`(임베딩용, 896px)와 `thumbs/`(표시용, 320px)는 같은 인덱스로 정렬돼 있고, `--with-motion`으로 만든 `clips/c_00142.mp4` 같은 클립 파일의 정확한 시작·끝 시각은 `data/<영상이름>/clip_meta.jsonl`에서 확인할 수 있다.
검색 없이 특정 시점을 바로 찾아보고 싶다면, 파일명의 5자리 인덱스(`t_00203.jpg` → 203)로 계산할 수 있다:
`t_sec = (인덱스 - 1) / fps` — 기본값(`--fps 1.0`)이면 `t_00203.jpg` = 202초. `frames/`(임베딩용, 896px)와 `thumbs/`(표시용, 320px)는 같은 인덱스로 정렬돼 있고, `--with-motion`으로 만든 `clips/c_00142.mp4` 같은 클립 파일의 정확한 시작·끝 시각은 `data/<영상이름>/clip_meta.jsonl`에서 확인할 수 있다.

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
6. **구간 질의(`ask`)**: 위 1~5는 "어디에 있나"를 찾는 **검색**이다. 반면 `ask`는 검색을 거치지 않고 사용자가 지정한 구간의 프레임을 생성형 VLM(4B-Instruct)에 직접 넣어 **자연어 답변을 생성**한다 — Embedding/Reranker는 벡터·점수만 내놓을 뿐 텍스트를 만들지 못하므로 별도 모델이 필요하다.

자세한 설계 근거, 실측 수치, 겪었던 이슈와 해결 과정은 [PLAN.md](PLAN.md) 참고.
