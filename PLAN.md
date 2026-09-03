# Qwen3-VL 기반 동영상 장면 검색 (vidsearch)

## 진행 상황 (최신)

- **완료**: §0 격리 venv 구성(`c:\temp\vidsearch\.venv`, `torch 2.14.0+cu130`, CUDA 인식 확인, 전역 Python은 `2.12.0+cpu` 그대로 보존)
- **완료**: §2 `media.py`(probe/frame/thumb 추출, scdet 경계) — 실제 영상으로 검증, 351프레임/scdet 3개 경계 모두 사전 조사 수치와 일치
- **완료**: `store.py`(인덱스 저장/로드), `segment.py`(구간 병합, 단위 테스트 통과), `cli.py`(`index`/`search` 서브커맨드), `embedder.py`/`reranker.py` 작성
- **완료**: `vidsearch index` 실행 — 351프레임 → `embeddings.npy (351, 2048)`, 벡터 norm 전부 ≈1.0. 대상 영상은 강변 산책로 버스킹 장면(기타·카혼 연주), t≈203s에 아이가 킥보드로 카메라 앞을 가로지르는 순간이 scdet 경계와 일치
- **완료**: §검증 2·3·4·6번 실환경 통과.
  - self-retrieval(킥보드 장면 질의) → t=202s 압도적 1위
  - **리랭커 효과 실측**: 임베더 단독은 실제 장면 vs 없는 장면의 1위 점수 격차가 0.01~0.05로 미미했으나(모달리티 갭), 리랭커 적용 시 실제 장면은 1·2위 격차 0.3125로 벌어지고 없는 장면("눈 내리는 야경")은 전 후보가 0점 이하로 내려감 — 2단계 설계(§Context)가 실제로 유효함을 확인
  - cli.py의 콘솔 UTF-8 출력 버그(cp949 깨짐) 수정 완료
- **완료**: `hf_cache.py` 추가 — 로컬 HF 캐시(`~/.cache/huggingface/hub/models--...`)에 필요한 모델이 전부 있으면 `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`을 자동으로 켜서 `search`가 매번 Hub에 캐시 검증 요청을 보내지 않도록 함(그 요청이 실패하면 나던 "unauthenticated requests" 경고와 간헐적 502가 사라짐). 아직 캐시에 없는 모델이 필요하면 자동으로 온라인 모드로 남아 첫 다운로드는 그대로 동작. `cli.py`의 `cmd_index`/`cmd_search` 양쪽에 연결, 재검증 완료(경고 사라짐, 결과 동일)
- **다음**: `search` 커맨드 §검증 5번(한/영 대조), 7·8번(육안·VRAM) 마무리 (§5는 이후 `ask` 커맨드로 구현 완료 — 아래 참고)
- **완료**: §8 클립 임베딩(동작 질의 지원) 구현 및 실측 검증.
  - `media.extract_clips()`, `store.py`의 `save_clip_index`/`load_clip_index`/`update_manifest`, `cli.py`의 `index --with-motion`/`search`의 클립 채널 병합·`--no-motion` 모두 구현.
  - **구현 중 발견한 문제**: `torchcodec`(비디오 디코딩 백엔드)이 처음엔 없어서 실패 → 설치 후에도 Windows에서 Python 3.8+ 는 PATH만으론 DLL의 간접 의존성을 못 찾아 또 실패 → vcpkg ffmpeg 공유 DLL과, 그 ffmpeg가 `--enable-libnpp`로 빌드되어 있어 CUDA Toolkit v12.8의 NPP DLL까지 필요하다는 걸 `ldd`로 직접 추적 → `config.py`에 `os.add_dll_directory()` 두 경로 등록으로 해결(§리스크에 반영).
  - §검증 9번(구조) 통과: 175클립, `(175, 2048)`, norm≈1.0, `mid_t` 2.0~349.4초 고르게 분포.
  - §검증 10번(내용) — "뛰어가는 사람들" 질의 결과: 1위 t=94s(점수 0.1250, 육안 확인 결과 주황 비니를 쓴 사람이 4초간 화면을 빠르게 가로지름 — 빠른 이동은 맞으나 명확한 전력질주는 아님), 2위 t=203s(킥보드 장면, 0.0625), 3위부터 전부 음수(-0.125 이하)로 급락. §검증4의 "확실히 없음" 기준(0점 이하)과 비교하면 **약한 긍정**: 정적 장면과는 확실히 구별해냈으나 이 영상엔 명확한 전력질주 사례가 없어 완벽한 positive 검증은 아님(§8에서 예상한 그대로).
  - 프레임 인덱스 재사용 확인: 이미 인덱싱된 영상에 `--with-motion`만 추가 실행 시 "Frame index already exists — reusing" 출력, 프레임 재인코딩 없이 클립 인덱스만 생성됨(§8 "기존 인덱스 재사용 필수" 요구사항 충족).
- **완료**: §8 리랭킹을 정지 프레임 → **클립 우선**으로 전환(사용자가 "283초에 실제로 뛰는 사람이 있는데 검색 결과에 안 나온다"고 신고하면서 발견).
  - **원인 진단(실측)**: 회수 단계는 정확했다 — 283초를 덮는 클립이 클립 채널 175개 중 2위(0.5392)로 정확히 잡혔다. 그런데 정지 프레임 리랭킹이 이걸 죽였다: 가장 선명한 뛰는 프레임(283s)조차 리랭커 점수 **0.0**(확신 없음), 실제 검색에 쓰인 프레임(284s)은 **-0.1875**("아니다" 판정). 같은 순간을 클립(비디오) 그대로 리랭킹하면 **0.3125**(확정 매치인 킥보드 장면의 0.375에 필적) — 정지 프레임 하나로는 "큰 걸음"과 "뛰는 동작"을 구별 못 하지만 여러 프레임에 걸친 클립은 구별한다는 게 실측으로 증명됨.
  - **수정**: `segment.py`의 `ScoredFrame`/`Segment`에 `clip_path`/`peak_clip` 필드 추가 → `cli.py` 리랭킹 단계에서 `peak_clip`이 있으면(=그 구간의 최고점이 클립 채널에서 왔으면) 정지 프레임 대신 **클립 자체**로 리랭킹(§7에서 "향후 확장"으로 남겨뒀던 항목을 실제로 승격).
  - **재검증**: "뛰어가는 사람들" 재실행 → 283초 구간이 -0.125 → **0.3125로 복구**, 2위로 재진입. 전체 점수 분포도 개선(0.1875~0.5000, 이전엔 최고 0.125에 나머지 전부 음수).
  - **새로 드러난 한계(솔직히 기록)**: 1위가 283초가 아니라 194초 군중 장면(0.5000)으로 나옴 — 사람 여럿이 뒤섞여 교차하는 장면. 초 단위로 190~198초를 다시 훑었지만 **뛰는 사람은 없음**, 진짜 오탐으로 판단됨. 클립 임베딩·클립 리랭킹 모두 "화면에 움직임이 많다"는 잘 잡아내지만 "그 움직임이 구체적으로 뛰는 동작인가"까지는 덜 정밀하다 — 혼잡한 군중 장면과 실제 동작 질의를 헷갈리는 게 §8의 남은 한계다. 이번 수정으로 재현율(recall)은 확실히 좋아졌지만 정밀도(precision)는 완벽하지 않음.
- **완료**: §5 `explain.py` 구현 — 다만 `search --explain` 플래그가 아니라 **독립 `ask` 서브커맨드**로 승격(사용자 질문 "query도 가능? 어떤 구간내의 사람수?"로 촉발).
  - **설계 변경 이유**: §5의 원안은 "검색 결과 상위 3구간에 대한 근거 설명"이었으나, 실제로 요청된 건 **검색을 거치지 않고 사용자가 지정한 구간에 직접 질문**하는 것이었다. 검색 결과에 종속된 플래그로는 이 용도를 못 담아 `ask <영상> [--start --end] [--question]` 독립 커맨드로 분리했다. `search`(어디에 있나) vs `ask`(여기서 무슨 일이 벌어지나)로 역할이 갈린다.
  - **구현**: `Explainer`가 `Embedder`/`Reranker`와 동일한 load/release 패턴을 따른다(2B+2B+4B 셋 다 10.5GB에 동시 상주 불가는 그대로). 샘플링한 프레임들을 **한 턴에 함께** 넣어(멀티 이미지 메시지) 모델이 프레임 간을 비교하게 했다 — 프레임별로 따로 물으면 "이 중 최대 몇 명"을 못 답한다. `media.parse_timecode()`로 `search` 출력의 `HH:MM:SS` 타임코드를 그대로 구간 인자에 붙여넣을 수 있게 했다.
  - **실측 3건 (전부 정상 동작)**:

    | 대상 | 질문 | 답변 |
    |---|---|---|
    | 190~198초(§8에서 오탐으로 지목된 군중 장면) | 사람 최대 몇 명(기본 질문) | **7명** + 위치 근거 |
    | 200~206초 | 뭘 하고 있나 | 버스킹 연주 + 킥보드 아이 (§Context의 실제 내용과 일치) |
    | 구간 미지정(0~350.89초 전체, 6프레임 균등) | 영상 전체 요약 | "산책로에서 밴드가 연주하는 일상적인 공연 영상" |
  - **드러난 한계(솔직히 기록)**: 장면의 큰 줄기는 맞히지만 **세부 명사가 틀린다** — 카혼을 "카라오케"로(그것도 한 문장에 두 번 중복), 강변 산책로를 "해변가"로 오인했다. 4B Instruct의 묘사 정확도 한계이므로 `ask` 답변의 세부 명사는 신뢰 대상이 아니다. 또한 인원 계수는 **객체 탐지가 아니라 VLM 프롬프팅**이라 사람이 겹치거나 군중이 많아질수록 부정확해진다(7명 수준의 이번 장면은 잘 나왔지만 이걸 정확도 근거로 일반화하면 안 됨). 정확한 계수가 필요하면 §7의 별도 객체 탐지(YOLO 등) 경로가 맞다.
  - 전체 영상 대상 시 351프레임 중 6장만 뽑으므로 **58초 간격** — 전체 요약엔 충분하지만 짧은 순간은 원리상 놓친다. 세부 질문은 `--max-frames`를 올리거나 구간을 좁혀야 한다.
- **완료**: `ask` 런타임 프로파일 실측 + 그 과정에서 **§2 토큰 추정 오류와 §6 `--width` 전제 무효를 발견**(사용자 질문 "Loading weights는 뭐하는 부분인가?"로 촉발).
  - **`Loading weights: 713/713` 진행바의 정체**: `from_pretrained()`가 4B 모델의 713개 텐서(언어 398 + 비전 315)를 구조에 꽂는 단계. 단 **0.23초**에 끝나는데, safetensors가 mmap이라 이 시점엔 매핑만 걸고 디스크를 읽지 않기 때문이다. 직후 확인한 상태가 `device=cpu`, `GPU allocated=0.00GB`. 실제 8.93GB 전송은 그 다음 `.to("cuda")`의 **3.75초**에서 일어난다. **진행바가 100%여도 로딩은 시작 전**이며 체감 대기는 그 뒤에 온다.
  - **단계별 실측(전체 11.89초)**: `import transformers` 3.60s / `.to(cuda)` 3.75s / `generate` 2.40s / `import torch` 1.39s / `from_pretrained` 0.23s / 전처리 0.14s / `store.load_index` 0.03s / 프레임 샘플링 0.00s. **모델과 무관한 순수 임포트가 5초로 전체의 42%** — `cli.py`가 무거운 임포트를 함수 안으로 미뤄둔 설계가 이 수치로 정당화된다(`--help`·인자 오류는 즉시 종료).
  - **피크 GPU 9.72GB** (입력 2726토큰, 6프레임 기준). 실가용 10.5GB에 **여유 0.8GB뿐** — §5에 적어둔 "여유 2.5GB"보다 실제로 더 빠듯하다. `--max-frames`를 6에서 올리면 프레임당 448토큰씩 선형 증가하므로 OOM이 먼저 온다. 현재 기본값 6이 보수적인 이유.
  - **§2 토큰 추정 오류(정정)**: §2는 패치를 28×28로 가정해 "약 144 비주얼 토큰/프레임"이라 적었으나, 두 모델의 `preprocessor_config.json`을 직접 확인하니 **`patch_size=16`, `merge_size=2`** 다. 실측 결과 896×504 프레임은 56×32=1792패치 → 2×2 병합 후 **448토큰/프레임**으로 **추정치의 약 3.1배**. 파이프라인 동작에는 문제가 없다(`MAX_PIXELS`는 픽셀 수 상한이라 패치 크기와 무관하게 유효) — 틀린 것은 예산 계산의 근거 수치다.
  - **§6 "해상도 상향 여지"가 실제로는 무효(중요)**: `--width`를 올려도 `config.MAX_PIXELS`(=28·28·640=501,760px) 상한이 먼저 걸려 **입력이 도로 축소된다**. 프로세서로 직접 측정한 값:

    | `--width` | 입력 크기 | grid (t,h,w) | 패치 | 토큰/프레임 |
    |---|---|---|---|---|
    | 896 | 896×504 | (1, 32, 56) | 1792 | **448** |
    | 1120 | 1120×630 | (1, 32, 58) | 1856 | 464 (상한 걸림) |
    | 1344 | 1344×756 | (1, 32, 58) | 1856 | 464 (동일) |
    | 1920 | 1920×1080 | (1, 32, 58) | 1856 | 464 (동일) |

    896 → 1120은 토큰이 **448 → 464(+3.6%)** 로 거의 안 늘고, 1344·1920은 1120과 **완전히 동일**하다. 즉 §6·§3·§검증8이 전제한 "`--width 1120`으로 작은 객체 대응, 대신 VRAM 마진이 좁아짐"은 **양쪽 다 일어나지 않는다** — 정확도 이득도 VRAM 비용도 없이 ffmpeg 추출 시간과 디스크만 더 쓴다. 해상도를 실제로 올리려면 `--width`가 아니라 **`config.MAX_PIXELS`를 함께 올려야** 하며, 그때 비로소 VRAM 마진 문제가 생긴다(피크가 이미 9.72GB이므로 여지가 크지 않다). 미착수.

## Context

`c:\temp\11106522-hd_1920_1080_30fps.mp4`(1920×1080, 29.97fps, **350.9초**, 10,516 프레임) 같은 영상에서
"해질녘 해변을 걷는 사람" 같은 **자연어(한국어) 질의로 해당 장면의 타임코드를 찾는** 도구가 필요하다.

Qwen3-VL 계열에는 생성형 VLM(`Qwen3-VL-*-Instruct`) 외에 **검색 전용 모델**이 별도로 존재한다:

- `Qwen/Qwen3-VL-Embedding-2B` — 듀얼 타워, 2048차원(MRL 64~2048), text/image/video 입력, 정규화된 벡터 출력
- `Qwen/Qwen3-VL-Reranker-2B` — 싱글 타워 크로스 어텐션, (query, document) 쌍 → 관련도 점수

따라서 **"임베딩 인덱스로 즉시 회수 → 리랭커로 정밀 재정렬"** 2단계 파이프라인을 Qwen3-VL 한 가족으로 구성한다.
결과물은 `c:\temp\vidsearch` 아래 **독립 Python CLI 프로토타입**이며, 추론은 **로컬 GPU(RTX A2000 12GB)** 에서 수행한다.

### 인덱싱이란?

"인덱싱"은 **영상을 검색 가능한 형태로 미리 가공해 디스크에 저장해두는 1회성 사전 작업**이다(`vidsearch index <영상>`).
검색 엔진이 웹페이지를 미리 색인해두는 것과 같은 개념으로, 다음을 한 번만 수행한다:

1. **프레임 추출** — ffmpeg로 영상을 고정 간격(기본 1fps) 정지 이미지로 분해 (§2 `media.py`)
2. **장면 경계 탐지** — scdet으로 컷 전환 지점을 부가 정보로 기록 (구간 병합 시 힌트로만 사용, §"샷 구조" 행 참고)
3. **벡터 변환(임베딩)** — Qwen3-VL-Embedding-2B가 프레임마다 2048차원 벡터를 생성 — "이 장면이 의미적으로 무엇인가"를 벡터 공간의 한 점으로 표현 (§3 `embedder.py`)
4. **저장** — `embeddings.npy` + `meta.jsonl` + `manifest.json`으로 영구 저장 (§ `store.py`)

**인덱싱을 생략하고 매 질의마다 즉석 스캔한다면** 위 1~3단계(프레임 재추출 + 모델 재로드 + 351프레임 재인코딩)를 검색할 때마다 반복해야 하며,
계획상 추정으로 질의 1회당 **1~3분**이 걸린다. 인덱싱을 미리 해두면 검색 시점엔 저장된 벡터와 질의 벡터를 비교하는 것뿐이라 **수십 ms**로 끝난다
(모델 로드 시간은 남지만 프레임 재추출·재인코딩은 사라진다). 이 차이는 같은 영상에 반복 질의하거나 영상이 길어질수록/영상 수가 늘수록 결정적이 되므로,
"즉석 스캔" 방식은 설계 단계에서 제외했다(§검토 과정에서 폐기된 대안).

### 리랭커란?

"리랭커"는 **회수(recall) 단계가 놓친 정밀도를 되짚어 바로잡는 2차 검증 모델**이다(§4 `search`의 4단계).

**임베더(Embedding-2B)의 한계** — 인덱싱 때 만든 351개 벡터는 프레임과 질의를 **각각 독립적으로** 인코딩한 뒤 내적만 비교하는 방식(듀얼 타워/바이 인코더)이다.
미리 다 계산해둘 수 있어 빠르지만(§검증에서 확인한 수십 ms), 프레임과 질의가 서로를 한 번도 "같이 들여다본" 적이 없어 판단이 거칠다.
이 한계는 실측으로 드러났다: 실제 있는 장면과 아예 없는 장면("눈 내리는 야경")의 1위 점수 차이가 **0.01~0.05**밖에 안 났다(모달리티 갭 — text/image 간 절대 유사도가 압축되는 현상).

**리랭커(Reranker-2B)의 방식** — 싱글 타워 **크로스 인코더**로, 질의 텍스트와 후보 프레임 이미지를 **같은 트랜스포머에 함께 넣어** 크로스 어텐션으로 서로를 직접 참조하게 한다.
"이 이미지가 이 질의에 맞나?"를 yes/no 토큰 생성 확률로 직접 판정해 훨씬 정밀하지만, 미리 계산해둘 수 없고 후보 하나하나마다 매번 다시 돌려야 해서 느리다.
그래서 회수 결과 상위 20구간에만(§4, `--rerank-top`) 적용한다 — "빠르지만 대충인 1차 필터 → 느리지만 정확한 2차 검증"의 조합.

**실측 검증 (§검증 6번)** — 같은 두 질의를 리랭커 유무로 비교한 결과, 효과가 명확했다:

| 질의 | 방식 | 1위 점수 | 2위 점수 | 1·2위 격차 |
|---|---|---|---|---|
| 킥보드 아이 (실제 있음, t≈203s) | 임베더만 | 0.4305 | 0.3850 | 0.045 |
| 킥보드 아이 (실제 있음, t≈203s) | **+ 리랭커** | 0.3750 | 0.0625 | **0.3125** |
| 눈 내리는 야경 (없음) | 임베더만 | 0.3974 | 0.3854 | 0.012 |
| 눈 내리는 야경 (없음) | **+ 리랭커** | 0.0000 | -0.1875 | 전 후보 ≤0 |

리랭커를 켜자 실제 장면은 1·2위 격차가 7배로 벌어졌고, 없는 장면은 모든 후보 점수가 0 이하로 떨어져 "전혀 아니다"라는 신호가 뚜렷해졌다.
§6에서 예측한 "2B 임베딩 단독보다 2B 임베딩+2B 리랭커 조합이 더 낫다"는 가설이 실측으로 확인된 셈이다.

### `cap_pixels_per_frame` 경고 조치 (§8 부가 수정)

클립 인코딩 때마다 `[transformers] Qwen3VL video processing does not apply the per-frame pixel cap...` 경고가 떴다.
`transformers` 소스(`video_processing_qwen3_vl.py`)를 직접 확인한 결과: 켜져 있으면(`cap_pixels_per_frame=True`)
프레임 수가 늘어날수록 **프레임당** 픽셀 예산을 그만큼 줄여 **영상 전체** 토큰 비용을 일정하게 유지한다(참조 구현 `qwen-vl-utils`와 동일 동작).
꺼져 있으면(현재 기본값) 프레임마다 이미지 한 장 몫의 예산을 그대로 써서, 프레임이 많을수록 비용이 그만큼 커진다 — v5.22부터 강제로 켜질 예정.

실측(`AutoProcessor`로 우리 클립 하나에 대해 `cap=None/False/True` 비교): **`pixel_values_videos` shape이 완전히 동일**했다 — 640px로 미리 줄여둔 우리 클립(§8) 해상도가 애초에 "capped 안 켠" 상태의 상한보다 낮아서, 이 옵션이 우리 파이프라인에는 실질적으로 아무 영향이 없다.
그래도 경고를 없애고 향후 버전 강제 전환에 대비해 `embedder.py`의 `encode_document()` 호출에 `processing_kwargs={"video": {"cap_pixels_per_frame": True}}`를 명시적으로 추가했다(이미지 입력에는 영향 없음).
`cmd_index --with-motion --force`로 351프레임+175클립 전체 재인덱싱해 재검증: 경고 사라짐, 결과 동일(shape (351,2048)/(175,2048)), 정상 완료.

### 조사로 확인된 사실 (설계에 직접 반영)

| 항목 | 확인 결과 | 설계 영향 |
|---|---|---|
| GPU | RTX A2000 12GB, `compute_cap 8.6`(Ampere). **디스플레이가 1.7GB 상주 → 실가용 ≈ 10.5GB** | bf16 + **2B 모델** 기본. 8B bf16(~20GB)은 단독 적재 불가 (§6의 오프로드 경로로만 가능) |
| 시스템 RAM | 31.8GB (가용 19GB) | 8B CPU 오프로드가 물리적으로 가능 — §6 옵션 B의 근거 |
| 동시 로드 | Embedding-2B ≈ 8GB, Reranker-2B ≈ 8GB | **10.5GB에 동시 상주 불가 → 순차 로드/해제 필수** |
| torch | 전역 Python에 `2.12.0+cpu` 설치됨 (`cuda.is_available()==False`) | 전역 환경은 기존 `torchreid`/`onnxruntime` 작업에 쓰이므로 **건드리지 말고 uv venv로 격리** |
| ffmpeg | 전역 PATH엔 없으나 `C:\dev\vcpkg\installed\x64-windows\tools\ffmpeg\ffmpeg.exe` (6.1, cuda/nvdec/d3d11va, `scdet`/`select`/`thumbnail` 필터 포함) | **절대 경로로 직접 호출**. 프레임 추출·메타 조회 전담 |
| 디코딩 속도 | 전체 351초 CPU 디코딩 ≈ 9초 (약 40배속). `-hwaccel cuda`는 오히려 느림(초기화 오버헤드) | **hwaccel 불필요**, CPU 디코딩으로 충분 |
| 샷 구조 | `select='gt(scene,0.3)'` → **0개**. threshold 0.08에서도 3개(202.6s, 220.5s, 253.6s)뿐 | **샷 분할 단독으로는 인덱스 단위가 안 됨**(351초에 4구간). **고정 fps 샘플링을 1차 단위**로 삼고 scdet 경계는 보조 힌트로만 사용 |
| 디스크 | C: 295GB 여유, HF 캐시 비어 있음 | 모델 신규 다운로드 ~9GB(2B ×2) + 선택적 4B-Instruct ~8GB |
| 기타 | `uv 0.10.8` 사용 가능. flash-attn은 Windows 설치 난이도 높음 | `attn_implementation="sdpa"` 사용 |

---

## 구현 계획

### 0. 격리 환경 구성 (전역 Python 오염 금지)

```powershell
# c:\temp\vidsearch 에서
uv venv --python 3.12
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
uv pip install "transformers>=4.57" "sentence-transformers>=5.4" qwen-vl-utils accelerate pillow numpy
```

- PyTorch 2.12는 cu128 휠이 제거되고 **cu130이 기본**. 드라이버 610.47이므로 cu130 사용, 문제 시 cu126 폴백.
- 게이트: `.venv\Scripts\python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` → `True NVIDIA RTX A2000 12GB` 확인 후 진행.

### 1. 프로젝트 구조

```
c:\temp\vidsearch\
  pyproject.toml
  README.md
  vidsearch\
    __init__.py
    config.py      # FFMPEG_EXE 경로, 모델 ID, 픽셀 예산, 기본 fps
    media.py       # ffprobe 메타 조회 / 프레임 추출 / scdet 경계 추출
    store.py       # 인덱스 저장·로드 (embeddings.npy + meta.jsonl + manifest.json)
    embedder.py    # Qwen3-VL-Embedding-2B 래퍼 (로드/해제 포함)
    reranker.py    # Qwen3-VL-Reranker-2B 래퍼
    explain.py     # Qwen3-VL-4B-Instruct 래퍼 — `ask` 커맨드의 질의응답 생성 (§5)
    segment.py     # 프레임 점수 → 시간 구간 병합
    cli.py         # argparse 엔트리포인트
  data\<video_stem>\
    frames\f_00001.jpg ...   # 임베딩 입력용 (896px)
    thumbs\t_00001.jpg ...   # 결과 표시용 (320px)
    embeddings.npy           # (N, 2048) float16
    meta.jsonl               # {idx, t_sec, frame, thumb}
    manifest.json            # video path/duration/fps/model id/pixel budget
    clips\c_00001.mp4 ...    # --with-motion 사용 시만 (§8, 640px, 4초/2초 스트라이드)
    clip_embeddings.npy      # (M, 2048) float16, --with-motion 사용 시만
    clip_meta.jsonl          # {idx, start_t, end_t, mid_t, clip}, --with-motion 사용 시만
```

### 2. `media.py` — 프레임 추출 (ffmpeg)

`config.FFMPEG_EXE = r"C:\dev\vcpkg\installed\x64-windows\tools\ffmpeg\ffmpeg.exe"` (ffprobe도 동일 폴더)

- **메타 조회**: `ffprobe -v error -show_entries format=duration -show_entries stream=width,height,r_frame_rate -of json`
- **인덱싱용 프레임**: `ffmpeg -v error -i <video> -vf "fps=1,scale=896:-2" -q:v 3 frames/f_%05d.jpg`
  - ~~896×504 → 32×18 패치 → 2×2 병합 후 **약 144 비주얼 토큰/프레임**~~ **(계획 단계의 오산 — 패치를 28×28로 가정했다)**. 실제 `patch_size=16`이라 896×504는 56×32=1792패치 → 2×2 병합 후 **448토큰/프레임**이다(실측, 진행 상황 로그 참고). 장면 검색 정확도 손실이 미미하다는 결론 자체는 §검증 3·4·6번으로 확인됐다.
  - ~~작은 객체·간판 질의가 약하면 `--width 1120`(≈200토큰)까지 상향~~ **(무효 — `MAX_PIXELS` 상한이 먼저 걸려 도로 축소되므로 448→464토큰으로 거의 안 변하고, 1344·1920은 1120과 완전히 동일하다)**. 해상도를 실제로 올리려면 `config.MAX_PIXELS`를 함께 올려야 한다. 측정표는 진행 상황 로그 참고.
  - fps=1 → 351프레임. `--fps` 로 조절(0.5~2 권장).
- **썸네일**: `-vf "fps=1,scale=320:-2"` 별도 패스 (동일 인덱스 정렬)
- **scdet 보조 경계**: `-vf "select='gt(scene,0.08)',showinfo"` → `pts_time` 파싱해 manifest에 기록. 구간 병합 시 경계 넘는 병합을 억제하는 용도로만 사용.
- 프레임 파일명 인덱스 → 타임스탬프는 `t_sec = (i - 1) / fps` 로 계산 (ffmpeg `fps` 필터는 등간격 보장).

### 3. `embedder.py` + `index` 커맨드

```python
model = SentenceTransformer(
    "Qwen/Qwen3-VL-Embedding-2B",
    device="cuda",
    model_kwargs={"torch_dtype": "bfloat16", "attn_implementation": "sdpa"},
    processor_kwargs={"min_pixels": 28*28*64, "max_pixels": 28*28*640},
)
emb = model.encode_document(frame_paths, batch_size=4, normalize_embeddings=True,
                            show_progress_bar=True)   # (N, 2048)
```

- `encode_document()` / `encode_query()` 는 sentence-transformers v5.4+ 가 모델별 instruction 프리픽스를 자동 적용해 준다. 직접 프롬프트를 만들지 말 것.
- 이미지는 **로컬 파일 경로 문자열**로 그대로 전달 가능 (PIL·URL·ndarray도 허용).
- 저장은 `float16` (351×2048×2B ≈ 1.4MB). 검색 시 float32로 승격 후 정규화.
- OOM 시 완화 순서: `batch_size` 4→2→1 → `max_pixels` 축소 → `truncate_dim=1024`(MRL).
  ~~`--width 1120` 사용 시엔 `batch_size=2`부터 시작~~ **(무효 — `--width`를 올려도 `MAX_PIXELS` 상한 때문에 토큰이 거의 안 늘어 VRAM 부담도 늘지 않는다. 부담이 실제로 늘어나는 건 `MAX_PIXELS`를 직접 올릴 때다.)**
- 예상 소요: 351프레임 × 0.1~0.3s ≈ **1~2분** (최초 실행은 모델 다운로드 ~4.5GB 별도).

### 4. `search` 커맨드 — 회수 → 병합 → 리랭킹

**VRAM 제약이 흐름을 결정한다. 두 2B 모델은 절대 동시에 상주시키지 않는다.**

1. **회수(recall)**: Embedding-2B 로드 → `model.encode_query(질의)` (텍스트 전용, 수십 ms) → `scores = emb_f32 @ q` → 상위 **M(기본 150)** 프레임. (2B의 회수 손실을 폭으로 보상 — 근거는 §6)
   - `--bilingual` 옵션: 한국어 질의 + 영어 번역본을 각각 인코딩해 프레임별 **max 점수** 사용. (모델 카드가 영어 instruction에서 1~5% 이득을 명시)
2. **모델 해제**: `del model; gc.collect(); torch.cuda.empty_cache()` — 이 단계를 빠뜨리면 3번에서 OOM.
3. **구간 병합** (`segment.py`): 상위 프레임을 시간순 정렬 → 간격 ≤ `--gap`(기본 2.0초)이면 하나의 구간으로 묶고, scdet 경계를 가로지르면 분리. 구간마다 `{start, end, peak_t, peak_frame, max_score, mean_score}` 산출.
4. **리랭킹**(기본 on, `--no-rerank`로 생략): Reranker-2B 로드 → 각 구간 대표 프레임(peak_frame)으로
   `CrossEncoder("Qwen/Qwen3-VL-Reranker-2B", ...).rank(query, peak_frame_paths)` → 점수로 재정렬.
   - 리랭킹 대상은 상위 20구간으로 제한(2B 크로스 인코더 1패스/구간 ≈ 0.3s → 수 초).
5. **출력**: `HH:MM:SS.mmm  score  thumb경로` 표 + `--json` 옵션. 최상위 구간에 대해
   `ffplay.exe -ss <peak_t> <video>` 명령줄을 함께 출력해 바로 확인 가능하게 한다.

### 5. `ask` — 구간 질의응답 (원안 `--explain`에서 변경, 구현 완료)

`Qwen3-VL-4B-Instruct`(bf16 ≈ 8GB, 단독 로드 시 실가용 10.5GB에 적재 가능하나 여유 2.5GB로 빠듯함)를 올려
프레임 + 질문을 주고 **한국어 답변을 생성**한다. Embedding/Reranker는 벡터·점수만 낼 뿐 텍스트를 만들지 못하므로 이 용도엔 생성형 VLM이 따로 필요하다.

**원안에서 바뀐 점**: 초안은 `search --explain`(검색 상위 3구간의 근거를 설명)이었으나, 실제 요구가 "검색과 무관하게 특정 구간에 직접 질문"이어서 **독립 `ask` 서브커맨드**로 구현했다(진행 상황 로그 참고).

```
vidsearch ask <영상> [--start <t>] [--end <t>] [--question <질문>] [--max-frames N] [--json]
```

- `--start`/`--end` 생략 시 **영상 전체**가 대상(한쪽만 줘도 됨). 초 또는 `HH:MM:SS` 둘 다 허용 — `search` 출력 타임코드를 그대로 붙여넣을 수 있다.
- 구간 내 인덱싱된 프레임에서 최대 `--max-frames`(기본 6)장을 **균등 샘플링해 한 턴에 함께** 투입 → 프레임 간 비교 질문("최대 몇 명")이 가능.
- `--question` 기본값은 인원 수 질문이지만 **카운팅 전용이 아니다** — 임의의 질문이 그대로 전달된다.
- 모델은 `config.py`의 `EXPLAIN_MODEL_ID`로 교체 가능(`Qwen3-VL-2B-Instruct`로 낮출 수 있음).
- 인덱싱된 프레임만 재사용하므로 영상 재디코딩이 없다. 비용은 4B 모델 로드(최초 실행 시 ~8GB 다운로드) + 생성 시간.

### 6. 모델 크기 선택의 근거 — FP8은 선택지가 아니며, 제약은 VRAM이다

**FP8을 쓰지 못하는 것 자체의 영향은 사실상 없다.** 세 겹으로 막혀 있기 때문이다:

1. **하드웨어** — FP8 텐서코어는 sm_89(Ada)/sm_90(Hopper) 이상. A2000은 sm_86이라 FP8 가중치를 올려도 연산은 bf16으로 디퀀트되어 **메모리 이득만 있고 속도 이득은 0**.
2. **런타임** — 공개된 FP8 체크포인트(`RamManavalan/Qwen3-VL-Embedding-8B-FP8`, 커뮤니티)는 vLLM 전용. 본 계획의 sentence-transformers/transformers 스택은 로드 불가.
3. **모델** — Qwen 공식 AWQ/GPTQ(W4A16)는 **Instruct 계열에만** 존재하고 Embedding/Reranker 계열엔 없다. 임베딩 모델은 출력이 벡터 자체라 양자화 잡음이 벡터 공간 기하를 직접 왜곡하며, 검증된 retrieval eval도 없어 리스크가 크다.

따라서 2B를 고른 실제 이유는 FP8 부재가 아니라 **실가용 VRAM 10.5GB**다.

**2B 선택의 정량적 손해 (MMEB-V2, 공식 모델 카드):**

| 모델 | Overall | Video Overall | Video RET | Video MRET |
|---|---|---|---|---|
| Qwen3-VL-Embedding-8B | 77.9 | 66.1 | 57.0 | 53.2 |
| Qwen3-VL-Embedding-2B | 73.4 | 61.1 | 52.3 | **51.6** |

본 과제에 가장 가까운 지표는 **MRET(moment retrieval)** 이고 격차는 **1.6pt**, Video RET은 4.7pt다.
게다가 이 격차는 **Reranker-2B가 메우도록 설계된 구조**다 — 회수 단계 손실을 크로스 어텐션 리랭커가 복구하므로,
"8B 임베딩 단독"보다 "2B 임베딩 + 2B 리랭커"가 더 나을 개연성이 높다.

**VRAM 추가 없이 격차를 메우는 조치 (기본값에 반영):**

- **회수 폭 확대**: 상위 M을 60 → **150**. 2B의 recall@60이 8B보다 낮아도 넓게 건져 리랭커에 넘기면 최종 순위는 보존된다. VRAM 비용 0, 리랭킹 시간만 선형 증가(구간 병합 후 대상은 20구간으로 캡).
- ~~**해상도 상향 여지**: 896px → **1120px**(≈40×20 패치, 200토큰)까지 올려 작은 객체/간판 질의 대응~~ — **실측으로 무효 판정**. `--width`를 올려도 `MAX_PIXELS` 상한이 먼저 걸려 토큰이 448→464로 사실상 그대로다(1344·1920도 1120과 동일). "2B라서 확보한 해상도 여유"라는 이 근거는 **현재 설정에서는 성립하지 않는다** — 여유를 실제로 쓰려면 `MAX_PIXELS`를 올려야 하고, `ask` 피크가 이미 9.72GB인 만큼 여지는 크지 않다. 다만 §6의 나머지 근거(회수 폭 150 확대, MRL 절단 금지, 리랭커로 2B 손실 보전)는 §검증 6번에서 실측으로 확인됐으므로 2B 선택 자체의 결론은 유지된다.
- **MRL 절단 금지**: 2048차원 그대로 저장 (OOM 최후 수단으로만 `truncate_dim` 고려).
- **`--bilingual` 기본 검토**: 모델 카드가 명시한 영어 instruction 이득 1~5%.

**옵션 B — 그래도 8B를 쓰는 현실적 경로 (INT4/FP8이 아니라 CPU 오프로드):**

```python
SentenceTransformer("Qwen/Qwen3-VL-Embedding-8B",
    model_kwargs={"torch_dtype": "bfloat16", "device_map": "auto",
                  "attn_implementation": "sdpa"})
```

- RAM 19GB 가용이므로 8B bf16(16GB)을 GPU 10.5GB + 시스템 RAM에 분산 적재 가능.
  단, 실가용이 12GB가 아닌 10.5GB이므로 **CPU로 넘어가는 레이어가 ~4GB가 아니라 ~5.5GB**로 늘어난다 —
  오프로드 레이어는 순전파마다 PCIe 전송이 필요해 오프로드량에 비례해 느려진다.
- 인덱싱은 오프라인 배치라 느려도 무방: 351프레임 **20~30분 이상**(2B 대비 15~20배, 위 오프로드 증가분 미반영 시 과소추정).
- 검색 시 `encode_query`는 텍스트 전용이라 연산은 가볍지만, **매 검색마다 재적재하면 로드에만 1~2분**이 든다.
  → 옵션 B는 **상주 프로세스(`vidsearch repl` / 로컬 서버 모드)가 전제**이며, 이를 함께 구현해야 실용적이다.
- Reranker는 8B와 동시 적재가 불가능하므로 옵션 B에서는 **Reranker-2B 유지**.
- 채택 기준: §검증 3·4번(self-retrieval, negative)에서 2B가 만족스러우면 옵션 B는 구현하지 않는다.
  `config.py`의 모델 ID와 `device_map` 만 바꾸면 전환되도록 코드 경로는 열어둔다.

### 7. 향후 확장 (이번 범위 밖, 구조만 열어둠)

- **다중 영상**: `data/` 아래 영상별 인덱스가 분리되어 있으므로, `manifest.json`을 모아 로드하는 것만으로 라이브러리 전역 검색으로 확장 가능. 규모가 커지면 numpy 브루트포스 → faiss 교체.
- ~~클립을 리랭커에도 직접 투입~~ — **구현 완료, §8 참고**(클립 우선 리랭킹으로 전환됨).
- **군중/혼잡 장면과 동작 질의 구별**: §8 재검증에서 드러난 남은 한계 — 클립 임베딩·클립 리랭킹이 "움직임이 많다"는 잘 잡지만 "그 움직임이 질의한 구체적 동작(뛰기 등)인가"까지는 덜 정밀해 혼잡한 군중 장면을 오탐할 수 있다. **부분 진전**: §5 `ask`로 의심 구간에 대해 VLM에게 직접 되물을 수 있게 됐다(실제로 오탐 지목된 190~198초에 "사람 몇 명"을 물어 7명이라는 답을 받음). 다만 이는 **사람이 수동으로 확인하는 경로**일 뿐, 검색 파이프라인이 자동으로 오탐을 거르는 단계는 여전히 미착수다. 자동화하려면 리랭킹 후 상위 후보에 `ask`를 걸어 답변으로 재필터링해야 하는데, 4B 모델 로드가 검색마다 추가되는 비용 문제가 남는다.
- **정확한 인원 계수**: `ask`의 카운팅은 VLM 프롬프팅이라 군중이 겹치면 부정확하다(§5 한계 참고). 신뢰할 수 있는 계수가 필요하면 별도 객체 탐지 모델(YOLO 등)로 프레임별 bbox를 세는 경로가 맞다 — 이 저장소 범위 밖.

### 8. 동작(모션) 질의 지원 — 클립 임베딩 (§7에서 승격, 사용자 질문 "뛰어가는 사람들을 검색하려면?"으로 촉발)

**배경** — 정지 프레임 하나로는 "걷는 사람"과 "뛰는 사람"이 우연히 비슷한 자세로 찍히면 구별이 약하다.
필요한 건 여러 프레임에 걸친 **움직임 자체**를 모델이 보게 하는 것 — 그게 클립(짧은 비디오) 임베딩이다.

**API 확인 (설치된 라이브러리 소스를 직접 읽어 검증, 추측 아님):**

- `sentence_transformers/base/modality.py`의 `is_video_url_or_path()` — `.mp4` 등 확장자를 가진 파일 경로 문자열을 자동으로 video 모달리티로 인식한다. 즉 **`embedder.encode_documents(clip_mp4_paths)`가 지금 코드 그대로 동작** — 새 API 학습이나 별도 클래스 불필요, 이미지 경로를 넣던 자리에 클립 경로만 넣으면 된다.
- `transformers/models/qwen3_vl/video_processing_qwen3_vl.py` 기본값: `fps=2`, `min_frames=4`, `max_frames=768`. 4초 클립이면 자동으로 약 8프레임을 샘플링 — 1차 구현에서는 `processing_kwargs` 튜닝 없이 라이브러리 기본값을 그대로 쓴다.
- `CrossEncoder`(리랭커)도 같은 `base/modality.py`를 공유하므로 `reranker.rank(query, clip_paths)`도 이론상 동작 — 1차 구현 범위에서는 쓰지 않고 §7 향후 확장으로 남긴다.
- ~~비디오 디코딩 백엔드(`av==18.1.0`)는 §0에서 transformers 의존성으로 이미 설치돼 있다 — 추가 설치 불필요~~ **(오판, 실제로는 아래 참고)**: `av` 패키지가 있어도 transformers의 Qwen3-VL 비디오 프로세서는 `torchcodec` 또는 `torchvision.io.read_video`(우리 torchvision 0.29에서는 제거됨)만 지원한다. `pip install torchcodec` 필요 + Windows에서 추가 조치 필요(아래 리스크 참고).

**설계:**

1. `media.py`에 `extract_clips(video_path, out_dir, window_sec=4.0, stride_sec=2.0, width=640)` 추가 — 기존 `extract_frames`/`extract_thumbs`와 같은 ffmpeg 서브프로세스 패턴.
   `ffmpeg -ss <start> -t <window> -i <video> -vf scale=<width>:-2 -an -c:v libx264 -preset veryfast clip_%05d.mp4`
   - `width=640`(프레임용 896보다 낮음): 동작 인식은 정밀한 공간 해상도보다 시간축 정보가 중요하고, 클립 수(≈175개, 351초 기준)가 많아 인코딩·디스크 비용을 눌러야 한다.
   - 4초/2초 스트라이드(50% 오버랩) 기본값 — 짧으면 동작을 못 담고 길면 시간 특정이 흐려지는 절충점. `--clip-window`/`--clip-stride`로 조절 가능하게 노출.
2. `store.py`에 `save_clip_index`/`load_clip_index` 추가(새 모듈 없이 기존 "인덱스 영속화" 책임에 병기) — `clip_embeddings.npy (M, 2048)`, `clip_meta.jsonl {idx, start_t, end_t, mid_t, clip}`.
3. `cli.py`:
   - `index --with-motion`(기본 off — 무거운 부가 단계는 opt-in이라는 §5와 같은 원칙) — 프레임 인덱싱 뒤 `extract_clips` → `embedder.encode_documents(clip_paths, batch_size=2)`로 클립 인덱스도 저장.
   - **기존 인덱스 재사용 필수**: 현재 `cmd_index`의 `if store.index_exists(out_dir) and not args.force: return`은 프레임 인덱스 유무만 보고 함수 전체를 끝내버린다. 이미 프레임 인덱싱된 영상에 `--with-motion`만 추가해도 다시 돌 수 있도록, **프레임 인덱스 존재 여부와 클립 인덱스 존재 여부를 따로 체크**하게 고친다: 프레임 인덱스가 있으면(그리고 `--force` 아니면) 프레임 추출·인코딩은 건너뛰고, `--with-motion`이면서 클립 인덱스가 없으면(또는 `--force`) 클립 인덱스만 새로 만든다. 즉 `vidsearch index <영상> --with-motion`을 이미 인덱싱된 영상에 다시 실행하면 **클립 인덱스만 추가되고 프레임 재인코딩은 없다.**
   - `search`: `clip_embeddings.npy`가 존재하면 자동으로 클립 채널도 같은 질의 벡터로 점수 계산 → 클립의 `mid_t`와 가장 가까운 **기존 프레임 썸네일**(1fps, 이미 있음)을 대표 이미지로 삼아 `ScoredFrame`으로 변환(새 썸네일 추출 불필요) → 프레임 채널의 `scored` 리스트와 합쳐서 같은 `merge_segments()`에 전달. `--no-motion`으로 끌 수 있음(`--no-rerank`와 대칭).
4. ~~리랭킹은 1차 구현에서 변경하지 않는다~~ **(정정: 실측 후 변경)** — 초기엔 정지 프레임 리랭킹을 유지할 계획이었으나, 실제 신고된 버그(283초 뛰는 사람이 검색에 안 나옴)를 진단하는 과정에서 정지 프레임 리랭킹이 진짜 동작 장면을 죽인다는 게 실측으로 드러나 **클립 우선 리랭킹**으로 전환했다 — 구간의 최고점이 클립 채널에서 왔으면(`peak_clip` 존재) 정지 프레임 대신 클립 자체를 리랭커에 넣는다. 근거와 수치는 진행 상황 로그 참고.

**스코어 스케일 주의**: 프레임 채널과 클립 채널은 같은 임베딩 공간이지만 인코딩 경로가 달라(정지 이미지 vs 다중 프레임 비디오) 절대 스케일이 완전히 같다고 보장되지 않는다. 리랭킹이 켜져 있으면 회수 단계 점수는 후보 선별용일 뿐 최종 표시 점수는 리랭커가 다시 매기므로 영향이 제한적이지만, `--no-rerank` 상태에서는 이 차이가 그대로 드러날 수 있다 — §검증에 반영.

**이 샘플 영상에 대한 솔직한 상황**: §7 초안 작성 시 "동작 질의는 약하다"의 근거는 육안 스팟체크(351장 중 20장)였다 — 사용자가 지적했듯 1fps라 1초 미만의 순간은 놓칠 수 있어 이것만으론 "뛰는 장면이 없다"를 증명하지 못한다.
더 강한 근거는 이미 실행해둔 scdet 전체 타임라인 스캔(§"샷 구조" 행 — **원본 10,516프레임 전부**를 훑음)이다: 351초 전체에서 급격한 화면 변화가 단 3번(202.6s/220.5s/253.6s, 전부 킥보드 아이 근접통과와 일치)뿐이었다. 화면을 크게 가로지르는 빠른 움직임이 있었다면 scdet도 반응했을 가능성이 높다. 다만 scdet은 픽셀 단위 프레임 차분이라, 화면 점유율이 작은(배경에서 멀리 작게 뛰는) 움직임은 못 잡을 수 있어 완전한 반증은 아니다.

**검증 방향**: 확실한 positive 사례가 보장된 별도 영상 없이, 구현 후 실제로 이 영상에 `"뛰어가는 사람들"` 질의를 던져 클립 인덱스가 무엇을 찾아내는지를 그 자체로 확인 근거로 삼는다 — 여태 못 본 순간을 찾아내면 긍정 검증, 전부 낮은 점수로 나오면(§검증4 negative 패턴과 동일 형태) 최소한 "찾을 게 없을 때 낮게 나온다"는 음성 검증은 된다. 확실한 positive 사례가 필요하면 실행 단계에서 실제 뛰는 장면이 있는 별도 영상을 추가해도 된다.

---

## 검증 방법

1. **환경 게이트**: `.venv\Scripts\python -c "import torch;print(torch.cuda.is_available())"` → `True`.
   전역 `python -c "import torch;print(torch.__version__)"` 가 여전히 `2.12.0+cpu` 인지 확인(격리 성공 증명).
2. **추출 검증**: `python -m vidsearch index c:\temp\11106522-hd_1920_1080_30fps.mp4` 후
   `data/11106522-hd_1920_1080_30fps/frames` 파일 수 == **351**, `embeddings.npy.shape == (351, 2048)`,
   모든 벡터 norm ≈ 1.0.
3. **정합성(sanity) 검증**: 인덱싱된 프레임 중 임의 3장을 골라 눈으로 내용을 확인한 뒤,
   그 내용을 그대로 질의로 넣어 **해당 타임코드가 1위**로 나오는지 확인(self-retrieval). 실패하면 파이프라인 결함.
4. **음성(negative) 검증**: 영상에 없는 장면("눈 내리는 도심 야경" 등)을 질의해 최고 점수가 눈에 띄게 낮게(상대적으로) 나오는지 확인 → 점수 컷오프 기본값 결정.
5. **한국어/영어 대조**: 동일 의미의 한/영 질의 결과가 유사한 구간을 가리키는지 비교. 차이가 크면 `--bilingual`을 기본 on으로 전환.
6. **리랭커 효과 측정**: 동일 질의를 `--no-rerank` / 기본(rerank) 로 각각 실행해 상위 5구간 순위 변화를 기록.
7. **육안 최종 확인**: 출력된 `ffplay -ss <t>` 명령을 실행해 실제 프레임이 질의와 맞는지 확인.
8. **VRAM 확인**: 검색 중 별도 콘솔에서 `nvidia-smi -l 1` — 총 사용량(디스플레이 1.7GB 포함) 피크가 **~11GB**(물리 한계 12GB에 여유 버퍼) 아래에 머무는지, 단계 전환 시 메모리가 실제로 해제되는지 관찰.
   ~~`--width 1120` 사용 시 이 마진이 2.5GB로 좁아지므로~~ **(무효 — `--width`는 `MAX_PIXELS` 상한 때문에 VRAM에 사실상 영향이 없다. 마진을 실제로 좁히는 건 `MAX_PIXELS` 상향이며, 그 경우 특히 주의 깊게 확인하고 초과 시 `batch_size`를 낮춘다.)**
9. **클립 인덱싱 검증(§8)**: `vidsearch index <영상> --with-motion` 후 `clip_embeddings.npy.shape == (M, 2048)`, `clip_meta.jsonl`의 `mid_t`가 영상 길이 내에 고르게 분포하는지 확인.
10. **동작 질의 검증(§8)**: `search <영상> "뛰어가는 사람들"` 실행 후 상위 결과의 `peak_thumb`를 직접 열어 실제로 뛰는 사람이 보이는지 확인. 보이면 긍정 검증 완료. 안 보이면 최고 점수가 §검증4의 negative 사례들과 비슷한 수준으로 낮은지 확인해 "찾을 게 없을 때 낮게 나온다"는 음성 검증으로 대체(§8 "이 샘플 영상에 대한 솔직한 상황" 참고 — 이 영상엔 확실한 positive 사례가 없을 수 있음).
11. **`ask` 검증(§5)**: 세 경로를 각각 실행해 (a) 구간 지정(`--start 190 --end 198`), (b) 임의 질문 + `HH:MM:SS` 입력, (c) 구간 미지정(영상 전체) 이 모두 동작하는지 확인. 각 실행의 `Sampling ...` 출력으로 **샘플링 범위·프레임 수가 의도대로인지** 먼저 검증하고(전체 지정 시 0~duration 전 구간에 고르게 퍼지는지), 그 다음 답변 내용을 §Context에 기록된 실제 영상 내용과 대조한다. 답변의 **세부 명사는 틀릴 수 있으므로**(§5 한계) 장면의 큰 줄기가 맞는지를 기준으로 판정한다.
12. **`ask` 런타임 프로파일(§5)**: 단계별 소요를 재서 `from_pretrained`(mmap, GPU 0GB)와 `.to(cuda)`(실제 전송)를 **분리 확인**하고, `torch.cuda.max_memory_allocated()`로 피크를 잰다. 실측 기준선은 전체 11.89초 / 피크 9.72GB(6프레임·2726토큰) — 실가용 10.5GB에 여유 0.8GB뿐이므로 `--max-frames` 상향 시 여기부터 확인한다.
13. **토큰 예산 검증**: 프로세서만 단독으로 돌려(모델 로드 불필요) `image_grid_thw`와 `pixel_values.shape`로 **실제 토큰/프레임**을 확인한다 — §2의 144토큰 추정이 3.1배 틀렸던 게 이 방법으로 드러났다. `--width`나 `MAX_PIXELS`를 건드릴 때마다 이 표를 다시 뽑아 상한에 걸리는 지점을 먼저 확인할 것.

## 리스크 / 주의

- **전역 Python 오염**: 기존 `torch 2.12.0+cpu` + `torchreid` + `onnxruntime` 환경은 face-blur 등 기존 작업용이다. 반드시 `c:\temp\vidsearch\.venv` 안에서만 설치한다.
- **동시 로드 OOM**: 2B 두 개를 같이 올리면 실가용 10.5GB를 넘는다. 단계 사이 `empty_cache()` 누락이 가장 흔한 실패 지점.
- **VRAM 여유 착시**: `nvidia-smi` 기준 디스플레이가 이미 1.7GB를 점유 중이다. 예산 계산은 12GB가 아니라 **10.5GB** 기준으로 한다.
- **모델 다운로드**: 최초 실행 시 ~9GB(+선택 8GB) 다운로드가 발생하므로 인덱싱 첫 실행 시간이 길다.
- **단일 샷 영상 특성**: 이 영상은 사실상 컷이 없어 인접 프레임 간 유사도가 매우 높다. 구간 병합 파라미터(`--gap`)와 점수 컷오프를 이 영상 기준으로 튜닝한 뒤, 컷이 많은 영상에서 재검증이 필요하다.
- **클립(비디오) 디코딩의 DLL 의존성(§8)**: `torchcodec`은 vcpkg FFmpeg 공유 DLL(`C:\dev\vcpkg\installed\x64-windows\bin`)에 링크되고, 그 FFmpeg가 `--enable-libnpp`로 빌드돼 있어 CUDA Toolkit v12.8의 NPP DLL(`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin`)까지 간접 의존한다. Windows + Python 3.8+ 에서는 PATH만으론 이런 간접 의존성을 못 찾으므로 `os.add_dll_directory()`로 명시 등록해야 한다(`config.py`에 반영 완료). **다른 머신으로 옮기면 CUDA Toolkit 버전(v12.8)이 다르거나 vcpkg 경로가 다를 수 있어 이 두 경로를 다시 맞춰야 할 수 있다** — 이미지 전용 사용(프레임 인덱싱/검색)은 이 의존성과 무관하게 그대로 동작한다.

## 참고 자료

- [Qwen/Qwen3-VL-Embedding-2B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) — 모델 카드, MRL·instruction 형식
- [Qwen/Qwen3-VL-Embedding-8B](https://huggingface.co/Qwen/Qwen3-VL-Embedding-8B) — §6의 MMEB-V2 2B/8B 대조표 출처
- [Qwen/Qwen3-VL-Reranker-2B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) / [Reranker-8B](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B)
- [RamManavalan/Qwen3-VL-Embedding-8B-FP8](https://huggingface.co/RamManavalan/Qwen3-VL-Embedding-8B-FP8) — 커뮤니티 FP8(vLLM 전용), 본 스택에서는 사용 불가
- [kaitchup/Qwen3-VL-2B-Instruct-W4A16](https://huggingface.co/kaitchup/Qwen3-VL-2B-Instruct-W4A16) — INT4는 Instruct 계열에만 존재함을 보여주는 예
- [QwenLM/Qwen3-VL-Embedding (GitHub)](https://github.com/QwenLM/Qwen3-VL-Embedding) — 네이티브 `Qwen3VLEmbedder`/`Qwen3VLReranker` API, 픽셀·fps 파라미터 기본값
- [Multimodal Embedding & Reranker Models with Sentence Transformers](https://huggingface.co/blog/multimodal-sentence-transformers) — `encode_query`/`encode_document`, `CrossEncoder.rank`, `processor_kwargs`, VRAM 가이드
- [Qwen3-VL-Embedding 논문 (arXiv 2601.04720)](https://arxiv.org/abs/2601.04720)
- [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) — Text–Timestamp Alignment, Interleaved-MRoPE
- [PyTorch 2.12 Release Blog](https://pytorch.org/blog/pytorch-2-12-release-blog/) — cu128 제거, cu130 기본화
