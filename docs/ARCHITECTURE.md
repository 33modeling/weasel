# ARCHITECTURE — 설계 구조

이 문서는 레포의 실제 소스를 읽고 작성했다. 모든 수치·기본값·경로 변수명은 코드에서
확인한 값이며, README/SETUP.md의 서술과 구현이 다른 부분은 본문과 마지막 절
"문서-구현 불일치(실측)"에 명시했다.

---

## 0. 레포 성격 판정과 규모(실측)

### 0.1 성격

**연구 재현용 실험 코드 + 클러스터 운영 스크립트 모음**이다. 라이브러리도 아니고
배포 가능한 도구도 아니다. 근거는 다음과 같다.

- `weasel/` 패키지는 `setup.py`/`pyproject.toml`이 없다. 설치 대상이 아니라
  `python -m weasel.<module>` 형태로 레포 루트에서 실행하는 스크립트 묶음이다.
- 모든 모듈이 `argparse` CLI + `main()` 구조이며, 다른 코드가 import해서 쓰는
  공개 API가 없다. `weasel/__init__.py`는 docstring 한 줄(1 LOC)뿐이다.
- 테스트가 하나도 없다. `tests/` 디렉터리, `test_*.py`, `conftest.py`, CI 설정
  (`.github/`) 모두 존재하지 않는다.
- `.gitignore`가 `*.json`, `*.jsonl`, `*.csv`, `*.pt`, `*.safetensors`, `logs/`를
  모두 무시한다. 즉 데이터·체크포인트·실행 결과는 레포에 들어오지 않는 전제다.
- 논문(ICML 2026) 공식 코드의 포크이며, 상류(upstream)의 데이터 선별 파이프라인 위에
  install → download → select → train → merge → serve → eval 전 구간을
  스크립트로 감싼 것이 이 포크의 추가분이다.

따라서 이 레포의 산출물은 "실행 가능한 파이프라인"이고, 정답 검증 수단은 테스트가
아니라 각 단계가 출력하는 통계 리포트(선별 개수, 학습 생존 샘플 수, 벤치마크 성공률)다.

### 0.2 규모(실측)

`git ls-files` 기준 추적 파일 **35개**, 총 **5,864 LOC**.

| 분류 | 파일 수 | LOC |
|---|---:|---:|
| Python (`*.py`) | 18 | 4,079 |
| Shell (`*.sh`) | 12 | 1,105 |
| Markdown (루트 3개) | 3 | 658 |
| `.gitignore` | 1 | 22 |
| 이미지 (`assets/weasel_overview.png`) | 1 | — (약 2.1MB) |

Python 4,079 LOC의 내역은 알고리즘 패키지 `weasel/` **2,440 LOC**(8개 파일),
운영 스크립트 `scripts/*.py` **1,639 LOC**(10개 파일)다.

파일별 LOC 상위:

```
580  weasel/prepare_scores.py
497  scripts/miniwob_report.py
471  weasel/select_clean.py
406  weasel/convert_gemini.py
368  scripts/train_lora_sft.py
362  weasel/prune_axtree.py
247  weasel/select_greedy.py
236  scripts/setup_env.sh
212  weasel/postprocess_dataset.py
161  weasel/select_trajectories.py
```

커밋 36개, 이력 범위 2026-05-11 ~ 2026-07-01.

전체 Python 18개 파일 `py_compile` 통과, Shell 12개 파일 `bash -n` 통과를 실측 확인했다.

---

## 1. 목적과 범위

### 1.1 무엇을 푸는가

WEASEL은 웹 에이전트(web agent) 학습용 궤적(trajectory) 데이터에서
**중요도(importance)와 다양성(diversity)을 동시에 만족하는 부분집합**을 골라,
적은 데이터로 out-of-domain 일반화 성능을 유지하거나 개선하는 데이터 선별
(data selection) 기법이다. 레포는 그 선별 알고리즘과, 선별 결과로 실제 모델을
학습·서빙·평가하는 전 과정을 담는다.

핵심 신호 두 가지:

- **importance** — 목표(goal) 대비 관측(observation)의 관련도 증가분.
  `r_t = BERTScore(state_t, goal)`, `phi_t = max(0, r_t − r_{t−1})`.
  직전 스텝보다 목표에 얼마나 더 가까워졌는지를 나타내는 한계 이득(marginal gain)이다.
- **diversity** — 궤적 내 스텝 쌍의 거리.
  `d(i,j) = max(1 − sim_state(i,j), 1 − sim_response(i,j))`.
  상태(AXTree)와 응답(assistant 텍스트) 중 **더 멀리 떨어진 쪽**을 채택한다.

이 둘을 하나의 그리디(greedy) 목적함수로 결합해 궤적당 `t0`개 스텝만 남긴다.

### 1.2 범위 안

- AXTree 프루닝(pruning), 점수 계산, 그리디 부분집합 선별, 최종 학습셋 후처리
- 함수 호출(function-calling, FC) 형식 궤적을 위 파이프라인에 태우기 위한 변환기
- FC 데이터에 직접 적용하는 대안 원패스 큐레이션(`select_clean`)
- transformers + peft 기반 독립 LoRA SFT 트레이너 및 어댑터 병합
- vLLM 서빙 + AgentLab/BrowserGym 벤치마크 실행 + 성공률 요약 + HTML 궤적 분석 리포트
- 3개 격리 venv 생성, 모델·데이터 다운로드, 경로 환경변수 일괄 정의

### 1.3 범위 밖 (레포에 코드 없음)

- **self-reasoning synthesis**: SETUP.md가 명시적으로 "이 레포에 코드 없음"이라 적었다.
  논문의 Qwen3-8B 행에 적용된 단계이므로 해당 이득은 재현되지 않는다.
- **WebArena 사이트 구축 본체**: `setup_webarena.sh`는 `docker run` 명령과 URL 환경파일을
  생성할 뿐, 이미지 tarball 다운로드/로드는 사용자 몫이다.
- **WebArena-Lite 태스크 목록**: `agentlab_eval.py`가 `configs/webarena_lite_tasks.txt`를
  읽지만 **레포에 `configs/` 디렉터리 자체가 없다**(실측). 없으면 `--bench webarena_lite`는
  전체 812태스크를 'lite' 라벨로 잘못 돌리는 대신 즉시 중단한다 — 의도된 안전장치다.
- **원본 학습 데이터**: AgentTrek 등은 외부에서 받아야 한다. `train_data/` 디렉터리도
  레포에 없다(실측). `convert_traindata.sh`가 참조하는 입력 위치일 뿐이다.
- **테스트 코드**: 없음.

---

## 2. 전체 구조

두 개의 층으로 나뉜다.

```
┌─────────────────────────────────────────────────────────────────┐
│ scripts/  — 오케스트레이션 층 (12 sh + 10 py, 2,744 LOC)          │
│   환경변수 정의 · venv 관리 · 다운로드 · 실행 래핑 · 결과 집계      │
└───────────────────────────┬─────────────────────────────────────┘
                            │ python -m weasel.<mod> / python scripts/<x>.py
┌───────────────────────────┴─────────────────────────────────────┐
│ weasel/   — 알고리즘 층 (8 py, 2,440 LOC)                        │
│   프루닝 · 점수 · 그리디 선별 · 후처리 · 변환 · 원패스 큐레이션    │
└─────────────────────────────────────────────────────────────────┘
```

알고리즘 층은 환경변수를 전혀 읽지 않는다. 모든 입출력이 CLI 인자로만 들어온다.
경로/모델/GPU 배정 같은 운영 관심사는 전부 오케스트레이션 층이 흡수한다.
이 분리 덕분에 `weasel/`의 모듈들은 클러스터 밖에서도 파일 경로만 주면 그대로 돈다.

### 2.1 오케스트레이션 층 내부 구조

`setup_env.sh`가 유일한 설정 원본이다. 나머지 모든 `run_*.sh`는 첫머리에서
"자기가 필요한 변수가 비어 있거나 `weasel_activate` 함수가 없으면 `setup_env.sh`를
source한다"는 동일한 가드를 가진다.

```bash
if [ -z "${OUTPUT_ROOT:-}" ] || ! type weasel_activate >/dev/null 2>&1; then
  echo "Sourcing scripts/setup_env.sh..."; source scripts/setup_env.sh
fi
```

가드가 두 조건인 이유: bash에서 `export -f weasel_activate`로 함수를 자식에게 물려주지만
zsh 같은 비-bash 부모는 함수를 export할 수 없다. 그래서 변수만 상속된 상황을
함수 존재 여부로 감지해 재-source한다.

호출 그래프(실측, echo 안내문 제외):

```
setup_env.sh   ← 모든 스크립트가 source
install.sh     → (venv 3개 생성, miniwob-plusplus clone)
download_models.sh / download_data.sh
run_select.sh  → weasel.prune_axtree → weasel.prepare_scores
                 → weasel.select_greedy → weasel.postprocess_dataset
convert_traindata.sh → weasel.convert_gemini
run_train.sh   → scripts/train_lora_sft.py   (torchrun 또는 python)
run_merge.sh   → scripts/merge_lora.py
serve_vllm.sh  → vllm serve
run_eval.sh    → scripts/agentlab_eval.py
                 → scripts/summarize_results.py
                 → scripts/miniwob_report.py
run_experiment.sh → run_select.sh, train_lora_sft.py, merge_lora.py,
                    vllm serve(백그라운드), run_eval.sh
setup_webarena.sh → (docker run + 환경파일 생성)
```

어떤 셸 스크립트에서도 호출되지 않는 **수동 전용 진입점**(실측):
`weasel/select_clean.py`, `weasel/select_trajectories.py`,
`scripts/convert_dataset.py`, `scripts/inspect_dataset.py`,
`scripts/_resume_eval.py`, `scripts/_vllm_qwen35.py`,
`scripts/_overlay_text_into_mm.py`.
(`convert_traindata.sh` 끝의 `select_trajectories` 언급은 `echo` 안내문이며 실행이 아니다.)

---

## 3. 컴포넌트와 책임

### 3.1 알고리즘 층 (`weasel/`)

| 모듈 | LOC | 책임 | 계산 자원 |
|---|---:|---|---|
| `prune_axtree.py` | 362 | 학습 예제의 AXTree 섹션을 타깃 중심(target-centered)으로 잘라내 프롬프트 길이를 줄인다. 유효 bid가 없으면 접두(prefix) 임계 방식으로 대체. | CPU, 정규식만 |
| `prepare_scores.py` | 580 | goal 기준 관련도 `r`, 한계 이득 `phi`, 그리고 상태/응답 쌍별 거리 행렬을 한 번에 계산해 goal 레코드 JSON 하나로 저장. | GPU (BERTScore roberta-large) |
| `select_greedy.py` | 247 | 위 산출물만 읽어 궤적 세그먼트마다 그리디로 `t0`개 스텝을 고른다. | CPU, 표준 라이브러리 |
| `postprocess_dataset.py` | 212 | 선택된 인덱스로 원본에서 예제를 회수하고, 길이 필터 + 균등 서브샘플링으로 최종 학습 파일을 만든다. | CPU, 표준 라이브러리 |
| `convert_gemini.py` | 406 | FC 내보내기(1줄=1궤적)를 (a) 스텝 단위 ShareGPT, (b) 네이티브 FC ShareGPT 두 형태로 스트리밍 변환. | CPU |
| `select_trajectories.py` | 161 | 스텝 단위 선별 결과의 `_traj_id`를 되짚어 궤적 전체를 되살린다. 변환본 필터링과 원본 스키마 재방출 두 모드. | CPU |
| `select_clean.py` | 471 | (a)~(c) 경로를 우회하는 원패스 큐레이션. 스텝 단위 importance + 궤적 단위 지문(fingerprint) 중복 제거를 원본 파일에서 바로 수행하고 원본 스키마로 재방출. | GPU 선택(`--no-importance`면 CPU) |
| `__init__.py` | 1 | docstring뿐. | — |

### 3.2 오케스트레이션 층 (`scripts/`)

셸(12개):

| 스크립트 | LOC | 책임 |
|---|---:|---|
| `setup_env.sh` | 236 | 모든 경로/토큰/오프라인 플래그 export, 출력 디렉터리 생성, `weasel_activate` 정의, 경고 전용 경로 점검 |
| `install.sh` | 107 | select/train/eval 3개 venv 생성, Playwright Chromium, MiniWob++ HTML 클론(커밋 고정) |
| `download_models.sh` | 52 | 베이스 체크포인트 4종 다운로드(이미 `config.json` 있으면 스킵) |
| `download_data.sh` | 68 | 사전 선별 10K 파일 또는 원본 AgentTrek 풀 다운로드 + 변환 |
| `run_select.sh` | 87 | 선별 4단계(0~3) 직렬 실행, 로그 tee |
| `convert_traindata.sh` | 53 | `train_data/*.jsonl` → 스텝/궤적 두 산출물 |
| `run_train.sh` | 144 | 모델별 논문 레시피 적용, DDP(순차) 또는 GPU당 1모델(병렬) |
| `run_merge.sh` | 41 | 어댑터 → 병합 bf16 모델 |
| `serve_vllm.sh` | 42 | 병합 모델을 OpenAI 호환 엔드포인트로 서빙 |
| `run_eval.sh` | 88 | 벤치마크 전제조건 확인 → 평가 → 요약 CSV → HTML 리포트 |
| `run_experiment.sh` | 116 | full vs weasel 두 실험을 단일 모델에 대해 end-to-end 수행 |
| `setup_webarena.sh` | 71 | WebArena 사이트 URL 환경파일 생성, 선택적 docker 기동 |

Python(10개):

| 스크립트 | LOC | 책임 |
|---|---:|---|
| `train_lora_sft.py` | 368 | 독립 LoRA SFT. 채팅 템플릿 렌더 + assistant 전용 손실 마스킹 |
| `merge_lora.py` | 71 | LoRA 어댑터를 베이스에 병합해 standalone 모델 저장 |
| `agentlab_eval.py` | 138 | AgentLab study 구성·실행. 버전 민감 블록 격리 |
| `summarize_results.py` | 160 | `summary_info.json` 순회 → 성공률 표/CSV/JSON (표준 라이브러리만) |
| `miniwob_report.py` | 497 | 자체 완결형 HTML 궤적 분석 리포트 |
| `convert_dataset.py` | 112 | 임의 데이터셋 → `messages` 스키마, 선택적 `## Goal:` 주입 |
| `inspect_dataset.py` | 91 | 데이터셋 필드/역할 분포 탐색 |
| `_resume_eval.py` | 52 | 중단된 AgentLab study를 디스크에서 재개 |
| `_vllm_qwen35.py` | 57 | vLLM에 text-only Qwen3.5 아키텍처 등록 + 가중치 이름 리맵 |
| `_overlay_text_into_mm.py` | 93 | 멀티모달 베이스 위에 텍스트 튜닝 가중치를 덮어써 서빙 가능한 체크포인트 생성 |

---

## 4. 데이터·처리 흐름

### 4.1 스키마 계약

전 구간이 **OpenAI messages 스키마**를 공통 통화(currency)로 쓴다.

```json
{"messages": [{"role": "system|user|assistant|tool", "content": "..."}], "tools": [...]}
```

파이프라인 A(논문 재현)에서는 여기에 더해 `user` 턴 본문이 다음 마커를 포함해야 한다.
정규식으로 직접 긁기 때문에 마커가 없으면 해당 신호가 조용히 비어버린다.

```
## Goal: <목표 텍스트>
## AXTree: <접근성 트리>
# Observation of current step: <누적 관측>
# Action space: <행동 목록>
# History of interaction with the task:
```

`prune_axtree`의 섹션 정규식은 `## AXTree:` 와
`# History of interaction with the task:` **두 마커 사이**를 잘라낸다. 즉
History 마커가 없으면 프루닝은 아무 일도 하지 않는다(`missing_axtree`로 집계).

### 4.2 파이프라인 A — 논문 재현 경로(스텝 단위)

`run_select.sh`가 4단계를 직렬로 돌린다.

```
train.json  ──0──▶ train_pruned.json ──1──▶ goals_with_scores.json
   (원본)     프루닝    (레코드 수/순서 보존)   점수+거리행렬
                                        │
                                        2 그리디
                                        ▼
                            selected_indices_T0_3.json
                                        │
                                        3 후처리(길이 필터 + 10K 샘플링)
                                        ▼
                            weasel_*_train_10k.json  → 학습 입력
```

**단계 0 — 프루닝.** 각 예제의 assistant 응답에서 `<action>...</action>`을 파싱해
대상 bid를 뽑는다. bid가 있으면 그 bid를 중심으로 좌우 `--window-size`(기본 60)개
bid 항목을 남긴다. bid가 없거나(스크롤·goto 등) 트리에 없으면
`--fallback-threshold`(기본 120)개 접두 bid만 남긴다. **레코드 수와 순서가 보존**되므로
이후 단계에서 쓰는 인덱스가 원본과 계속 정렬된다 — 이것이 설계상 중요한 불변식이다.

**단계 1 — 점수.** goal 텍스트로 예제를 묶고(`group_trajectories`), 인덱스가 연속인
구간을 하나의 궤적 세그먼트로 본다(`split_contiguous`). 세그먼트마다:

- `r = BERTScore(obs_history, goal)` (F1) — `--phi-field` 기본 `obs_history`
- `r_norm` = 세그먼트 내 min-max 정규화
- `phi_raw[t] = max(0, r[t] − r[t−1])`, `r[−1] = 0`으로 시작
- `phi_norm` = 합이 1이 되도록 정규화(합이 0이면 전부 0)
- `sims_states` = AXTree 텍스트 쌍별 유사도 — `--state-field` 기본 `axtree`
- `sims_responses` = assistant 텍스트 쌍별 유사도 — `--response-field` 기본 `assistant`
- `distance[i][j] = max(1 − sims_states[i][j], 1 − sims_responses[i][j])`

BERTScore는 비대칭이므로 `(i→j)`와 `(j→i)`를 모두 계산해 평균한다. 세그먼트 크기 `n`에
대해 유사도 호출은 행렬당 `n(n−1)`회, 두 행렬이면 `2n(n−1)`회 + 관련도 `n`회다.
`--skip-response-distance`를 주면 응답 행렬을 건너뛰고 상태 거리만 쓴다.

**단계 2 — 그리디.** 세그먼트별 예산 `t0`(기본 고정 3)을 정하고:

1. `argmax_{i<j} [ u_i + u_j + λ·d(i,j) ]` 로 시드 쌍 선택
2. 이후 매 반복 `argmax_{c∉S} [ u_c + λ·Σ_{s∈S} d(c,s) ]` 를 추가
3. `|S| = t0`이 될 때까지 반복, 결과는 원래 순서로 정렬해 반환

`λ`는 `--lambda-weight` 기본 1.0. 예외 처리: `n ≤ t0`이면 전부 선택, `t0 == 1`이면
importance argmax 하나만(거리 행렬을 아예 검증하지 않음).

**단계 3 — 후처리.** 선택 인덱스를 평탄화하고 범위를 검증한 뒤 원본에서 회수한다.
`user` 프롬프트가 `--max-user-chars`(기본 40,000자)를 넘는 예제를 버리고,
남은 것이 `--max-examples`(기본 10,000)보다 많으면 `random.Random(seed).sample`로
균등 추출한다. 기본은 **샘플 순서 그대로** 저장하고, `--preserve-order`를 주면
데이터셋 인덱스 오름차순으로 재정렬한다.

### 4.3 파이프라인 B — FC 내보내기 변환 경로

원본이 "1줄 = 1개 멀티턴 궤적"인 함수 호출 로그일 때 `convert_gemini`가 두 산출물을
같은 `_traj_id` 아래 동시에 쓴다.

```
export.jsonl ──convert_gemini --mode both──┬─▶ gemini_steps.jsonl  (a) 스텝 단위 ShareGPT
                                           └─▶ gemini_traj.jsonl   (b) 네이티브 FC ShareGPT
```

- **(a)** 각 assistant 액션마다 레코드 하나를 만들고, `## Goal:` / `## AXTree:` /
  `# Observation of current step:` / `# Action space:` 마커를 합성해 넣는다.
  `## AXTree:` 자리에는 **직전 tool 관측 하나**(현재 상태)를,
  `# Observation of current step:` 자리에는 **지금까지의 관측 전체**(누적 이력)를 넣는다.
  액션은 `<think>…</think>` + `<action>name(args)</action>` 텍스트로 직렬화된다.
  이 파일은 파이프라인 A에 그대로 투입 가능하다.
- **(b)** tool_calls를 `function_call` 역할로, tool 결과를 `observation` 역할로 보존하고
  `tools`를 JSON 문자열 컬럼으로 붙인다. 트레이너가 이 역할명을 정규화하므로 바로 학습된다.

선별 후 (b)나 원본으로 되돌리는 것이 `select_trajectories`다. `--traj-dataset`은
변환본을 `_traj_id`로 필터링하고, `--original-input`은 **원본 파일을 같은 순서로 다시 읽어**
러닝 인덱스가 선택 집합에 들어가는 레코드만 그대로 재방출한다. 두 곳의 인덱싱 규칙
(공백/파손 줄 건너뛰고 성공 파싱한 레코드만 카운트)이 일치해야 정렬이 맞는다 —
`convert_gemini.iter_jsonl`과 `select_trajectories.iter_records`가 같은 규약을 쓴다.

> 실측 주의: `convert_gemini.iter_jsonl`은 `enumerate(f)`로 **파일의 물리적 줄 번호**를
> yield하고 파손 줄에서는 아무것도 yield하지 않는다. 반면 `select_trajectories.iter_records`와
> `select_clean.iter_records`는 **파싱 성공 레코드만 세는 별도 카운터**를 yield한다.
> 다만 `convert_gemini.main`은 yield된 `local_idx`를 `--limit` 비교에만 쓰고
> `_traj_id`는 자체 전역 카운터 `gid`로 부여하므로, 파손 줄이 있어도 세 파일의 번호는
> 결국 "성공 파싱 순번"으로 일치한다. `--limit`만 물리 줄 기준이라 디버그 전용이다.

### 4.4 파이프라인 C — `select_clean` 원패스 경로

A/B를 우회한다. 변환 왕복 없이 원본 파일에서 바로 큐레이션하고 원본 스키마로 내보낸다.

```
export.jsonl ─┬─ pass 1: 궤적별 (quality, fingerprint, task) 메타만 추출
              │            └ importance는 GPU 배치로 flush
              ├─ task별 그룹핑 → 근사 중복 제거 → keep-frac/keep-k 상위 선별
              └─ pass 2: 파일 재읽기 → 선택된 gid만 원본 그대로 재방출
```

두 신호를 **서로 다른 입도**에 배치한 것이 이 모듈의 핵심 설계다.

- importance는 **스텝 단위**로 유지한다. `obs_histories[t]`(액션 t 이전까지 본 tool 관측
  누적)와 goal을 BERTScore로 비교하므로 입력이 짧게 유지된다. 궤적 전체를 한 문서로
  넣으면 공통 시스템 프롬프트(주석에 따르면 65k자 규모)가 유사도를 지배해 신호가 무너진다.
- 중복 제거는 **궤적 단위**로 한다. 궤적을 "액션 토큰 집합 + 최종 답변의 단어 shingle"
  로 요약한 지문(fingerprint)을 만들고, task 그룹 안에서 Jaccard 유사도로 근사 중복을
  제거한다. 전역 all-pairs BERTScore를 피해 비용이 `O(N)` 읽기 + `O(group²)` 집합 연산에
  머문다.

품질 순으로 정렬한 뒤 그리디로 대표를 남기므로, 중복 클러스터에서는 **가장 품질이 높은
궤적**이 살아남는다. `--no-importance`이면 품질이 스텝 수로 대체되어 **가장 긴 궤적**이
대표가 된다.

### 4.5 학습 → 서빙 → 평가

```
학습 파일 ──train_lora_sft.py──▶ 어댑터  ──merge_lora.py──▶ 병합 bf16
                                                              │
                                              serve_vllm.sh ──┘
                                                    │ OpenAI 호환 :8000/v1
                                            agentlab_eval.py
                                                    │ study 디렉터리
                                     ┌──────────────┴──────────────┐
                            summarize_results.py          miniwob_report.py
                              성공률 표/CSV/JSON              HTML 궤적 리포트
```

`VARIANT`(기본 `weasel`)가 학습→병합→서빙→평가 전 구간을 관통하는 데이터 태그다.
어댑터는 `$OUTPUT_ROOT/<model>/<variant>`, 병합본은 `$MERGED_ROOT/<model>/<variant>`,
평가 결과는 `$EVAL_RESULTS_ROOT/<variant>`에 들어간다. full-data 실험과 WEASEL-subset
실험이 서로를 덮어쓰지 않게 하는 유일한 장치이므로, 네 단계에 **같은 VARIANT**를
넘기는 것이 사용자 책임이다.

---

## 5. 설계 결정과 이유

### 5.1 venv 3개 격리 (select / train / eval)

`install.sh`가 서로 다른 세 환경을 만든다.

- **select**: torch(cu124) + `bert-score==0.3.13` + tqdm + huggingface_hub + datasets + gdown
- **train**: torch(cu124) + transformers + peft + datasets + accelerate + bitsandbytes + liger-kernel
- **eval**: vllm + agentlab + browsergym(+webarena/miniwob/workarena) + Playwright Chromium

이유는 의존성 충돌이다. vLLM은 자체 torch를 핀(pin)하고, AgentLab은 Python <3.13을
요구하며, bert-score와 트레이너가 원하는 transformers 버전이 다르다. 하나로 합치면
어느 한쪽이 깨진다. 그 대가로 사용자는 단계마다 `weasel_activate {select|train|eval}`을
호출해야 하고, 디스크는 세 배로 든다.

`install.sh`는 인터프리터가 3.11~3.12 범위인지 확인하고 벗어나면 **경고만** 낸다
(중단하지 않음). MiniWob++ HTML은 클론 후 특정 커밋으로 `reset --hard`해 고정한다.

### 5.2 단일 워크스페이스 루트 + 파생 경로

`setup_env.sh`는 `WEASEL_WORK` 하나만 정하면 venv/모델/데이터/HF 캐시/체크포인트/
병합본/평가결과가 전부 그 아래로 파생되도록 설계했다.

```
WEASEL_WORK = ${GROUP_VOLUME}/${WEASEL_USER}/weasel   (기본)
  ├─ venvs/{select,train,eval}
  ├─ models/            ├─ data/            ├─ checkpoints/<model>/<variant>
  ├─ merged/<model>/<variant>               ├─ eval-results/<variant>
  ├─ experiments/       └─ cache/{huggingface,ms-playwright}
```

모든 변수가 `${VAR:-기본값}` 형태라 사전 export로 어떤 항목이든 개별 대체 가능하다.
`WEASEL_WORK` 하나만 바꾸면 마운트 구조가 다른 머신으로 통째로 이식된다.

HF 캐시(`HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`)와 Playwright 브라우저 경로
(`PLAYWRIGHT_BROWSERS_PATH`)까지 리다이렉트하는 이유는, 기본 위치가 홈 디렉터리라
용량이 작은 사용자 볼륨을 채워버리기 때문이다.

### 5.3 경고 전용 경로 점검

`setup_env.sh`는 필수 경로가 없어도 **절대 중단하지 않는다**. 없는 항목마다
`[missing] <변수> <경로>` 와 `fix: <실행할 명령>`을 출력하고 넘어간다.
source되는 파일이므로 `exit`나 `set -e`가 사용자의 대화형 셸을 죽일 수 있고,
설치 이전 시점에도 변수는 export되어 있어야 `install.sh`가 어디에 만들지 알기 때문이다.

점검 대상 6개(실측): `GROUP_VOLUME`, 세 venv, `WEASEL_TRAIN_JSON`, `MODEL_QWEN25_7B`.

### 5.4 기본 오프라인

`HF_DATASETS_OFFLINE` / `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE`을 기본 1로 둔다.
학습·평가 중 예기치 않은 허브 접속으로 시간을 낭비하거나 다른 리비전을 끌어오는 사고를
막는 의도다. 네트워크가 필요한 세 스크립트(`install.sh`, `download_models.sh`,
`download_data.sh`)만 자기 프로세스 안에서 0으로 뒤집는다 — 부모 셸은 영향받지 않는다.

> 실측 함정: BERTScore가 쓰는 `roberta-large`를 **미리 받아두는 코드가 어디에도 없다**
> (`grep -rn roberta` 결과: 기본값 문자열과 주석뿐). 캐시가 비어 있는 상태에서
> 기본 오프라인으로 `prepare_scores`/`select_clean`을 처음 돌리면 모델 로드에서 실패한다.
> 첫 실행에 한해 오프라인 플래그를 꺼야 한다. USAGE.md 문제 해결 절 참조.

### 5.5 LLaMA-Factory 제거, 독립 트레이너 채택

논문은 LLaMA-Factory로 SFT했지만 이 포크는 `train_lora_sft.py`(transformers + peft)로
교체했다(커밋 "Replace LLaMA-Factory with standalone LoRA SFT stack"). 얻은 것:

- 데이터셋 등록(registration) 절차 없이 **파일 경로 하나**로 학습
- ShareGPT 철자 차이(`conversations`/`from`/`value`, `human`/`gpt`,
  `function_call`/`observation`)를 코드가 직접 정규화하므로 변환 단계 불필요
- 프레임워크 버전과 무관하게 레시피가 코드에 그대로 드러남

레시피는 논문 Table 9를 그대로 옮겼다: LoRA rank 8 / alpha 8 / dropout 0 /
`all-linear` 타깃 / cosine 스케줄 / warmup ratio 0.1 / AdamW / bf16 / 에폭마다 저장.
rank와 alpha가 같으므로 LoRA 스케일 `α/r`은 정확히 1.0이다.

### 5.6 assistant 전용 손실과 접두 안정성 검증

프롬프트 토큰에 손실을 걸지 않기 위해 메시지별 토큰 구간을 알아내야 한다.
채팅 템플릿마다 규칙이 달라 안전한 방법이 없으므로, **증분 렌더 + 접두 일치 검증**을 쓴다.

`k = 1..len(msgs)`에 대해 `msgs[:k]`를 렌더하고, 그 결과가 전체 렌더의 정확한 접두인지
확인한다. 어긋나면(예: 비-최종 assistant 턴의 `<think>`를 지우는 템플릿) 그 샘플은
**전체 시퀀스 손실로 폴백**하고 카운트한다. 학습 시작 시 폴백 비율을 퍼센트로 출력하므로
템플릿이 이 가정을 깨는지 즉시 드러난다.

assistant 헤더 토큰을 손실에서 빼기 위해 `add_generation_prompt=True` 렌더를 추가로
비교한다. 그것도 접두가 아니면 헤더까지 학습한다(무해한 쪽으로 실패).

### 5.7 Arrow 스키마 통합 회피

`datasets.Dataset.from_list`에 레코드를 그대로 넣지 않고 **JSON 문자열 한 컬럼(`payload`)**
으로 감싼 뒤 `map` 안에서 풀어 쓴다. 코드 주석이 이유를 명시한다: Arrow가 레코드 간
struct 스키마를 통합하면서 서로 다른 tool 파라미터 스키마를 합집합으로 만들고,
렌더된 프롬프트에 `null` 필드를 주입하거나 혼합 타입 컬럼에서 크래시하기 때문이다.

### 5.8 장문맥 메모리 대책 — QLoRA와 Liger

`[seq, vocab]` 로짓 텐서가 장문맥에서 메모리를 지배한다. 32,768 토큰 × 248K 어휘로
계산하면 bf16 기준 `32768 × 248000 × 2B ≈ 16.3GB`, 손실 계산의 fp32 복사본이
`× 4B ≈ 32.5GB`, 합계 약 **49GB**다(SETUP.md 서술과 일치).

- `--liger`: Liger 융합 cross-entropy로 이 텐서를 아예 물리적으로 만들지 않는다.
- `--load-4bit`: 베이스를 4-bit NF4로 로드(9B 기준 약 18GB → 5.5GB). `run_train.sh`의
  `--qlora`는 `--load-4bit --liger`를 **함께** 켠다.

`--liger`에 방어 코드가 두 겹 들어 있다. transformers는 `trainer.train()` 시점에
Liger를 적용하고 미지원 모델 타입이면 **조용히 no-op**한다 — 메모리 절감이 사라졌는데
오류가 없다. 그래서 (1) 패키지 부재는 `ModuleNotFoundError`만 잡아 즉시 종료하고,
(2) `MODEL_TYPE_TO_APPLY_LIGER_FN`에 모델 타입이 없으면 경고를 출력한다.

4-bit 경로에는 추가 보정이 있다. peft의 `prepare_model_for_kbit_training`이 비양자화
파라미터를 fp32로 올리는데, 248K 어휘에서는 임베딩 + lm_head만 fp32로 약 8GB다.
그래서 **2차원 fp32 파라미터만** 계산 dtype으로 되돌리고, 안정성이 필요한 1차원 norm
벡터는 fp32로 남긴다. LoRA 파라미터는 이 처리 **이후에** 추가되므로 영향받지 않는다.

### 5.9 평가 하네스 버전 민감성 격리

AgentLab은 릴리스마다 에이전트/모델 인자 클래스명이 바뀐다. 이를 파일 전체로 번지지
않게 `agentlab_eval.py`의 `build_agent_args()` 하나에 가두고
`### AGENTLAB-VERSION-SENSITIVE` 주석으로 표시했다. 내부에서는 신형
`OpenAIModelArgs`를 먼저 시도하고 실패하면 `SelfHostedModelArgs`로 폴백한다.
토큰 예산은 `max_total_tokens=32,000`, `max_input_tokens=30,000`, `max_new_tokens=2,000`.

`--limit`도 같은 방어 태도다. `study.exp_args_list` 슬라이싱이 실패하면 경고만 내고
전체를 돈다. 반대로 `webarena_lite` 필터는 실패 시 **반드시 중단**한다 — 165태스크
서브셋인 줄 알고 812태스크 전체 성공률을 보고하는 사고가 조용한 실패보다 나쁘기 때문이다.

### 5.10 결과 집계를 두 겹으로 나눈 이유

- `summarize_results.py`는 **표준 라이브러리만** 쓴다. 어느 venv에서도 돌게 하려는 의도가
  docstring에 명시돼 있다. `summary_info.json`의 `cum_reward`만 읽는다.
- `miniwob_report.py`는 AgentLab이 import 가능하면 스텝별 궤적까지 읽어 액션 시퀀스·
  think·액션 오류를 드릴다운으로 보여주고, import에 실패하면 `summary_info.json`만으로
  **자동 강등**하며 리포트 상단에 그 사실을 배너로 표시한다.

성공 기준은 두 스크립트 모두 `cum_reward > 0`이다. MiniWob++ 보상이 `[-1, 1]`이고
실패 시 작은 음수가 나오므로 `> 0`이 표준 판정이며, 0/1 벤치마크에서는 `== 1`과 같다.

### 5.11 원본 스키마 재방출 원칙

`select_clean`과 `select_trajectories --original-input`은 선택된 레코드를
**손대지 않고** 그대로 다시 쓴다. `tools` / `messages` / `tool_calls` /
`reasoning_content` / `__source_*__`가 전부 보존된다. 선별은 부분집합 연산이지
형식 변환이 아니라는 관점이며, 결과물이 사용자의 기존 학습 하네스에 그대로 들어간다.
이 때문에 `select_clean`은 파일을 두 번 읽는다 — 1패스에서 메타만 뽑고, 2패스에서
원본을 다시 읽어 선택분만 흘려보낸다. 멀티 GB 파일을 RAM에 올리지 않기 위한 대가다.

### 5.12 스트리밍 우선

`convert_gemini`는 입력을 줄 단위로 읽고 출력도 줄 단위로 쓴다(멀티 GB 대응).
`select_clean`도 마찬가지다. 반대로 파이프라인 A의 `prune_axtree` /
`prepare_scores` / `postprocess_dataset`은 **전체를 메모리에 올린다** —
`prepare_scores`가 goal 단위 그룹핑과 인덱스 기반 상호 참조를 하려면 전량 접근이
필요하고, 산출물이 거리 행렬을 포함하는 단일 JSON이기 때문이다. 이 비대칭은
파이프라인 A의 실질적 데이터 크기 상한을 만든다.

---

## 6. 확장 지점

| 바꾸고 싶은 것 | 손댈 위치 |
|---|---|
| 프루닝 강도 | `run_select.sh`의 `WINDOW`/`FALLBACK`, 또는 `PRUNE=0`으로 비활성 |
| importance 정의 | `prepare_scores.py --phi-field {axtree,obs_history,user_prompt}` |
| 거리 정의 | `--state-field` / `--response-field` / `--skip-response-distance` |
| 선별 예산 | `select_greedy --t0-mode {fixed,percentage}` + `--t0-fixed`/`--t0-percentage` |
| 중요도/다양성 균형 | `select_greedy --lambda-weight` |
| 그리디가 쓰는 점수 종류 | `select_greedy --score-key` (기본은 `r_norm`, `phi_norm`도 산출돼 있음 — 6.1 참조) |
| 최종 학습셋 크기 | `postprocess_dataset --max-examples` / `--max-user-chars` |
| 새 베이스 모델 추가 | `run_train.sh:model_spec()` + `run_merge.sh:base_model()` + `setup_env.sh`의 `MODEL_*`/`HFID_*` |
| 새 벤치마크 추가 | `agentlab_eval.py:BENCH_TO_BROWSERGYM` + `run_eval.sh`의 전제조건 `case` |

### 6.1 그리디가 쓰는 기본 점수에 대한 실측 사항

`select_greedy`의 `--score-key` 기본값은 `bert_scores_obs_history_norm`이다.
`prepare_scores`가 쓰는 이 키의 내용은 **min-max 정규화된 관련도 `r_norm`** 이지
`phi`가 아니다. `phi_scores_obs_history_norm`(합-정규화된 `phi`)도 같은 레코드에
함께 저장되어 있으므로, 목적함수의 단항(unary) 항을 `phi`로 바꾸려면
`--score-key phi_scores_obs_history_norm`을 명시해야 한다. `run_select.sh`는
이 인자를 넘기지 않으므로 기본 실행은 `r_norm` 기준이다.
(README는 이 구분을 언급하지 않는다.)

---

## 7. 문서-구현 불일치 및 미사용 코드(실측)

세부 재현 절차와 코드 위치는 `CODE.md`에 있다. 여기서는 설계 영향이 있는 것만 요약한다.

1. **`download_data.sh`의 안내문이 낡았다.** `agenttrek` 경로 끝에서
   "`weasel.prune_axtree`가 레포에 없다(README says 'added soon')"고 출력한다.
   실제로는 `weasel/prune_axtree.py`(362 LOC)가 존재하고 `run_select.sh`가 이를
   **기본 단계 0으로 실행**한다. 이 문구는 잘못된 안내다.
2. **`prune_axtree`의 통계 키 충돌.** 폴백 건수 카운터가 임계값 상수를 담은 키를
   증가시킨다. 결과적으로 요약 출력의 `fallback_threshold=`는 **항상 0**으로 찍히고,
   통계 JSON의 `fallback_threshold` 값은 `120 + 폴백건수`가 된다. 실측 재현 완료.
3. **타깃 중심 프루닝의 첫 줄 중복.** 윈도가 트리 첫 줄을 포함하면 그 줄이 결과에
   두 번 들어간다. 실측 재현 완료.
4. **타깃 중심 프루닝의 개행 누락.** 잘라낸 AXTree 마지막 줄과
   `# History of interaction with the task:` 마커가 개행 없이 붙는다. 임계 방식
   (`prune_threshold`)에는 개행이 들어가므로 두 경로의 출력 형식이 다르다.
   `prepare_scores`의 AXTree 정규식이 `\n#` 선행 조건을 쓰므로, 이 경우 AXTree 캡처가
   History 섹션까지 삼킨다. 실측 재현 완료.
5. **`prepare_scores --lang`은 효과가 없다.** 파싱해서 `score_metadata`에 기록만 하고
   `BERTScorer` 생성자에 넘기지 않는다. `--model-type`을 명시하고
   `rescale_with_baseline=False`이므로 실동작에는 영향이 없지만, 인자 이름이 오해를 부른다.
6. **`configs/webarena_lite_tasks.txt`가 레포에 없다.** `--bench webarena_lite`는
   외부에서 목록을 가져와야 동작한다(코드가 이를 명시하고 중단하므로 위험하진 않다).
7. **`select_clean --near-dup-threshold`의 도움말이 낡았다.** "Signature Jaccard"라고
   적혀 있는데, 커밋에서 용어를 `signature` → `fingerprint`로 바꿀 때 이 문자열만
   남았다. 모듈 docstring과 변수명은 모두 `fingerprint`다.
8. **`miniwob_report.py`의 `import gzip`은 미사용이다.** docstring이 "per-step 파일은
   gzip된 pickle"이라 설명하지만 실제 압축 해제는 AgentLab의
   `yield_all_exp_results`가 수행하므로 이 import는 죽은 코드다.
9. **`_overlay_text_into_mm.py`의 인자 기본값이 특정 머신 절대경로로 하드코딩돼 있다.**
   다른 환경에서는 `--base` / `--text` / `--out`을 반드시 명시해야 한다.
10. **`select_clean`은 어떤 셸 스크립트에도 연결돼 있지 않다.** README와 SETUP.md에
    사용법이 있지만 `run_*.sh` 어디서도 호출하지 않는 수동 전용 진입점이다.
    `select_trajectories`, `convert_dataset`, `inspect_dataset`, `_resume_eval`,
    `_vllm_qwen35`, `_overlay_text_into_mm`도 마찬가지다.
11. **`roberta-large` 사전 다운로드 단계가 없는데 기본이 오프라인이다.** 5.4 참조.
12. **`EXPERIMENTS.md`에 미완결 항목이 남아 있다.** "데이터셋 필드와 goal/궤적/스텝 정의
    두 가지가 아직 필요하다"는 요청 문단이 그대로 있어, 새 데이터셋용
    `convert_dataset.py` 매핑이 확정되지 않은 상태임을 나타낸다.
