# USAGE — 사용 방법

환경 준비, 실행 절차, 설정 변경 포인트, 재현, 문제 해결을 다룬다.
모든 명령과 기본값은 레포 코드에서 확인한 것이다. 명령은 항상
**레포 루트에서** 실행한다(스크립트들이 `cd "$(dirname "$0")/.."`로 루트를 가정한다).

---

## 1. 환경 준비

### 1.1 요구 사항

| 항목 | 요구 | 근거 |
|---|---|---|
| Python | 3.11 또는 3.12 | `install.sh`가 이 범위를 벗어나면 경고(AgentLab이 <3.13 요구) |
| GPU | CUDA 지원 NVIDIA GPU | torch cu124 휠, vLLM |
| 시스템 패키지 | `git git-lfs python3-venv build-essential tmux` | README의 generic VM 절 |
| 루트 권한 | Playwright `install-deps`에 필요 | 없으면 `WEASEL_XLIBS_DIR` 우회 |
| 디스크 | 전체 세트 200GB 이상 / 단일 모델 60GB | README 디스크 예산 절 |

`install.sh`는 `PYTHON_BIN`(기본 `python3.12`, 없으면 `python3`)을 쓴다.
버전이 범위를 벗어나도 **중단하지 않고 경고만** 하므로, eval venv가 깨질 수 있음을
알고 진행해야 한다.

### 1.2 작업 공간 지정

모든 큰 산출물이 `WEASEL_WORK` 아래에 모인다. 기본값은
`${GROUP_VOLUME}/${WEASEL_USER}/weasel`이고 `GROUP_VOLUME`의 기본값은
관리형 클러스터용 마운트다. 그런 마운트가 없는 일반 VM에서는 **source 전에** 지정한다.

```bash
export WEASEL_WORK=/your/large/disk/weasel
echo 'export WEASEL_WORK=/your/large/disk/weasel' >> ~/.bashrc   # 새 셸/tmux 창 상속
source scripts/setup_env.sh
```

또는 루트만 바꾸고 기본 레이아웃을 유지해도 된다.

```bash
export GROUP_VOLUME=/your/large/disk
source scripts/setup_env.sh
```

`setup_env.sh`는 없는 경로를 경고만 하고 절대 셸을 죽이지 않는다.
출력 끝에 `WEASEL_WORK`, `WEASEL_TRAIN_JSON`, `OUTPUT_ROOT`, `EVAL_RESULTS_ROOT`,
세 venv 경로가 요약되므로 항상 확인하고 넘어간다.

**셸을 새로 열 때마다 `source scripts/setup_env.sh`를 다시 해야 한다.**
tmux 창을 새로 만들 때도 마찬가지다.

### 1.3 venv 설치

```bash
bash scripts/install.sh all       # select + train + eval
# 또는 필요한 것만
bash scripts/install.sh select
bash scripts/install.sh train
bash scripts/install.sh eval
```

세 환경의 의존성이 실제로 충돌하므로 합칠 수 없다. 단계마다 전환한다.

```bash
weasel_activate select   # 선별 파이프라인
weasel_activate train    # LoRA SFT / 병합
weasel_activate eval     # vLLM 서빙 / AgentLab 평가
```

`run_*.sh`들은 필요한 venv를 스스로 활성화하므로, 셸에서 직접 파이썬 모듈을 부를 때만
수동 전환이 필요하다.

eval venv 설치는 `playwright install-deps chromium`을 시도하고, 실패하면 안내만 남긴다.
루트가 있다면 한 번 `sudo`로 실행하고, 없다면 필요한 X11 라이브러리를 별도로 마련한 뒤
`WEASEL_XLIBS_DIR`을 그 `lib` 디렉터리로 지정한다(`setup_env.sh`가
`LD_LIBRARY_PATH`에 붙여준다). conda가 있고 env 이름이 `weasel-xlibs`이면 자동 탐지한다.

### 1.4 Hugging Face 토큰

`google/gemma-3-4b-it`은 게이트 모델이라 로그인이 필요하다. 세 방법 중 하나:

```bash
weasel_activate select && huggingface-cli login       # 권장
# 또는 셸 rc에서 source 전에
export HF_TOKEN=<토큰>
# 또는 별도 파일 (setup_env.sh가 자동 source)
echo 'export HF_TOKEN=<토큰>' > ~/.weasel_secrets.sh && chmod 600 ~/.weasel_secrets.sh
```

`setup_env.sh`는 git 추적 대상이므로 토큰을 그 안에 적으면 안 된다.
요약 출력에서 토큰은 값이 아니라 길이만 표시된다.

### 1.5 모델·데이터 내려받기

```bash
bash scripts/download_models.sh all
# 개별: qwen25_7b | gemma3_4b | qwen3_8b | qwen35_9b
bash scripts/download_data.sh            # 사전 선별 10K (빠른 경로)
```

대상 디렉터리에 `config.json`이 이미 있으면 건너뛴다. 이미 다른 곳에 체크포인트가
있다면 다운로드 대신 `MODEL_*` 변수를 그쪽으로 지정하면 된다.

```bash
export MODEL_QWEN25_7B=/existing/mirror/Qwen2.5-7B-Instruct
```

---

## 2. 실행 절차

### 2.1 빠른 경로 — 선별을 건너뛰고 학습부터

사전 선별된 10K 파일을 받아 바로 학습·평가한다.

```bash
source scripts/setup_env.sh
bash scripts/install.sh all
weasel_activate select && huggingface-cli login       # gemma를 쓸 때만
bash scripts/download_models.sh all
bash scripts/download_data.sh
bash scripts/run_train.sh --gpus 0,1,2,3,4,5,6,7
bash scripts/run_merge.sh
bash scripts/serve_vllm.sh qwen25 --gpus 0            # 이 창은 그대로 둔다
# 다른 창에서
source scripts/setup_env.sh
bash scripts/run_eval.sh --bench miniwob
```

서빙과 평가는 각각 별도 tmux 창에서 돌려 SSH 끊김에 대비한다.

```bash
tmux new -s serve 'source scripts/setup_env.sh && bash scripts/serve_vllm.sh qwen25 --gpus 0'
tmux new -s eval  'source scripts/setup_env.sh && bash scripts/run_eval.sh --bench miniwob'
```

vLLM은 `127.0.0.1:8000`에만 바인딩하므로 방화벽 포트를 열 필요가 없다.

### 2.2 선별 파이프라인 재실행 (논문 재현 경로)

```bash
bash scripts/download_data.sh agenttrek     # 원본 풀 → $TRAIN_INPUT_JSON
bash scripts/run_select.sh --gpus 0
```

4단계가 순서대로 돈다.

```
0/3 prune_axtree        (CPU)  → $WEASEL_DATA/<입력이름>_pruned.json + _stats.json
1/3 prepare_scores      (GPU)  → $GOALS_SCORES_JSON, train_with_phi_scores.json
2/3 select_greedy       (CPU)  → $SELECTED_INDICES_JSON
3/3 postprocess_dataset (CPU)  → $WEASEL_TRAIN_JSON
```

전체 로그는 `logs/run_select.log`에 tee된다.

환경변수로 조절한다.

```bash
T0=3 LAMBDA=1.0 BATCH=64 bash scripts/run_select.sh --gpus 0
PRUNE=0 bash scripts/run_select.sh --gpus 0                    # 프루닝 없이 점수 계산
WINDOW=60 FALLBACK=120 bash scripts/run_select.sh --gpus 0
bash scripts/run_select.sh --input /path/to/other.json --gpus 0
```

`--gpus`에 여러 개를 줘도 **첫 번째만** 쓴다(BERTScore 단일 프로세스).

각 단계를 직접 부르고 싶다면(모두 `weasel_activate select` 상태에서):

```bash
python -m weasel.prune_axtree --input train.json --output train_pruned.json \
  --window-size 60 --fallback-threshold 120 --stats-output prune_stats.json

python -m weasel.prepare_scores --input train_pruned.json \
  --output goals_with_scores.json \
  --augmented-dataset-output train_with_phi_scores.json --batch-size 64

python -m weasel.select_greedy --input goals_with_scores.json \
  --output selected_indices_T0_3.json --t0-fixed 3 --lambda-weight 1.0

python -m weasel.postprocess_dataset --dataset train_pruned.json \
  --selected-indices selected_indices_T0_3.json --output weasel_train_10k.json \
  --max-user-chars 40000 --max-examples 10000 --seed 0
```

단계 3의 `--dataset`은 반드시 **단계 0의 출력**(프룬본)이어야 한다.
인덱스가 그 파일 기준이기 때문이다. 프루닝을 껐다면 원본을 그대로 넘긴다.

### 2.3 자체 함수 호출 데이터 사용

원본이 "1줄 = 1궤적"인 FC 내보내기라면 세 가지 경로가 있다.

**(a) 논문 방식 — 스텝 단위 선별 후 스텝 데이터로 학습**

```bash
# train_data/ 아래에 .jsonl을 두거나 INPUTS로 지정
bash scripts/convert_traindata.sh
TRAIN_INPUT_JSON=$WEASEL_DATA/gemini_steps.jsonl bash scripts/run_select.sh --gpus 0
CUTOFF=32768 bash scripts/run_train.sh --gpus 0
```

**(b) 선별 후 네이티브 FC 형태로 학습**

```bash
python -m weasel.select_trajectories \
  --selected-dataset "$WEASEL_TRAIN_JSON" \
  --traj-dataset "$WEASEL_DATA/gemini_traj.jsonl" \
  --output "$WEASEL_DATA/gemini_traj_selected.jsonl"
DATA_FILE=$WEASEL_DATA/gemini_traj_selected.jsonl CUTOFF=32768 bash scripts/run_train.sh --gpus 0
```

**(c) 선별만 하고 원본 스키마 유지**

```bash
python -m weasel.select_trajectories \
  --selected-dataset "$WEASEL_TRAIN_JSON" \
  --original-input train_data/a.jsonl train_data/b.jsonl \
  --original-output "$WEASEL_DATA/selected_original.jsonl"
```

`--original-input`에는 **변환할 때와 같은 파일을 같은 순서로** 넘겨야 한다.
`_traj_id`가 파일들을 가로지르는 러닝 번호이기 때문이다. 변환 시
`--limit`을 썼다면 번호가 어긋나므로 쓰지 않는다(디버그 전용).

### 2.4 원패스 큐레이션 (`select_clean`)

변환 왕복 없이 원본 파일에서 바로 선별하고 원본 스키마로 내보낸다.
`weasel_activate select` 상태에서 실행한다.

```bash
# 중요도(GPU) + 근사 중복 제거, 태스크별 상위 절반 유지
python -m weasel.select_clean --input export.jsonl --output weasel_clean.jsonl --keep-frac 0.5

# 중복 제거만 — GPU/bert_score 불필요
python -m weasel.select_clean --input export.jsonl --output weasel_clean.jsonl --no-importance

# 여러 파일을 한 풀로
python -m weasel.select_clean --input a.jsonl b.jsonl --output clean.jsonl --keep-k 20
```

주요 옵션(기본값):

| 옵션 | 기본 | 의미 |
|---|---|---|
| `--quality` | `meanphi` | `meanphi` = 평균 phi, `final` = 마지막 관련도 `r_T` |
| `--near-dup-threshold` | 0.9 | 지문 Jaccard가 이 값 이상이면 근사 중복 |
| `--keep-frac` / `--keep-k` | 없음 | 태스크별 상위 비율/개수 (둘 중 하나만) |
| `--min-steps` | 1 | 액션 수가 이보다 적으면 폐기 |
| `--answer-lang` | 없음 | `ko` 또는 `zh` — 다른 CJK 스크립트 답변 제거 |
| `--task-field` | `__source_task__` | 태스크 식별 필드, 없으면 goal 텍스트 |
| `--no-importance` | off | BERTScore 생략, 품질을 스텝 수로 대체 |
| `--batch-size` / `--score-chunk` | 64 / 64 | GPU 배치 / flush 단위 |
| `--max-obs-chars` / `--max-history-chars` | 4000 / 8000 | 관측·이력 절단 |
| `--arg-chars` / `--answer-shingle` | 60 / 5 | 지문 구성 |
| `--max-jaccard-group` | 3000 | 이보다 큰 그룹은 정확 일치 중복만 제거 |

출력은 원본 스키마 JSONL이므로 `DATA_FILE=<출력>`으로 바로 학습할 수 있다.

실행 끝에 나오는 두 리포트를 임계값 조정에 쓴다.

- **rollouts/task 히스토그램** — 태스크당 롤아웃이 몇 개인지. 대부분 1이면 중복 제거로
  얻을 게 없다.
- **Jaccard 분포** — 중복 제거 판정에 쓰인 최대 Jaccard의 10구간 분포와 현재 임계값
  위치. 임계값 바로 아래에 큰 봉우리가 있으면 임계값을 낮춰 더 걸러낼 여지가 있다.

`--quality`와 `--no-importance`는 **대표로 살아남는 궤적의 성격을 반대로 바꾼다**.
`meanphi`는 스텝 수로 나누므로 짧고 목표 지향적인 궤적을, `--no-importance`는
스텝 수 자체가 품질이므로 가장 긴 궤적을 남긴다.

### 2.5 학습

```bash
bash scripts/run_train.sh --gpus 0,1,2,3,4,5,6,7        # 순차 DDP
bash scripts/run_train.sh --gpus 0,1,2 --parallel       # GPU당 1모델 동시
MODELS="qwen25 qwen3" bash scripts/run_train.sh --gpus 0,1
MODELS="qwen35_9b" bash scripts/run_train.sh --gpus 0   # 기본 세트에 없으므로 명시 필요
```

모델별 레시피는 코드에 고정되어 있다.

| 키 | 베이스 | lr | epochs | 전역 배치 |
|---|---|---|---|---|
| `qwen25` | Qwen2.5-7B-Instruct | 2.0e-5 | 4.0 | 8 |
| `gemma3` | gemma-3-4b-it | 2.0e-5 | 2.0 | 16 |
| `qwen3` | Qwen3-8B | 1.0e-6 | 2.0 | 8 |
| `qwen35_9b` | Qwen3.5-9B | 1.0e-6 | 2.0 | 8 |

전역 배치는 GPU 수와 무관하게 유지된다
(`grad-accum = 전역배치 / (GPU수 × PER_DEVICE)`, 최소 1). 따라서 GPU가 1장이든
8장이든 같은 결과를 기대할 수 있다.

메모리 옵션:

```bash
bash scripts/run_train.sh --gpus 0 --liger                     # 융합 CE만
QLORA=1 CUTOFF=32768 bash scripts/run_train.sh --gpus 0,1      # 4-bit + 융합 CE
```

`--qlora`는 `--load-4bit --liger`를 함께 켠다. 24GB급 카드에서 9B를 32K 컷오프로
돌릴 때 필요하다. A100 80GB에서도 `CUTOFF>=32768`이면 `--liger`를 권장한다 —
32,768 × 248,000 어휘 로짓이 bf16 약 16GB + 손실의 fp32 사본 약 33GB로 합계 약 49GB다.

데이터 변형(variant):

```bash
bash scripts/run_train.sh --gpus 0                                     # VARIANT=weasel
VARIANT=full DATA_FILE="$NEWDATA_FULL_JSON" bash scripts/run_train.sh --gpus 0
```

`VARIANT`가 `weasel`이 아니면 `DATA_FILE`이 **필수**다(없으면 종료).
어댑터는 `$OUTPUT_ROOT/<model>/<variant>`에, 에폭마다 `checkpoint-*`가 생기고
최종 어댑터는 디렉터리 루트에 저장된다. 로그는 `logs/train_<model>.log`.

트레이너를 직접 부를 수도 있다.

```bash
weasel_activate train
python scripts/train_lora_sft.py \
  --model-path <베이스> --data <학습파일> --output-dir out/adapter \
  --lr 1e-6 --epochs 2 --cutoff-len 8192 --lora-r 8 --lora-alpha 8
```

학습 시작 시 다음 두 줄을 반드시 확인한다.

```
[train_lora_sft] N/M usable records from <파일>
[train_lora_sft] K/N examples after tokenization (cutoff C)
```

`K`가 급감했다면 컷오프가 짧아 assistant 토큰이 전부 잘린 것이다.
폴백 경고가 뜨면 채팅 템플릿이 접두 안정성을 깨는 것이므로,
해당 비율이 높을 때는 손실 마스킹이 사실상 무력화됐다고 봐야 한다.

### 2.6 병합과 서빙

```bash
bash scripts/run_merge.sh                          # VARIANT=weasel
MODELS="qwen25" VARIANT=full bash scripts/run_merge.sh
bash scripts/serve_vllm.sh qwen25 --gpus 0
TP=2 VLLM_PORT=8001 bash scripts/serve_vllm.sh qwen3 --gpus 0,1
VARIANT=full bash scripts/serve_vllm.sh qwen25 --gpus 0
```

`serve_vllm.sh` 기본값: 텐서 병렬 1, 최대 문맥 32,768, 호스트 `127.0.0.1`,
포트 8000, 서빙 모델명 `weasel`.

수동 병합:

```bash
python scripts/merge_lora.py --base <베이스> --adapter out/adapter --output out/merged
```

`--adapter`는 학습 출력 디렉터리(루트에 어댑터가 있는 형태)나 특정 `checkpoint-*`
둘 다 받는다. 루트에 `adapter_config.json`이 없으면 **번호가 가장 큰 체크포인트**를
자동 선택하고 그 사실을 출력한다.

### 2.7 평가

```bash
bash scripts/run_eval.sh --bench miniwob                    # 완전 로컬
bash scripts/run_eval.sh --bench workarena_l1               # ServiceNow 개발 인스턴스 필요
bash scripts/run_eval.sh --bench webarena --n-jobs 4        # 자체 호스팅 WebArena 필요
bash scripts/run_eval.sh --bench miniwob --limit 20         # 스모크
VARIANT=full bash scripts/run_eval.sh --bench miniwob       # 서빙한 것과 같은 VARIANT
```

서빙과 평가에 **같은 `VARIANT`를 주는 것이 중요하다**. 결과가
`$EVAL_RESULTS_ROOT/<variant>/`로 분리되므로, 다르면 full 데이터 결과와
WEASEL 부분집합 결과가 섞인다.

산출물:

```
$EVAL_RESULTS_ROOT/<variant>/<study>/weasel_summary.json    성공률 요약(JSON)
logs/eval_<variant>_<bench>.log                             전체 로그
logs/eval_<variant>_<bench>_summary.csv                     태스크별 CSV
logs/eval_<variant>_<bench>_report.html                     HTML 궤적 리포트
```

요약과 리포트는 **실패해도 평가를 실패시키지 않는다**. 나중에 따로 만들 수 있다.

```bash
python scripts/summarize_results.py --root "$EVAL_RESULTS_ROOT/weasel" --csv out.csv
python scripts/summarize_results.py --study-dir <study 경로>
python scripts/miniwob_report.py --study-dir <study 경로> --out report.html --max-traj 800
```

`miniwob_report.py`는 **eval venv에서 돌려야** 스텝별 궤적까지 읽는다.
다른 환경에서는 `summary_info.json` 수준 통계로 자동 강등되고 리포트 상단에
그 사실이 표시된다.

성공 판정은 두 스크립트 모두 `cum_reward > 0`이다. MiniWob++ 보상이 `[-1, 1]`이라
실패 시 작은 음수가 나오기 때문이고, 0/1 벤치마크에서는 `== 1`과 동일하다.

#### WebArena 준비

```bash
HOST=<사이트 호스트> bash scripts/setup_webarena.sh env
source "$WEASEL_WORK/webarena_env.sh"
bash scripts/run_eval.sh --bench webarena --n-jobs 4
```

`env`는 `WA_*` 7개 변수를 담은 파일을 쓸 뿐이다. 사이트 자체는 Docker 이미지
tarball을 로드해 띄우거나 공식 AMI를 쓰는 별도 작업이다. `up` 서브커맨드는
이미지가 이미 `docker load`된 상태를 가정하고 컨테이너 4개를 띄운다.
GPT 판정기 때문에 `OPENAI_API_KEY`가 필요하다(없으면 경고 후 진행하다 실패한다).

`--bench webarena_lite`는 165태스크 공식 목록 파일이 있어야 한다.
**레포에 `configs/webarena_lite_tasks.txt`가 없으므로** 직접 넣거나
`WEBARENA_LITE_TASKS=<파일>`로 지정해야 한다. 없으면 전체 812태스크를
'lite' 라벨로 잘못 돌리는 대신 즉시 중단한다.

#### WorkArena 준비

```bash
export SNOW_INSTANCE_URL=<인스턴스 URL>
export SNOW_INSTANCE_UNAME=<사용자명>
export SNOW_INSTANCE_PWD=<비밀번호>
bash scripts/run_eval.sh --bench workarena_l1
```

`run_eval.sh`가 자동으로 `workarena-install`을 한 번 실행한다(멱등).

#### 중단된 평가 재개

```bash
weasel_activate eval
python scripts/_resume_eval.py --study-dir <study 경로> --n-jobs 4
```

`agentlab_eval.py`는 항상 새 study 디렉터리를 만들기 때문에, 중간에 끊긴 study를
이어서 돌리려면 이 스크립트를 쓴다. 완료된 태스크는 재사용하고 미완만 다시 실행한다.

### 2.8 full vs weasel 비교 실험

단일 모델에 대해 선택→학습→병합→서빙→평가를 한 번에 돈다.

```bash
bash scripts/run_experiment.sh --exp full   --gpus 0,1,2,3,4,5,6,7
bash scripts/run_experiment.sh --exp weasel --gpus 0,1,2,3,4,5,6,7
```

`--exp weasel`은 먼저 `$NEWDATA_FULL_JSON`에 대해 선별을 돌려
`$NEWDATA_WEASEL_JSON`을 만든다. 두 실험의 학습 레시피는 동일하고
데이터만 다르다. 결과는 `$EXP_OUTPUT_ROOT/<exp>/qwen35_9b/eval/<exp>`에 놓인다.

플래그: `--bench --cutoff --per-device --serve-gpu --tp --no-eval`,
환경변수: `LR=` `EPOCHS=` `QLORA=1` `LIGER=1`.

서빙은 백그라운드로 띄우고 `$OPENAI_API_BASE/models`를 10초 간격 최대 120회
(= 20분) 폴링해 준비를 기다린다. 그 안에 뜨지 않거나 프로세스가 죽으면 오류로 끝나고,
`EXIT` 트랩이 서버를 정리한다.

새 데이터셋을 이 실험에 태우려면 먼저 스키마를 맞춘다.

```bash
weasel_activate train
python scripts/inspect_dataset.py /path/to/raw.jsonl           # 필드 파악
python scripts/convert_dataset.py --in /path/to/raw.jsonl --out "$NEWDATA_FULL_JSON" \
  --user-field <필드> --assistant-field <필드> [--system-field <필드>] \
  --goal-field <필드>
```

`--goal-field`가 핵심이다. 이것이 user 턴 앞에 `## Goal: <텍스트>`를 넣어주고,
그래야 `prepare_scores`가 궤적을 goal 단위로 묶을 수 있다. 없으면 모든 예제가
`<NO_GOAL_FOUND>` 하나로 뭉쳐 선별이 무의미해진다. `run_experiment.sh`가
실행 전 `grep -q "## Goal:"`로 확인해 경고하지만 **중단하지는 않는다**.

---

## 3. 설정 변경 포인트

### 3.1 경로

전부 `setup_env.sh`의 `${VAR:-기본값}`이므로, source **이전에** export하면 대체된다.

```bash
export WEASEL_WORK=/other/disk/weasel        # 한 줄로 전체 이동
export MODEL_QWEN25_7B=/existing/Qwen2.5-7B-Instruct
export WEASEL_TRAIN_JSON=/my/own/train.json
export OUTPUT_ROOT=/other/checkpoints
source scripts/setup_env.sh
```

### 3.2 선별 알고리즘

| 바꾸고 싶은 것 | 방법 | 기본값 |
|---|---|---|
| 프루닝 창 | `WINDOW=` 또는 `--window-size` | 60 |
| 프루닝 폴백 임계 | `FALLBACK=` 또는 `--fallback-threshold` | 120 |
| 프루닝 자체 끄기 | `PRUNE=0` | 켜짐 |
| 궤적당 선택 수 | `T0=` 또는 `--t0-fixed` | 3 |
| 비율 기반 예산 | `--t0-mode percentage --t0-percentage 0.25` | fixed 모드 |
| 다양성 가중 | `LAMBDA=` 또는 `--lambda-weight` | 1.0 |
| 중요도로 쓸 점수 | `--score-key` | `bert_scores_obs_history_norm`(= `r_norm`) |
| 중요도 계산 대상 텍스트 | `--phi-field {axtree,obs_history,user_prompt}` | `obs_history` |
| 상태 유사도 대상 | `--state-field` | `axtree` |
| 응답 유사도 대상 | `--response-field {assistant,reasoning,action,assistant_without_think}` | `assistant` |
| 응답 거리 끄기 | `--skip-response-distance` | 켜짐(= max 결합) |
| BERTScore 모델 | `--model-type` | `roberta-large` |
| BERTScore 배치 | `BATCH=` 또는 `--batch-size` | run_select 64 / 모듈 32 |
| 최종 학습셋 크기 | `--max-examples` | 10000 |
| 프롬프트 길이 상한 | `--max-user-chars` | 40000 |
| 샘플링 시드 | `--seed` | 0 |

> `--score-key`의 기본값은 정규화된 **관련도 `r`** 이지 `phi`가 아니다.
> 논문 서술의 importance를 단항 항으로 쓰려면
> `--score-key phi_scores_obs_history_norm`을 명시해야 하며,
> `run_select.sh`는 이 인자를 넘기지 않으므로 스크립트를 통한 기본 실행은 `r_norm` 기준이다.

### 3.3 학습

| 항목 | 방법 |
|---|---|
| 학습할 모델 집합 | `MODELS="qwen25 qwen3"` |
| 모델별 lr/epochs/전역배치 | `run_train.sh`의 `model_spec()` 편집 |
| 새 모델 추가 | `model_spec()` + `run_merge.sh:base_model()` + `setup_env.sh`의 `MODEL_*`/`HFID_*` |
| 컨텍스트 길이 | `CUTOFF=` 또는 `--cutoff` |
| GPU당 배치 | `PER_DEVICE=` (전역 배치는 accum으로 자동 보정) |
| 메모리 절감 | `--liger` / `--qlora` (또는 `LIGER=1` / `QLORA=1`) |
| LoRA rank/alpha | 트레이너 직접 호출 시 `--lora-r` `--lora-alpha` (`run_train.sh`는 8/8 고정) |
| LoRA 타깃 모듈 | `--lora-targets q_proj,v_proj,...` (기본 `all-linear`) |
| 스케줄러/워밍업 | `--scheduler` `--warmup-ratio` |
| 체크포인트 정책 | `--save-strategy {epoch,steps,no}`, `--save-state` |
| 마스킹 끄기 | `--train-on-all` (전체 토큰 손실) |
| 재개 | `--resume` (`--save-state`와 함께 써야 정확) |

### 3.4 서빙·평가

| 항목 | 방법 | 기본 |
|---|---|---|
| 포트 | `VLLM_PORT=` | 8000 |
| 서빙 모델명 | `VLLM_SERVED_NAME=` | `weasel` |
| 텐서 병렬 | `TP=` 또는 `--tp` | 1 |
| 최대 문맥 | `MAXLEN=` | 32768 |
| 병렬 태스크 | `--n-jobs` | 1 |
| 태스크 수 제한 | `--limit` | 없음 |
| 결과 분리 태그 | `VARIANT=` | `weasel` |
| 리포트 드릴다운 상한 | `--max-traj` | 400 |
| WebArena-Lite 목록 | `WEBARENA_LITE_TASKS=<파일>` | `configs/webarena_lite_tasks.txt` |

AgentLab 버전이 바뀌어 에이전트 생성에서 실패하면
`scripts/agentlab_eval.py`의 `### AGENTLAB-VERSION-SENSITIVE` 블록만 고치면 된다.
나머지 부분은 버전 안정적으로 작성돼 있다.

---

## 4. 재현

### 4.1 논문 결과 재현 시 유의점

- **자기 추론 합성(self-reasoning synthesis) 코드가 레포에 없다.** SETUP.md가 명시한다.
  Qwen3-8B의 해당 이득은 재현되지 않는다.
- **`--score-key` 기본값이 `phi`가 아니다**(3.2 참조). 논문 수식을 그대로 쓰려면
  명시해야 한다.
- **프루닝 경로에 형식 버그가 있다.** 타깃 중심 프루닝이 AXTree 첫 줄을 중복 출력하고,
  잘린 마지막 줄과 `# History...` 마커 사이 개행을 빠뜨린다. 후자 때문에
  `prepare_scores`의 AXTree 캡처가 이력 섹션까지 삼킨다. 엄밀한 재현이 필요하면
  `PRUNE=0`으로 비교군을 함께 돌려보는 것이 안전하다.
- **프루닝 통계의 폴백 건수는 신뢰할 수 없다.** 요약 출력의 `fallback_threshold=`는
  구현상 항상 0이고, 통계 JSON의 같은 이름 필드는 `임계값 + 폴백건수`가 뒤섞인 값이다.
  실제 폴백 비율이 필요하면 `centered`와 `threshold` 카운트, `total_examples`,
  `pruned_examples`로 역산해야 한다.

### 4.2 결정성

| 요소 | 결정적인가 |
|---|---|
| 프루닝 | 예 — 정규식만 |
| 그리디 선별 | 예 — 동점은 먼저 발견된 후보가 이김 |
| 후처리 서브샘플링 | 예 — `random.Random(--seed)`, 기본 시드 0 |
| BERTScore | 부동소수 수준에서 하드웨어/배치 크기에 따라 미세 변동 가능 |
| 학습 | `set_seed(42)`가 있으나 GPU 커널·DDP 랭크 수에 따라 완전 비트 재현은 아님 |
| 평가 | 아니오 — 브라우저 환경과 샘플링이 개입 |

같은 선별 결과를 다시 얻으려면 `T0`, `LAMBDA`, `WINDOW`, `FALLBACK`,
`--score-key`, `--max-examples`, `--seed`, 그리고 BERTScore 모델과 입력 파일이
같아야 한다. `--batch-size`는 결과에 영향을 주지 않아야 하지만 부동소수 누적 순서를
바꾸므로 미세 차이가 날 수 있다.

### 4.3 실험 기록에 남길 값

재현 가능한 기록을 위해 최소한 다음을 저장한다.

```
입력 파일 경로와 크기/레코드 수
prune_axtree 통계 JSON (centered / threshold / 평균 토큰 감소)
prepare_scores의 "Found N distinct goals and M trajectory segments" 줄
select_greedy의 "Selected N total steps" 줄
postprocess의 "X selected -> Y after length filter -> Z final examples" 줄
train_lora_sft의 "N/M usable records" / "K/N examples after tokenization" / 폴백 비율
weasel_summary.json (성공률, 에피소드 수, 오류 수)
```

이 값들만 있으면 어느 단계에서 데이터가 얼마나 줄었는지 사후에 추적할 수 있다.

---

## 5. 문제 해결

### 5.1 환경

**`weasel_activate: command not found`**
셸에서 `source scripts/setup_env.sh`를 하지 않았다. 새 셸/tmux 창마다 필요하다.
zsh를 쓰면 함수가 자식 프로세스로 export되지 않지만, `run_*.sh`들이 이를 감지해
스스로 재-source하므로 스크립트 실행에는 문제가 없다.

**`[setup_env] WARNINGS: the following paths do not exist yet.`**
정상이다. 아직 만들지 않은 경로를 알려주는 것뿐이고 변수는 모두 export된다.
각 항목 아래 `fix:` 줄에 만들 명령이 적혀 있다. `GROUP_VOLUME` 경고는
`WEASEL_WORK`를 지정했다면 무시해도 된다.

**venv를 만들었는데 못 찾는다**
`WEASEL_WORK`를 바꾼 뒤 `install.sh`를 다시 돌리지 않았을 가능성이 크다.
venv 경로는 `WEASEL_WORK` 아래로 파생되므로 루트가 바뀌면 새로 만들어야 한다.

### 5.2 다운로드

**BERTScore가 오프라인 오류로 죽는다 (가장 흔한 함정)**
`setup_env.sh`가 `HF_HUB_OFFLINE=1`을 기본으로 두는데, `roberta-large`를 미리
받아두는 스크립트가 **레포에 없다**. 캐시가 비었으면 첫 선별 실행이 실패한다.
첫 실행에 한해 오프라인을 끈다.

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 bash scripts/run_select.sh --gpus 0
```

또는 미리 캐시에 넣어둔다.

```bash
weasel_activate select
HF_HUB_OFFLINE=0 python -c "from bert_score import BERTScorer; BERTScorer(model_type='roberta-large', rescale_with_baseline=False, device='cpu')"
```

`select_clean`도 같은 모델을 쓰므로 동일하다.

**gemma 다운로드가 403으로 실패한다**
게이트 모델이다. `huggingface-cli login` 후 HF 웹에서 라이선스에 동의해야 한다.
`download_models.sh all`은 gemma를 **마지막**에 받으므로 앞의 세 모델은 이미 받아졌다.

**`huggingface-cli`가 없다는 오류**
hub 1.0부터 CLI 이름이 `hf`로 바뀌었다. `download_models.sh`와
`download_data.sh`는 두 이름을 순서대로 시도하므로 보통 자동 폴백된다.
그래도 실패하면 venv 활성화 여부를 확인한다.

### 5.3 선별

**"Found 1 distinct goals" — 모든 예제가 한 그룹으로 뭉쳤다**
user 프롬프트에 `## Goal:` 마커가 없어 전부 `<NO_GOAL_FOUND>`로 묶인 것이다.
`convert_dataset.py --goal-field`로 goal을 주입하거나
`convert_gemini`를 쓴다. 입력 줄마다 별도 그룹으로 만들고 싶으면
`convert_gemini --unique-goal`을 준다.

**"Pruned AXTree examples: 0/N"**
`## AXTree:`와 `# History of interaction with the task:` 두 마커가 모두 있어야
섹션 정규식이 매치된다. 하나라도 없으면 `missing_axtree`로 집계된다.
AXTree가 없는 데이터라면 `PRUNE=0`으로 단계를 건너뛴다.

**`prepare_scores`가 CUDA OOM으로 죽는다**
`BATCH`를 줄인다(예: 64 → 16). 세그먼트가 크면 쌍별 행렬 계산이 지배적이므로,
`--skip-response-distance`로 응답 행렬을 빼면 계산량이 절반이 된다.

**`prepare_scores`가 너무 오래 걸린다**
세그먼트 크기 `n`에 대해 유사도 호출이 행렬당 `n(n−1)`회, 두 행렬이면
`2n(n−1)`회다. `n`이 커지면 제곱으로 늘어난다. `--skip-response-distance`를 쓰거나,
`convert_gemini --unique-goal`로 그룹을 작게 쪼개는 것이 실질적인 대응이다.

**`select_greedy`가 "Missing 'distance_matrix'"로 종료**
`prepare_scores` 출력이 아닌 파일을 넘겼거나 산출이 중간에 끊겼다.

**`postprocess_dataset`가 `IndexError`**
선택 인덱스가 데이터셋 범위를 벗어났다. `--dataset`에 넘긴 파일이
`prepare_scores`에 넣었던 파일과 다른 경우가 대부분이다. 프루닝을 켰다면
프룬본을, 껐다면 원본을 넘겨야 한다.

**`select_clean`에서 중복이 하나도 제거되지 않는다**
Jaccard 히스토그램을 본다. 분포가 전부 낮은 구간에 있으면 실제로 중복이 없는 것이고,
임계값 바로 아래에 봉우리가 있으면 `--near-dup-threshold`를 낮춘다.
rollouts/task 히스토그램에서 `1=` 값이 대부분이면 태스크당 롤아웃이 하나뿐이라
애초에 제거할 중복이 없다. 이 경우 `--task-field`가 잘못 지정돼
태스크가 과도하게 잘게 쪼개졌을 가능성도 확인한다.

### 5.4 학습

**"all examples were dropped; check --cutoff-len / data"**
손실이 assistant 턴에만 걸리므로, assistant 토큰이 전부 컷오프 뒤로 밀린 예제는
버려진다. 시스템 프롬프트가 긴 FC 내보내기(2만 토큰 규모)에서는 기본 8192로
모든 예제가 사라진다. `CUTOFF=32768`을 쓴다.

**"N examples (X%) fell back to full-sequence loss"**
채팅 템플릿이 이전 턴을 다시 쓰기 때문에 증분 렌더가 전체 렌더의 접두가 되지 않는
경우다. 해당 샘플은 프롬프트 토큰에도 손실이 걸린다. 비율이 높으면
다른 템플릿을 쓰거나 데이터 형태를 조정해야 한다.

**"tokenizer has no chat template"**
체크포인트의 토크나이저에 템플릿이 없다. 이 트레이너는 템플릿으로만 포맷을 만들므로
템플릿이 있는 체크포인트를 쓰거나 토크나이저에 템플릿을 넣어야 한다.

**CUDA OOM (학습)**
순서대로 시도한다: `--liger` → `CUTOFF` 축소 → `PER_DEVICE=1` 유지 →
`--qlora`. 전역 배치는 accum으로 자동 보정되므로 GPU를 줄여도 레시피는 유지된다.

**`--liger`를 켰는데 메모리가 그대로다**
경고 줄에 `liger-kernel has no patch for model_type='...'`가 있는지 본다.
미지원 모델 타입이면 transformers가 조용히 no-op하므로 융합 CE가 적용되지 않는다.

**`--parallel`로 아무것도 안 뜬다**
베이스 모델 디렉터리가 전부 없어 스킵된 것이다. 스킵될 때마다
`[skip] base model missing for <model>: <경로>`와 다운로드 명령이 stderr에 찍힌다.

### 5.5 서빙·평가

**`[error] merged model not found`**
`$MERGED_ROOT/<model>/<variant>`가 없다. 같은 `VARIANT`로 `run_merge.sh`를
먼저 돌렸는지 확인한다.

**AgentLab이 에이전트 생성에서 실패한다**
설치된 AgentLab 버전과 클래스명이 다르다.
`agentlab_eval.py`의 `### AGENTLAB-VERSION-SENSITIVE` 블록만 고친다.

**`webarena_lite`가 태스크 목록이 없다며 종료한다**
의도된 동작이다. 165태스크 목록 파일을 `configs/webarena_lite_tasks.txt`에 두거나
`WEBARENA_LITE_TASKS=<파일>`로 지정한다. 한 줄에 id 하나,
`#` 뒤는 주석, `.`이 없으면 `webarena.` 접두가 자동으로 붙는다.
전체를 돌리려면 `--bench webarena`를 쓴다.

**리포트에 "Trajectory detail unavailable" 배너가 뜬다**
eval venv가 아닌 곳에서 `miniwob_report.py`를 돌렸다.
`weasel_activate eval` 후 다시 생성한다.

**`[summarize] no study with summary_info.json found`**
평가가 에피소드를 하나도 완료하지 못했거나 `--root`가 잘못됐다.
`$EVAL_RESULTS_ROOT/<variant>` 아래를 직접 확인하고,
필요하면 `--study-dir`로 정확한 경로를 지정한다.

**성공률이 0%인데 오류는 없다**
`weasel_summary.json`의 `n_errored`를 먼저 본다. 0이 아니면 에피소드가
실행 중 죽은 것이다. HTML 리포트의 실패 모드 표에서
`out of steps (truncated)`가 지배적이면 최대 스텝에 걸린 것이고,
`action-exec errors`가 많으면 모델이 유효하지 않은 액션 문법을 내는 것이다.
후자는 컷오프가 짧아 학습이 제대로 안 된 경우와 자주 겹친다.

**Playwright/Chromium이 라이브러리 부족으로 실행되지 않는다**
루트가 있으면 `playwright install-deps chromium`을 sudo로 한 번 실행한다.
없으면 X11 라이브러리를 사용자 권한으로 마련한 뒤 `WEASEL_XLIBS_DIR`을
그 `lib` 디렉터리로 지정하고 셸을 다시 source한다.

**SSH가 끊기면 서버가 죽는다**
tmux 창에서 띄운다(2.1 참조). `run_experiment.sh`는 서버를 백그라운드로
띄우고 `EXIT` 트랩으로 정리하므로, 그 스크립트 자체를 tmux 안에서 돌린다.

### 5.6 안내 문구가 틀린 곳

**`download_data.sh agenttrek` 끝의 "prune_axtree is MISSING from the repo" 안내**
낡은 문구다. `weasel/prune_axtree.py`는 존재하고 `run_select.sh`가 이를 기본
단계 0으로 실행한다. 무시해도 된다.

**`select_clean --help`의 "Signature Jaccard"**
용어를 `fingerprint`로 통일한 뒤 도움말 문자열만 남았다. 동작에는 영향 없다.

**`_overlay_text_into_mm.py`의 기본 경로**
특정 머신의 절대경로가 하드코딩돼 있다. 항상 `--base` / `--text` / `--out`을
명시해서 쓴다.
