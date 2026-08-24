# CODE — 코드 구조

파일별·함수별 역할, 핵심 로직의 상수와 수식 대응, 의존 관계를 정리한다.
모든 줄 번호와 기본값은 현재 커밋 기준 실측치다.

---

## 0. 파일 인벤토리

추적 파일 35개 / 5,864 LOC. 이미지 1개(`assets/weasel_overview.png`)를 제외한
34개가 텍스트다.

### 0.1 알고리즘 패키지 `weasel/` — 8 파일 2,440 LOC

| 파일 | LOC | 함수 수 | 외부 의존 |
|---|---:|---:|---|
| `prepare_scores.py` | 580 | 22 | `bert_score`, `torch`, `tqdm`(선택) |
| `select_clean.py` | 471 | 15 (+중첩 1) | `bert_score`(선택), `torch`(선택) |
| `convert_gemini.py` | 406 | 13 | 표준 라이브러리만 |
| `prune_axtree.py` | 362 | 11 | 표준 라이브러리만 |
| `select_greedy.py` | 247 | 9 | 표준 라이브러리만 |
| `postprocess_dataset.py` | 212 | 8 | 표준 라이브러리만 |
| `select_trajectories.py` | 161 | 5 | 표준 라이브러리만 |
| `__init__.py` | 1 | 0 | — |

### 0.2 스크립트 `scripts/` — 22 파일 2,744 LOC

Python 10개(1,639 LOC), Shell 12개(1,105 LOC). 상세는 3장·4장.

### 0.3 루트 문서 — 3 파일 658 LOC

`README.md`(263), `SETUP.md`(311), `EXPERIMENTS.md`(84).

---

## 1. `weasel/prune_axtree.py` — 362 LOC

논문 단계 0. 학습 예제의 `user` 프롬프트 안에 있는 AXTree 섹션을 잘라낸다.

### 1.1 모듈 상수

```python
AXTREE_SECTION_RE = re.compile(
    r"(##\s*AXTree:\s*)(.*?)(# History of interaction with the task:?[\t ]*)", re.S)
ACTION_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.S)
THINK_RE  = re.compile(r"<think>\s*(.*?)\s*</think>", re.S)
```

섹션 정규식은 3개 그룹으로 나뉜다. 그룹 2가 잘라낼 본문이고, 그룹 3(History 마커)이
**끝 경계**다. History 마커가 없으면 매치 자체가 실패하므로 프루닝이 일어나지 않는다.

### 1.2 함수별 역할

| 함수 | 역할 |
|---|---|
| `parse_args()` | `--input --output --window-size(60) --fallback-threshold(120) --stats-output --indent(2)` |
| `load_jsonl_or_json(path)` | 전체를 JSON 배열로 먼저 시도하고, 실패하면 줄 단위 JSONL로 재시도. 둘 다 실패 시 `ValueError` |
| `write_json(data, path, indent)` | 부모 디렉터리 생성 후 `ensure_ascii=False`로 저장. `indent=0`이면 `None`(압축) |
| `get_message_content(item, role, fallback_index, last)` | 역할로 메시지를 찾고, 없으면 위치 인덱스로 폴백. `last=True`면 마지막 매치 |
| `set_user_prompt(item, content)` | 첫 `user` 메시지 내용 교체, 실패 시 `messages[1]` 교체 |
| `extract_bid_from_action(text)` | `<action>`에서 함수명과 대상 bid 추출 |
| `extract_axtree_stats(text)` | `(본문, bid 개수, 토큰 개수)` 반환 |
| `prune_threshold(text, threshold)` | 접두에서 bid를 세며 `threshold`개에 도달하면 절단 |
| `prune_gold_centered(text, bid, window)` | 대상 bid 인덱스 기준 좌우 `window`개 bid만 유지 |
| `prune_item(item, window, fallback)` | 위 두 전략 중 선택 + 예제 갱신, `(성공, 방법, 원본토큰, 프룬토큰)` 반환 |
| `main()` | 전량 로드 → 예제별 프루닝 → 통계 집계/출력 |

### 1.3 `extract_bid_from_action` — 액션 종류별 bid 위치

액션 텍스트에서 `^([a-zA-Z_]+)\(` 로 함수명을 뽑은 뒤 분기한다.

| 액션 | bid 추출 정규식 | 비고 |
|---|---|---|
| `click` `dblclick` `hover` `press` `focus` `clear` | `{name}\('([\w\d]+)'` | 첫 인자 |
| `fill` | `fill\('([\w\d]+)',` | 첫 인자, 뒤에 값이 따름 |
| `select_option` | `select_option\('([\w\d]+)',` | 첫 인자 |
| `drag_and_drop` | `drag_and_drop\('(\w+)',\s*'(\w+)'\)` | **두 bid 중 숫자가 큰 쪽** 선택. `int(re.search(r"\d+", v).group())`로 비교하고 예외 시 첫 번째 |
| `upload_file` | `upload_file\('([\w\d]+)'` | 첫 인자 |
| `noop` `scroll` `send_msg_to_user` `go_back` `go_forward` `goto` | — | bid 없음 → 임계 방식 |
| 그 외 | — | 이름만 반환, bid `None` |

### 1.4 `prune_gold_centered` 알고리즘

1. 본문에서 `\[([^\]]+)\]` 전부를 순서대로 모아 `bid_list` 생성
2. 대상 bid가 없으면 원문 그대로 반환(변화 없음 → 호출자가 폴백 판단)
3. `index = bid_list.index(bid)`, `start = max(index − window, 0)`,
   `end = min(index + window + 1, len)` → `bid_subset` 집합 구성
4. 줄을 순회하며:
   - bid가 있고 subset에 속하면 채택하고 `started = True`
   - `started` 이후 bid 없는 줄(들여쓰기 연속 텍스트)은 함께 채택
   - `started` 이후 subset 밖 bid 줄을 만나면 즉시 `break`

즉 **연속 구간 하나**만 남는다. 윈도 크기 60은 대상 bid 좌우 각각 60개(최대 121개)를
의미한다.

#### 실측 이슈 A — 첫 줄 중복

반환문이 다음과 같다.

```python
return before + lines[0] + "\n" + pruned_axtree + after
```

`lines[0]`은 AXTree 본문의 첫 줄이며, 루트 노드 헤더를 항상 남기려는 의도로 보인다.
그러나 윈도가 첫 줄의 bid를 포함하면 `pruned_axtree`에도 이미 그 줄이 들어 있어
**두 번 출력된다**. 재현:

```python
from weasel.prune_axtree import prune_gold_centered
axtree = "\n".join(f"[{i}] button 'b{i}'" for i in range(1, 200))
user = "## Goal: g\n\n## AXTree: \n" + axtree + "\n# History of interaction with the task:\nnone\n"
out = prune_gold_centered(user, "5", 60)
# 결과 첫 세 줄: ["[1] button 'b1'", "[1] button 'b1'", "[2] button 'b2'"]
```

#### 실측 이슈 B — History 마커 앞 개행 누락

같은 반환문에 `pruned_axtree`와 `after` 사이 개행이 없다. `after`는
`# History of interaction with the task:`로 시작하므로 결과가 이렇게 된다.

```
[5] button 'b5'# History of interaction with the task:
```

`prune_threshold`는 `f"{before}{pruned_axtree}\n{after}"`로 개행을 넣으므로 두 경로의
출력 형식이 서로 다르다. 이 차이는 하류에 영향을 준다. `prepare_scores`의

```python
AXTREE_RE = re.compile(r"##\s*AXTree:\s*(.*?)(?=\n##\s|\n#\s|\Z)", re.S)
```

는 종료 조건으로 `\n#`를 요구하므로, 개행이 없으면 AXTree 캡처가 History 섹션을 넘어
다음 `\n#`까지 확장된다. 즉 centered로 프룬된 예제는 상태 텍스트에 상호작용 이력이
섞여 들어간다.

### 1.5 `prune_item` 전략 선택

```
액션에서 bid 추출
├─ bid 있음 → prune_gold_centered
│   ├─ 결과 본문이 원본과 다름          → method = "centered"
│   └─ 결과가 원본과 동일(= bid 미발견)  → prune_threshold, method = "fallback_threshold"
└─ bid 없음 → prune_threshold,           method = "threshold"
```

실측 확인(bid 199개짜리 합성 트리, window 60 / fallback 120):

| 케이스 | 액션 | 판정 | 토큰 |
|---|---|---|---|
| A | `click('100')` | `centered` | 1193 → 731 |
| B | `scroll(0, 200)` | `threshold` | 1193 → 719 |
| C | `click('9999')`(트리에 없는 bid) | `fallback_threshold` | 1193 → 719 |

### 1.6 `main()` 통계 — 실측 이슈 C (키 충돌)

통계 딕셔너리가 이렇게 초기화된다.

```python
"threshold": 0,
"fallback_threshold_count": 0,
...
"fallback_threshold": args.fallback_threshold,   # 값 120 (카운터가 아님)
```

그런데 집계는 `stats[method] += 1`로 하고, `prune_item`이 반환하는 방법 문자열은
`"fallback_threshold"`다. 따라서 폴백이 일어날 때마다 **임계값 상수가 증가**하고
`fallback_threshold_count`는 영원히 0이다. 요약 출력은

```python
f"fallback_threshold={stats['fallback_threshold_count']}"
```

를 쓰므로 **항상 0**으로 표시된다. 실측: 폴백 1회 후 딕셔너리가
`{'fallback_threshold': 121, 'fallback_threshold_count': 0}`이 된다.

수정하려면 `prune_item`이 `"fallback_threshold_count"`를 반환하거나
`main()`에서 매핑하면 된다.

### 1.7 토큰 계수 방식

```python
tokens = re.findall(r"\S+|\s+", full_axtree)
```

공백 덩어리도 하나의 토큰으로 센다. 따라서 보고되는 "AXTree tokens"는 단어 수의
대략 두 배다(실측: 3단어 × 199줄 ≈ 597단어 → 1,193 토큰). 절대값이 아니라
프루닝 전후 **비율**을 볼 지표로 해석해야 한다.

---

## 2. `weasel/prepare_scores.py` — 580 LOC

가장 큰 모듈이자 유일한 GPU 무거운 선별 단계. 이전에 3개 스크립트로 나뉘어 있던
전처리를 하나로 합쳤다고 docstring에 적혀 있다.

### 2.1 텍스트 추출 정규식

```python
GOAL_RE        = r"##\s*Goal:\s*(.*?)(?=\n##\s|\n#\s|\Z)"
AXTREE_RE      = r"##\s*AXTree:\s*(.*?)(?=\n##\s|\n#\s|\Z)"
OBS_HISTORY_RE = r"# Observation of current step:\s*(.*?)(?=\n# Action space:|\Z)"
```

`extract_axtree`는 `finditer`로 **모든** AXTree 섹션을 찾아 `\n\n`으로 이어붙인다.
`extract_goal`은 매치 실패 시 `"<NO_GOAL_FOUND>"`를 반환하는데, 이 문자열이
그룹 키로 쓰이므로 goal 마커가 없는 예제들은 전부 **하나의 거대한 궤적**으로 뭉친다.
`run_experiment.sh`가 실행 전 `grep -q "## Goal:"`로 경고하는 이유가 이것이다.

### 2.2 필드 선택기 `text_field(item, field)`

| `field` 값 | 반환 |
|---|---|
| `axtree` | user 프롬프트에서 추출한 AXTree 전부 |
| `obs_history` | `# Observation of current step:` 섹션 |
| `user_prompt` | user 메시지 원문 |
| `assistant` | 마지막 assistant 메시지 원문 |
| `reasoning` | assistant의 `<think>` 내용 |
| `action` | assistant의 `<action>` 내용 |
| `assistant_without_think` | assistant에서 `<think>…</think>` 제거 |

CLI 기본값: `--phi-field obs_history`, `--state-field axtree`, `--response-field assistant`.

### 2.3 궤적 그룹핑

```python
group_trajectories(data)      # goal 텍스트 → 데이터셋 인덱스 목록
split_contiguous(indices)     # 인덱스가 연속인 구간을 세그먼트로 분할
```

같은 goal이라도 데이터셋에서 인덱스가 떨어져 있으면 별개의 세그먼트가 된다.
"연속 인덱스 = 같은 롤아웃"이라는 가정이며, 원본 파일이 궤적 순서대로 나열돼 있을 때만
성립한다. 점수/거리 계산과 그리디 선별은 **세그먼트 단위**로 이뤄진다.

### 2.4 BERTScore 호출 구조

```python
load_bert_scorer(model_type, device)
    → BERTScorer(model_type=..., rescale_with_baseline=False, device=...)
```

`device`가 `None`이면 `torch.cuda.is_available()`로 자동 결정한다.
`--lang`은 파싱되어 `score_metadata`에만 기록되고 **생성자에 전달되지 않는다**(실측).
`model_type`이 명시되고 baseline 재조정이 꺼져 있으므로 실동작 영향은 없다.

`score_pairs(scorer, candidates, references, batch_size)`는 후보/참조 길이를 검증한 뒤
`batch_size`만큼 잘라 `scorer.score(...)`를 호출하고 F1만 뽑아 리스트로 이어붙인다.

### 2.5 쌍별 유사도 행렬 — `pairwise_similarity_matrix`

BERTScore는 비대칭(`score(a,b) ≠ score(b,a)`)이므로 대칭화한다.

```python
similarity(i, j) = (F1(texts[i] | texts[j]) + F1(texts[j] | texts[i])) / 2
matrix[i][i] = 1.0
```

`(i, j)` 쌍을 `pair_batch`에 모으고 `batch_size`에 도달하면 flush한다.
flush 시 각 쌍마다 후보/참조를 양방향 2개씩 넣으므로 실제 문장 쌍 개수는
`2 × len(pair_batch)`가 된다. 세그먼트 크기 `n`에 대한 총 호출은 행렬당 `n(n−1)`회다.
`n == 0`이면 `[]`, `n == 1`이면 `[[1.0]]`을 조기 반환한다.

### 2.6 거리와 정규화

```python
distance[i][j] = 1 − sims_states[i][j]                       # --skip-response-distance
distance[i][j] = max(1 − sims_states[i][j],
                     1 − sims_responses[i][j])               # 기본
```

`max`를 쓰는 이유는 논리적으로 "상태가 비슷해도 응답이 다르면 다른 스텝"이기 때문이다.
둘 중 더 큰 거리를 채택하는 보수적 다양성 판정이다.

정규화 함수 3종:

```python
minmax_normalize(v)     # (x − min) / (max − min);  max == min이면 전부 0.0
sum_normalize(v)        # x / Σx;                   Σ == 0이면 전부 0.0
phi_from_relevance(r)   # phi[t] = max(0, r[t] − r[t−1]),  r[−1] = 0
```

`phi_from_relevance`의 `previous` 초기값이 `0.0`이므로 **첫 스텝의 phi는 `r[0]` 그 자체**다.
관측 이력이 비어 있는 첫 스텝의 BERTScore가 그대로 첫 phi가 된다는 뜻이다.

### 2.7 출력 스키마

goal 레코드 하나의 구조:

```jsonc
{
  "goal": "...",
  "dataset_indices": [정렬된 전역 인덱스],
  "score_metadata": {"phi_field","state_field","response_field","model_type","lang"},
  "bert_scores":            [r],
  "bert_scores_norm":       [r_norm],
  "phi_scores_raw":         [phi],
  "phi_scores_norm":        [phi_norm],
  // 구 스크립트 호환용 별칭 — 위 4개와 값이 동일
  "bert_scores_obs_history":      [...],
  "bert_scores_obs_history_norm": [...],
  "phi_scores_obs_history_raw":   [...],
  "phi_scores_obs_history_norm":  [...],
  "trajectory_groups": [
    {"dataset_indices", "bert_scores", "bert_scores_norm", "phi_raw", "phi_norm",
     "sims_states", "sims_responses", "distance_matrix",
     "state_field", "response_field", "phi_field"}
  ],
  "user_prompts": [...]   // --include-prompts 일 때만
}
```

별칭 4개는 상류 구버전 스크립트가 `*_obs_history*` 이름을 기대해서 남겨둔 것이다.
`--phi-field`를 `axtree`로 바꿔도 이름은 여전히 `obs_history`라서 오해의 소지가 있다.

`--augmented-dataset-output`을 주면 원본 데이터셋의 각 예제에 8개 필드
(`obs_history_bert_score_r`, `..._r_norm`, `obs_history_phi_raw`, `obs_history_phi_norm`
및 접두어 없는 4개)를 붙여 별도 파일로 저장한다. 점수가 계산되지 않은 예제
(`r is None`)는 건너뛴다.

### 2.8 메모리 특성

거리 행렬이 `n × n` 부동소수 리스트로 goal 레코드 JSON에 그대로 직렬화된다.
세그먼트가 커질수록 출력 파일이 제곱으로 커진다. `prepare_scores`는 입력 전량과
출력 전량을 메모리에 올린다.

---

## 3. `weasel/select_greedy.py` — 247 LOC

`prepare_scores` 산출물만 읽는다. 원본 데이터셋을 보지 않는다.

### 3.1 CLI 기본값

| 인자 | 기본값 | 의미 |
|---|---|---|
| `--score-key` | `bert_scores_obs_history_norm` | 단항 중요도로 쓸 키 |
| `--distance-key` | `distance_matrix` | 궤적 그룹 안의 거리 행렬 키 |
| `--t0-mode` | `fixed` | `fixed` \| `percentage` |
| `--t0-fixed` | `3` | 궤적당 선택 스텝 수 |
| `--t0-percentage` | `0.25` | percentage 모드 비율 |
| `--lambda-weight` | `1.0` | 다양성 가중 λ |
| `--flat` | off | 평탄 리스트로 저장 |
| `--annotated-output` | 없음 | 선택 결과를 붙인 입력 사본 |

**기본 `--score-key`는 `r_norm`이지 `phi`가 아니다**(2.7의 별칭 표 참조).
논문 서술의 importance를 그대로 쓰려면 `--score-key phi_scores_obs_history_norm`을
명시해야 한다. `run_select.sh`는 넘기지 않는다.

### 3.2 `compute_t0(length, mode, fixed, percentage)`

```python
length <= 0                    → 0
mode == "fixed"                → clamp(fixed, 1, length)
mode == "percentage"           → clamp(ceil(length * percentage), 1, length)
```

실측: `compute_t0(5, "percentage", 3, 0.25) = 2`, `compute_t0(9, ...) = 3`.

### 3.3 `greedy_select(importance, distance, t0, lambda_weight)`

```
n = len(importance)
n == 0 or t0 <= 0  → []
n <= t0            → 전부 선택 (거리 행렬 미검증)
t0 == 1            → argmax importance 하나 (거리 행렬 미검증)
그 외              → validate_square_matrix 후 본 알고리즘
```

본 알고리즘:

```
시드:  (i*, j*) = argmax_{i<j}  u_i + u_j + λ · d(i,j)
반복:  c*      = argmax_{c∉S}  u_c + λ · Σ_{s∈S} d(c,s)
       |S| = t0 이 될 때까지
반환:  sorted(S)   # 원래 궤적 순서로 복원
```

동점 처리는 엄격 부등호(`>`)라 **먼저 발견된 후보**가 이긴다. 실측 검증
(importance `[0.9, 0.85, 0.1, 0.8, 0.05]`, 거리 대칭 행렬):

| t0 | 선택 결과 |
|---:|---|
| 1 | `[0]` |
| 2 | `[0, 2]` — `(0,2)`와 `(0,4)`가 1.9로 동점, 먼저 나온 `(0,2)` 채택 |
| 3 | `[0, 1, 2]` |
| 5 | `[0, 1, 2, 3, 4]` |
| 7 | `[0, 1, 2, 3, 4]` (`n <= t0` 경로) |

계산 복잡도: 시드 탐색 `O(n²)`, 이후 각 추가가 `O(n·|S|)`이므로 총
`O(n² + t0²·n)`. `t0 = 3`에서는 시드 탐색이 지배한다.

### 3.4 `segment_importance(record, score_key, segment_indices)`

goal 레코드의 `dataset_indices`와 `score_key` 배열을 위치로 짝지어
`{데이터셋 인덱스: 점수}` 사전을 만든 뒤, 세그먼트 인덱스 순서대로 값을 꺼낸다.
사전에 없는 인덱스는 `0.0`으로 채운다(조용한 폴백).

### 3.5 `select_from_records`

레코드 → `trajectory_groups` → 그룹별로 `t0` 계산 → `greedy_select` → 로컬 인덱스를
전역 데이터셋 인덱스로 환원. 각 그룹에 `selected_dataset_indices`와
`selection_metadata`(`t0`, `score_key`, `distance_key`, `lambda_weight`)를 붙인다.
거리 행렬 키가 없으면 `ValueError`로 즉시 중단한다. 세그먼트가 비면 빈 리스트를 넣고
`selected_groups`에는 추가하지 않는다.

출력은 기본적으로 **그룹별 중첩 리스트**, `--flat`이면 평탄 리스트다.
`postprocess_dataset`이 어느 쪽이든 평탄화하므로 실무상 차이는 없다.

---

## 4. `weasel/postprocess_dataset.py` — 212 LOC

| 함수 | 역할 |
|---|---|
| `flatten_selected_indices(raw)` | 중첩/평탄 모두 받아 `int` 리스트로 평탄화 |
| `dedupe_indices(indices)` | 첫 등장만 유지(`--dedupe`일 때만 호출) |
| `user_prompt(item)` | 첫 `user` 메시지, 없으면 `messages[1]` |
| `validate_indices(indices, size)` | 범위 밖 인덱스가 있으면 개수와 앞 10개를 담아 `IndexError` |
| `main()` | 회수 → 길이 필터 → 서브샘플 → 저장 → 통계 |

처리 순서와 기본값:

1. 인덱스 평탄화, `--dedupe`면 중복 제거
2. `validate_indices` — 범위 검증 (프루닝이 레코드 수를 보존하는 것이 여기서 중요)
3. `len(user_prompt(item)) <= --max-user-chars`(**기본 40,000자**)만 통과
4. 남은 수가 `--max-examples`(**기본 10,000**)를 넘으면
   `random.Random(--seed).sample(...)`로 균등 추출
5. `--preserve-order`면 데이터셋 인덱스 오름차순 정렬, 아니면 **샘플 순서 유지**

통계 JSON(`--stats-output`) 필드: `dataset_size`, `selected_indices`,
`after_length_filter`, `final_size`, `max_user_chars`, `max_examples`, `seed`,
`dedupe`, `preserve_order`. `after_length_filter`는 서브샘플 이전 값을
**원본 인덱스를 다시 훑어** 재계산한다(중복 인덱스가 있으면 중복 계수됨).

> `run_select.sh`는 `--dedupe`도 `--preserve-order`도 넘기지 않는다.
> 세그먼트끼리 인덱스가 겹치지 않으므로 중복은 실제로 발생하지 않는다.

---

## 5. `weasel/convert_gemini.py` — 406 LOC

FC 내보내기(1줄 = 1궤적) → WEASEL 입력 두 형태.

### 5.1 공통 유틸

| 함수 | 역할 |
|---|---|
| `to_text(content)` | `str` / `None` / 블록 리스트 / dict를 텍스트로 정규화. 리스트는 `text` 또는 `content` 키를 뽑아 `\n` 결합 |
| `iter_jsonl(path)` | `[`로 시작하면 JSON 배열, 아니면 JSONL. 파손 줄은 stderr 경고 후 건너뜀 |
| `_args_to_text(args)` | 문자열이면 JSON 파싱 시도 후 `json.dumps(sort_keys=True)`, 실패 시 원문 strip |
| `_args_to_obj(args)` | 네이티브 FC 출력용 객체 파싱 |
| `serialize_action(msg)` | `<think>…</think>` + 텍스트 + `<action>name(args)</action>` 조립 |
| `first_role(messages, role)` | 첫 해당 역할 메시지 텍스트 |
| `tool_names(record)` | `tools[].function.name`을 `", "`로 연결 |
| `clip(text, limit)` | 초과분 절단 + `"\n...[truncated N chars]"` 부기 |
| `source_meta(record)` | `__source_task__`/`__source_agent__` → `_source_task`/`_source_agent` |

### 5.2 상수 `EMPTY_OBS`

```python
EMPTY_OBS = "(no observation)"
```

주석이 이유를 밝힌다. 관측이 비면 `prepare_scores`의 `\s*` 정규식이 경계 개행을
삼켜 다음 마커가 캡처 필드로 새어 들어온다. 자리표시자를 넣어 섹션을 비지 않게 한다.

### 5.3 `build_steps` — 모드 (a) 스텝 단위

각 assistant 액션마다 레코드 하나. 조립되는 user 본문:

```
## Goal: {goal}[ (traj#{id})]

## AXTree:
{직전 tool 관측 하나, clip(max_obs_chars)}  또는 EMPTY_OBS

# Observation of current step:
{누적 관측 전체 "\n\n" 결합, clip(max_history_chars)}  또는 EMPTY_OBS
# Action space:
{tool 이름 목록}
```

`## AXTree:`에 **직전 관측(현재 상태)**, `# Observation of current step:`에
**누적 이력**이 들어간다. 이름과 내용이 반대로 보이지만 하류의 필드 기본값
(`--state-field axtree`, `--phi-field obs_history`)과 맞춘 결과다.
즉 거리는 현재 상태로, 중요도는 누적 이력으로 계산된다.

`--unique-goal`을 주면 goal 뒤에 `(traj#<id>)`를 붙여 입력 줄마다 별도 궤적 그룹이
되게 한다. 기본은 goal 원문으로 묶으므로 같은 태스크의 여러 롤아웃이 한 그룹이 된다.

출력 레코드: `{"messages": [...], "_traj_id": tid, "_step": n}` + provenance 메타.
goal이 없으면 빈 리스트를 반환해 해당 궤적을 건너뛴다.

### 5.4 `build_traj` — 모드 (b) 네이티브 FC

역할 매핑:

| 원본 역할 | 출력 역할 | 내용 |
|---|---|---|
| `system` / `user` | 그대로 | 텍스트 |
| `tool` | `observation` | `clip(text, traj_max_obs_chars)` |
| `assistant` + `tool_calls` | `function_call` | `{"name","arguments"}` 를 JSON 직렬화. 호출 1개면 객체, 여러 개면 배열 |
| `assistant` + 텍스트만 | `assistant` | 텍스트 |

`user` 턴이 없거나 `assistant`/`function_call` 타깃이 없으면 `None`을 반환해 버린다
(학습 대상이 없는 프롬프트 전용 궤적 제거). `tools`는 **JSON 문자열**로 붙인다 —
`train_lora_sft.normalize_record`가 문자열 `tools`를 파싱하도록 되어 있다.

### 5.5 CLI 기본값과 통계

| 인자 | 기본값 |
|---|---|
| `--mode` | `both` |
| `--max-obs-chars` | 4000 (스텝 상태) |
| `--max-history-chars` | 8000 (스텝 누적 이력) |
| `--max-system-chars` | 0 (제한 없음) |
| `--traj-max-obs-chars` | 0 (제한 없음, 권장) |
| `--unique-goal` / `--limit` / `--stats-output` | off / 없음 / 없음 |

`gid`(전역 궤적 id)는 여러 입력 파일에 걸쳐 단조 증가하므로, 여러 내보내기를
하나의 풀로 합쳐도 `_traj_id`가 충돌하지 않는다. 1,000궤적마다 stderr에 진행 상황을,
파일마다 요약을, 끝에 전체 통계 JSON을 출력한다.

---

## 6. `weasel/select_trajectories.py` — 161 LOC

스텝 단위 선별 결과를 궤적 단위로 되돌린다.

| 함수 | 역할 |
|---|---|
| `iter_records(path)` | JSONL/JSON배열 판별. **파싱 성공 레코드만 카운트**하는 인덱스 yield |
| `selected_traj_ids(path)` | 선택된 스텝 파일에서 `_traj_id` 집합 수집. 누락 개수 경고 |
| `filter_by_field(traj_path, keep, out, strip_meta)` | 변환본을 `_traj_id`로 필터. `--strip-meta`면 필드 제거 |
| `filter_originals(inputs, keep, out)` | 원본 파일들을 순서대로 재읽어 러닝 인덱스가 `keep`에 있으면 **그대로** 출력 |
| `main()` | 인자 검증 후 두 모드 중 하나 또는 둘 다 실행 |

인자 검증(모두 즉시 종료):
`--traj-dataset`과 `--original-input` 중 최소 하나 필요,
`--traj-dataset`에는 `--output`, `--original-input`에는 `--original-output` 필수,
존재하지 않는 파일 경로.

핵심 규약: `filter_originals`의 러닝 카운터는 `convert_gemini`가 `_traj_id`를 부여한
방식과 같아야 한다. 따라서 **같은 파일을 같은 순서로** 넘겨야 하며, 변환 시
`--limit`을 썼다면 어긋난다(디버그 전용인 이유).

---

## 7. `weasel/select_clean.py` — 471 LOC

원본 FC 파일에 대한 원패스 큐레이션. 파이프라인 A/B를 쓰지 않는 대안 경로다.

### 7.1 입출력 규약

- 입력: JSONL 또는 JSON 배열, **1줄 = 1궤적**, 여러 파일 허용(순서대로 연결)
- 출력: **원본 스키마 그대로** JSONL. 필드 추가·삭제·변형 없음

### 7.2 답변 언어 필터

```python
_HANGUL = ((0xAC00,0xD7A3), (0x1100,0x11FF), (0x3130,0x318F))
_CJK    = ((0x4E00,0x9FFF), (0x3400,0x4DBF), (0xF900,0xFAFF))

passes_answer_lang(answer, "ko")  → hangul_count >= cjk_count
passes_answer_lang(answer, "zh")  → cjk_count >= hangul_count
```

문자 스크립트 비율만 보는 무의존 판정이다. 라틴 문자만 있거나 빈 답변은 두 카운트가
모두 0이라 항상 통과한다. `--answer-lang`을 주지 않으면 필터 자체가 꺼진다.

### 7.3 궤적 순회 — `walk_trajectory`

메시지를 순서대로 훑으며:

- `role == "tool"` → `clip(내용, --max-obs-chars)`를 `history`에 추가
- `role == "assistant"`:
  - `tool_calls`가 있으면 그 시점의 **누적 history**를 `obs_histories`에 스냅샷하고
    `_action_repr` 결과를 `action_tokens`에 추가 → 이것이 하나의 importance 스텝
  - 텍스트 내용이 있으면 `final_answer`를 갱신(마지막 것이 최종 답변)

반환: `(goal, obs_histories, action_tokens, final_answer, n_steps)`.
첫 액션의 `obs_histories[0]`은 아직 tool 결과가 없으므로 빈 문자열이다.

`_action_repr(msg, arg_chars)`는 `name(arguments)` 형태의 압축 문자열을 만든다.
인자가 문자열이 아니면 `json.dumps(sort_keys=True)`로 안정 직렬화하고
`--arg-chars`(기본 60)로 자른다.

### 7.4 지문(fingerprint)과 Jaccard

```python
fingerprint(actions, answer, answer_shingle=5) =
    set(action_tokens) ∪ { "ans:" + " ".join(words[i:i+5]) }
```

`words`는 `re.findall(r"\w+", answer.lower())`. 토큰이 하나도 없으면
`{"<empty>"}`로 대체해 빈 집합을 피한다.

```python
jaccard(a, b) = |a ∩ b| / |a ∪ b|      # 합집합이 비면 1.0
```

### 7.5 중요도 — `quality_from_steps`

빈 obs_history는 BERTScore를 호출하지 않고 `r = 0`으로 직접 처리한다(빈 문자열을
넣으면 스코어러가 불안정해지기 때문). 비어 있지 않은 스텝만 평탄화해 한 번에
배치 채점한 뒤 궤적별로 복원한다.

```python
quality = r[-1]                       # --quality final
quality = mean(phi_from_relevance(r)) # --quality meanphi (기본)
```

`mean(phi)`는 스텝 수로 나누므로, 최종 관련도가 같다면 **짧은 궤적이 높은 품질**을
받는다. `--no-importance`이면 품질이 `float(n_steps)`로 대체되어 반대로
**긴 궤적이 높은 품질**이 된다. 두 모드에서 "대표로 살아남는 궤적"의 성격이
정반대라는 점을 인지해야 한다.

### 7.6 중복 제거 — `dedup_group`

```
품질 내림차순 정렬
├─ 그룹 크기 > --max-jaccard-group(3000)
│     → 정확 지문 일치만 제거 (O(n) 집합 조회), jaccard 표본 없음
└─ 그 외
      각 후보의 "이미 채택된 것들에 대한 최대 Jaccard"를 계산
      >= --near-dup-threshold(0.9) 이면 버림, 아니면 채택
```

버려질 후보의 최대 Jaccard도 표본에 기록되므로, 실행 끝의 히스토그램은
"임계값을 어디로 옮기면 몇 개가 더/덜 걸리는지"를 보여준다.

### 7.7 2패스 구조와 gid 정합

1패스에서 `gid`는 **읽은 모든 레코드**에 대해 증가하고(스텝 부족·언어 필터로 버려진
것 포함), 살아남은 것만 `metas`에 들어간다. 2패스에서 같은 파일을 같은 순서로
다시 읽으며 `gid`를 처음부터 다시 세어 `keep_gids` 소속 여부로 출력한다.
두 패스의 `iter_records`가 동일하므로(공백/파손 줄 동일 처리) 번호가 일치한다.

GPU flush는 `--score-chunk`(기본 64) 궤적마다 일어나고, 2,000궤적마다 진행 로그를 낸다.

### 7.8 리포트 출력

```
read trajectories / dropped (< N step) / [dropped (answer != lang)] /
near-duplicates removed / after dedup / selected (written) /
importance 모드 / near-dup threshold / [keep-frac|keep-k]
tasks (rollouts/task: max, median) + 히스토그램 버킷 1 / 2-5 / 6-20 / 21+
Jaccard 분포 10구간 막대 + 현재 임계값 위치 표시
```

`near-duplicates removed`는 뺄셈으로 구한다:
`n_read − n_dropped_steps − n_dropped_lang − n_after_dedup`.

실측 스모크(합성 12궤적 = 3태스크 × 4롤아웃, 태스크당 답변 2종):

```
read 12 → dropped(<1 step) 0 → near-dup removed 6 → after dedup 6 → written 6
Jaccard 표본 9개: 0.5-0.6 구간 3개, 0.9-1.0 구간 6개 (임계 0.9)
rollouts/task: max 4, median 4, 히스토그램 2-5=3
```

### 7.9 실측 이슈 — 도움말 문자열 잔존

```python
ap.add_argument("--near-dup-threshold", ...,
                help="Signature Jaccard >= this == near-duplicate ...")
```

용어를 `signature` → `fingerprint`로 통일한 이후에도 이 도움말만 옛 이름을 쓴다.
동작에는 영향이 없다.

---

## 8. `scripts/train_lora_sft.py` — 368 LOC

### 8.1 역할 정규화

```python
TRAINABLE_ROLES = {"assistant"}
ROLE_ALIASES = {"human": "user", "gpt": "assistant",
                "function_call": "assistant",   # ShareGPT: assistant tool call
                "observation": "tool"}          # ShareGPT: tool result
```

`normalize_record`는 `messages`가 없으면 `conversations`(`from`/`value`)에서 만들고,
별칭을 적용한 뒤 `{system,user,assistant,tool}` 밖의 역할이 하나라도 있으면
레코드 전체를 버린다(`None` 반환). `content`가 `None`이면 빈 문자열로 채운다.
학습 가능한 턴이 하나도 없으면 버린다. `tools`가 문자열이면 파싱하고,
dict면 리스트로 감싸고, falsy면 `None`으로 만든다.

이 정규화 덕분에 `weasel_agenttrek_train_10k.json`,
`convert_gemini`의 두 출력, 그리고 원본 `tools+messages` 내보내기가 모두 별도 변환 없이
학습된다.

### 8.2 `render(tok, msgs, tools, add_generation_prompt=False)`

`tok.apply_chat_template(..., tokenize=True, return_dict=False)`를 호출한다.
`return_dict=False`를 명시한 이유는 transformers 5부터 기본이 dict이기 때문이다.
예외가 나면 **빈 리스트**를 반환한다 — 엄격한 템플릿(예: Qwen3.5의
"No user query found in messages.")이 system 전용 접두를 거부하는 경우를
"이 접두는 렌더되지 않는다"로 취급하기 위한 처리다.

### 8.3 `build_example` — assistant 전용 손실 마스킹

```
full = render(전체);  비었으면 샘플 폐기
full = full[:cutoff_len]
train_on_all 이면 labels = full 복사, 종료

labels = [-100] * len(full)
prev = 0
for k in 1..len(msgs):
    ids = render(msgs[:k])
    n = min(len(ids), len(full))
    if ids[:n] != full[:n]:      # 접두 불안정 → 폴백
        fallback = True; break
    bound = min(len(ids), len(full))
    if msgs[k-1].role in TRAINABLE_ROLES:
        start = prev
        head = render(msgs[:k-1], add_generation_prompt=True)
        if len(head) > prev and head == full[:len(head)]:
            start = min(len(head), bound)     # assistant 헤더 토큰 제외
        labels[start:bound] = full[start:bound]
    prev = bound
    if bound >= len(full): break

fallback → (full, full 복사, True)
labels가 전부 -100 → None (학습 대상이 전부 잘려나감)
```

접두 검증이 필요한 이유: 일부 템플릿은 최종이 아닌 assistant 메시지에서 `<think>`를
제거하는 등 **이전 턴을 다시 쓴다**. 그러면 "메시지 k까지의 렌더는 전체 렌더의 접두"라는
가정이 깨져 마스크 위치가 어긋난다. 어긋나는 샘플은 전체 시퀀스 손실로 폴백하고
개수를 센다.

> 실측 관찰: 중간 접두 렌더가 예외로 `[]`를 반환하면 `bound = 0`, `prev = 0`이 된다.
> 바로 다음 메시지가 assistant이고 그 시점의 생성 프롬프트 렌더도 실패하면
> `start = 0`이 되어 **처음부터 그 턴 끝까지 전부** 라벨이 붙는다.
> 실제로 문제되는 배치는 `[system, assistant, ...]`처럼 실패 접두 직후에 assistant가
> 오는 경우뿐이다. 흔한 `[system, user, assistant, ...]`에서는 user 턴이
> `prev`를 정상 위치로 복구하므로 영향이 없다.

### 8.4 데이터 파이프라인

```python
ds = Dataset.from_list([{"payload": json.dumps(r, ensure_ascii=False)} for r in records])
ds = ds.map(encode, num_proc=max(1, args.num_workers), remove_columns=..., desc="tokenize+mask")
n_fallback = sum(ds["fallback"])
ds = ds.filter(lambda ex: len(ex["input_ids"]) > 0)
```

레코드를 JSON 문자열 컬럼으로 감싸는 이유는 Arrow의 struct 스키마 통합이
tool 파라미터 스키마를 합집합으로 만들어 프롬프트에 `null` 필드를 주입하거나
혼합 타입에서 크래시하기 때문이다(주석 명시). 필터 전에 폴백 수를 세고
백분율과 함께 경고를 출력한다.

`collate`는 배치 최대 길이로 우측 패딩하고 `input_ids`는 `pad_token_id`,
`labels`는 `-100`, `attention_mask`는 0으로 채운다.

### 8.5 모델 로드 옵션

**`--load-4bit`**:

```python
BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
device_map = {"": int(os.environ.get("LOCAL_RANK", "0"))}
```

양자화 가중치는 로드 후 이동할 수 없어 각 DDP 랭크를 로드 시점에 자기 GPU로 고정한다.
이후 `prepare_model_for_kbit_training`을 적용하고, peft가 fp32로 올린 파라미터 중
**`ndim == 2`인 것만** 계산 dtype으로 되돌린다(1차원 norm 벡터는 fp32 유지).
LoRA 파라미터는 이 처리 이후에 추가되므로 영향받지 않는다.

**`--liger`**: 패키지 없으면 `ModuleNotFoundError`만 잡아 종료(다른 예외는 통과시켜
원인을 숨기지 않는다). `MODEL_TYPE_TO_APPLY_LIGER_FN`에 모델 타입이 없으면 경고.
실제 적용은 `TrainingArguments(use_liger_kernel=True)`가 담당한다.

### 8.6 `TrainingArguments` 실측 값

| 항목 | 값 |
|---|---|
| `optim` | `adamw_torch` |
| `lr_scheduler_type` | `--scheduler` 기본 `cosine` |
| `warmup_ratio` | 0.1 |
| `save_strategy` | `epoch` (선택지 `epoch`/`steps`/`no`) |
| `save_only_model` | `not --save-state` → 기본 True |
| `report_to` | `"none"` |
| `seed` | 42 |
| `gradient_checkpointing` | 기본 True, `use_reentrant: False` |
| `ddp_timeout` | 180,000,000 |
| `ddp_find_unused_parameters` | False |
| `remove_unused_columns` | False |
| LoRA | `r=8, alpha=8, dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM"` |
| `--cutoff-len` | 8192 |
| `--num-workers` | 8 |
| `--logging-steps` | 10 |

`--resume`인데 `--save-state`가 없으면 "옵티마이저/스케줄러 상태가 없어 AdamW 모멘트와
LR 스케줄이 체크포인트 스텝에서 재시작된다"고 경고한다.

토크나이저 사전 검증 두 가지: `chat_template`가 없으면 즉시 종료,
`pad_token`이 없으면 `eos_token`으로 대체하고 그것도 없으면 종료.

학습 종료 후 어댑터를 `--output-dir` 루트에 저장하고, world process zero에서만
토크나이저를 함께 저장한다(`merge_lora.py`가 이 토크나이저를 우선 사용한다).

---

## 9. 나머지 `scripts/*.py`

### 9.1 `merge_lora.py` — 71 LOC

`resolve_adapter(path)`: 루트에 `adapter_config.json`이 있으면 그 경로, 없으면
`checkpoint-(\d+)` 디렉터리 중 **번호가 가장 큰 것**을 고른다. 후보가 없으면 종료.

병합: `AutoModelForCausalLM.from_pretrained(base, torch_dtype=bf16|fp16)` →
`PeftModel.from_pretrained` → `merge_and_unload()` →
`save_pretrained(safe_serialization=True, max_shard_size="5GB")`.

토크나이저는 어댑터 디렉터리에 `tokenizer_config.json`이 있으면 거기서, 없으면
베이스에서 읽어 출력에 저장한다.

### 9.2 `agentlab_eval.py` — 138 LOC

```python
BENCH_TO_BROWSERGYM = {
  "miniwob": "miniwob", "webarena": "webarena",
  "webarena_lite": "webarena",          # 필터로 165태스크만 남김
  "workarena_l1": "workarena_l1", "workarena_l2": "workarena_l2"}
```

`build_agent_args`: `OPENAI_API_KEY`(없으면 `"dummy"`), `OPENAI_BASE_URL`,
`OPENAI_API_BASE`를 `setdefault`로 채운 뒤 `GenericAgentArgs` + `FLAGS_GPT_4o` 조합을
만든다. 모델 인자 클래스는 `OpenAIModelArgs` 우선, 실패 시 `SelfHostedModelArgs`.
토큰 예산 `32,000 / 30,000 / 2,000`.

`_filter_webarena_lite(study)`: `$WEBARENA_LITE_TASKS` 또는
`configs/webarena_lite_tasks.txt`에서 줄당 하나씩 태스크 id를 읽는다.
`#` 이후는 주석, `.`이 없으면 `webarena.` 접두를 붙인다. 파일이 없거나 매칭이
0건이면 **종료**한다.

`--limit`은 `study.exp_args_list` 슬라이싱을 시도하고 실패 시 경고만 낸다.

### 9.3 `summarize_results.py` — 160 LOC

표준 라이브러리 전용. 어느 venv에서든 실행 가능.

```python
TASK_RE = re.compile(r"(?:miniwob|webarena|workarena)\.[.A-Za-z0-9-]+")
```

언더스코어를 제외한 이유가 주석에 있다: 디렉터리명 뒤에 붙는 `_<seed>_<hash>`가
태스크 id로 빨려 들어가지 않게 하기 위함이다. (`miniwob_report.py`의 같은 이름
정규식은 `[.\w-]+`라 언더스코어를 포함한다 — 두 파일의 태스크 이름 추출 결과가
디렉터리명 폴백 시 달라질 수 있다.)

`newest_study(root)`: `summary_info.json`을 하나라도 포함하는 하위 디렉터리 중
mtime 최댓값. 없으면 root 자체를 검사.

`task_name(exp_dir)`: `exp_args.pkl`의 `env_args.task_name` 우선, 실패하면 정규식,
그것도 실패하면 디렉터리명.

`collect`: `cum_reward`가 없거나 `err_msg`가 있으면 오류로 세고 제외.
성공 판정 `cum_reward > 0`.

출력: 콘솔 표(태스크명 45칸, n 5칸, 성공률 10칸, 평균보상 10칸) +
study 디렉터리에 `weasel_summary.json` + 선택적 CSV.

### 9.4 `miniwob_report.py` — 497 LOC

자체 완결형(인라인 CSS/JS) 다크 테마 HTML 리포트.

- `extract_episodes(study_dir)` → `(episodes, mode)`.
  `agentlab.experiments.loop.yield_all_exp_results` import에 성공하면 `"full"`,
  실패하면 `summary_info.json`만 읽는 `"summary"` 모드로 강등하고 배너를 표시한다.
- 스텝 필드: `action`, `type`(`^\s*([A-Za-z_]\w*)\s*\(` 로 함수명 추출),
  `think`(`agent_info`의 `think`/`thought`/`reasoning` 중 첫 비어있지 않은 값),
  `reward`, `action_error`(`obs["last_action_error"]`), `axtree_chars`.
- `compute_stats`: 태스크별 표(성공률 오름차순 = **최악 우선**), 액션 종류 빈도,
  실패 모드 4버킷, 그리고 `stats.*` 키 중 이름에 `token`이 들어가면 토큰 지표로,
  `time`/`elapsed`/`duration`이면 시간 지표로 자동 분류해 평균을 낸다.

실패 모드 분류 순서(먼저 맞는 것 채택):

```
err_msg 있음   → "agent/env error (err_msg)"
truncated      → "out of steps (truncated)"
terminated     → "finished, no reward (wrong/gave up)"
그 외          → "incomplete / unknown"
```

- 렌더링: 요약 카드(성공률 0.5 이상이면 녹색), 스텝 분포, 실패 모드 막대,
  액션 빈도 막대, 태스크별 표, 비용 지표, 그리고 에피소드 드릴다운.
  드릴다운은 **실패 먼저** 정렬하고 `--max-traj`(기본 400)개까지만 출력한다.
  `think`는 280자, `action_error`는 200자, `err_msg`는 500자로 자른다.
  상단 입력창의 인라인 JS가 `data-k` 속성(태스크명 + success/fail + 디렉터리명)으로
  필터링한다.

**실측 미사용 코드**: `import gzip`(23행)이 파일 어디에서도 쓰이지 않는다.
docstring이 "per-step 파일은 gzip된 pickle"이라 설명하지만 실제 압축 해제는
AgentLab이 수행한다.

### 9.5 `convert_dataset.py` — 112 LOC

임의 데이터셋 → `{"messages":[...]}` JSON 리스트.

- 경로 A: `messages`/`conversations`/`conversation` 중 하나가 있으면
  `normalise_messages`로 역할만 정규화(`human`→`user`, `gpt`→`assistant`).
  내용이 문자열이 아니면 `json.dumps`로 직렬화.
- 경로 B: `--user-field`와 `--assistant-field`가 모두 주어지면 평면 행에서 조립
  (`--system-field` 선택).
- `--goal-field`: 해당 필드 텍스트의 개행을 공백으로 바꾼 뒤 첫 user 턴 앞에
  `## Goal: <텍스트>\n\n`을 붙인다. WEASEL 그룹핑을 가능하게 하는 유일한 장치다.
- 결과가 0건이면 오류 메시지와 함께 종료.
- 입력이 `.parquet`이면 pandas 경유.

### 9.6 `inspect_dataset.py` — 91 LOC

행 수, 상위 2,000행 기준 최상위 필드 빈도/타입,
`messages`류 필드가 있으면 역할 분포, 그리고 `-n`(기본 2)개 샘플을 240자로 잘라 출력.
`.parquet`이면 pandas 필요(없으면 안내 후 종료).

### 9.7 `_resume_eval.py` — 52 LOC

`agentlab.experiments.study.Study.load(study_dir)`로 기존 study를 읽고 `run()`을
호출한다. `Study.run()`의 `find_incomplete()` + `n_relaunch` 로직이 완료된 태스크를
재사용하고 미완만 다시 돌린다. `agentlab_eval.py`가 항상 `make_study()`로 새 디렉터리를
만들기 때문에 필요한 보조 도구다.

### 9.8 `_vllm_qwen35.py` — 57 LOC

vLLM CLI 대체 진입점. 두 가지를 한다.

1. `CONFIG_MAPPING.register("qwen3_5_text", _Qwen35TextConfig, exist_ok=True)`
2. `Qwen3_5ForCausalLM`을 상속해 `load_weights`에서 가중치 이름의
   `model.language_model.` 접두를 `model.` 로 바꾸는 서브클래스를 만들고
   `ModelRegistry.register_model("Qwen3_5ForCausalLM", ...)`로 등록

배경: 특정 vLLM 버전이 `Qwen3_5ForCausalLM` 클래스는 있으나 멀티모달
`Qwen3_5ForConditionalGeneration`만 등록하며, 이 레포가 저장한 LoRA 병합 텍스트
체크포인트는 병합 시점에 멀티모달 config가 유효해 `language_model.` 세그먼트를 달고
저장된다. 마지막에 `sys.argv[0] = "vllm"`로 바꾸고 vLLM CLI `main()`을 호출하므로
사용법이 `vllm`과 동일하다.

SETUP.md는 이 shim보다 9.9의 오버레이 경로가 현재 실동작하는 방법이라고 적고 있다.

### 9.9 `_overlay_text_into_mm.py` — 93 LOC

멀티모달 베이스의 샤드 레이아웃을 그대로 유지하면서 텍스트 키만 LoRA 병합본으로
덮어쓴다. `model.safetensors.index.json`이 베이스와 동일하게 유지되는 것이 목적이다.

절차: 텍스트 체크포인트의 모든 샤드를 로드해 `{키: 텐서}` 조회표를 만들고,
베이스 샤드를 하나씩 열어 조회표에 있는 키만 교체한 뒤 같은 파일명으로 저장한다.
샤드마다 `overlaid N/total`을 출력한다.

이후 설정 파일 10종(`config.json`, 인덱스, 토크나이저 3종, `chat_template.jinja`,
`generation_config.json`, `merges.txt`, `vocab.json`, 전처리 설정 2종)을 베이스에서
복사하되, **`chat_template.jinja`만은 텍스트 체크포인트 것이 있으면 덮어쓴다**
(학습에 쓴 템플릿을 유지하기 위함).

**실측 이슈**: `--base` / `--text` / `--out`의 기본값이 특정 머신의 절대경로로
하드코딩되어 있다. 다른 환경에서는 세 인자를 반드시 명시해야 한다.

---

## 10. `scripts/*.sh`

### 10.1 `setup_env.sh` — 236 LOC

전체가 `${VAR:-기본값}` 패턴이라 사전 export로 모두 대체 가능하다.

주요 변수군:

| 군 | 변수 |
|---|---|
| 볼륨/루트 | `GROUP_VOLUME`, `USER_VOLUME`, `WEASEL_USER`, `WEASEL_REPO`, `WEASEL_WORK` |
| venv | `WEASEL_VENV_SELECT`, `WEASEL_VENV_TRAIN`, `WEASEL_VENV_EVAL` |
| 데이터 | `WEASEL_DATA`, `AGENTTREK_DIR`, `TRAIN_INPUT_JSON`, `GOALS_SCORES_JSON`, `SELECTED_INDICES_JSON`, `WEASEL_TRAIN_JSON`, `WEASEL_TRAIN_GDRIVE_ID` |
| 모델 | `MODELS_DIR`, `MODEL_QWEN25_7B`, `MODEL_GEMMA3_4B`, `MODEL_QWEN3_8B`, `MODEL_QWEN35_9B`, 대응 `HFID_*` 4종 |
| 신규 실험 | `NEWDATA_RAW`, `NEWDATA_FULL_JSON`, `NEWDATA_WEASEL_JSON`, `EXP_OUTPUT_ROOT` |
| 출력 | `OUTPUT_ROOT`, `MERGED_ROOT`, `EVAL_RESULTS_ROOT` |
| 캐시 | `HF_HOME`, `HF_DATASETS_CACHE`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `PLAYWRIGHT_BROWSERS_PATH` |
| 오프라인 | `HF_DATASETS_OFFLINE=1`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` |
| 평가 | `VLLM_HOST=127.0.0.1`, `VLLM_PORT=8000`, `VLLM_SERVED_NAME=weasel`, `OPENAI_API_BASE`, `OPENAI_API_KEY`, `MINIWOB_URL`, `SNOW_INSTANCE_URL/UNAME/PWD` |
| 런타임 | `TOKENIZERS_PARALLELISM=false`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `OMP_NUM_THREADS=16`, `MKL_NUM_THREADS=16` |

`WEASEL_REPO`는 `BASH_SOURCE`로 스크립트 위치를 역산한다.

Chromium 공유 라이브러리 보완: `WEASEL_XLIBS_DIR`이 비어 있고 `conda`가 있으면
`weasel-xlibs`라는 이름의 env를 찾아 `lib` 디렉터리를 잡고, 디렉터리가 실제로
존재할 때만 `LD_LIBRARY_PATH`에 앞쪽으로 붙인다. 라이브러리가 이미 있는 머신에서는
무해한 no-op이다.

토큰 처리: `~/.weasel_secrets.sh`가 있으면 source하고,
`HF_TOKEN` → `HUGGING_FACE_HUB_TOKEN` 순으로 전파한다.
"이 파일은 git 추적 대상이므로 토큰을 하드코딩하지 말라"는 경고가 주석에 있다.
마지막 출력에서 토큰은 값이 아니라 **길이(문자 수)**만 표시한다.

`weasel_activate {select|train|eval}`: venv 경로에 `bin/activate`가 없으면
"`bash scripts/install.sh <which>`로 만들라"는 안내와 함께 1을 반환한다.
bash일 때만 `export -f`한다.

`_weasel_warn`은 경고만 하고 절대 중단하지 않으며, 점검 대상은
`GROUP_VOLUME`, 3개 venv, `WEASEL_TRAIN_JSON`, `MODEL_QWEN25_7B` 총 6개다.
끝에 `unset -f _weasel_warn`으로 헬퍼를 정리한다.

### 10.2 `install.sh` — 107 LOC

`PYTHON_BIN`(기본 `python3.12`, 없으면 `python3`) 사용. 3.11~3.12 범위 밖이면 경고만.
자기 프로세스에서 오프라인 플래그를 0으로 뒤집는다.

| venv | 설치 내역 |
|---|---|
| select | `torch`(cu124 인덱스) → `bert-score==0.3.13` `tqdm` → `huggingface_hub[cli]` `datasets` `gdown` |
| train | `torch`(cu124) → `transformers` `peft` `datasets` `accelerate` `sentencepiece` `protobuf` → `bitsandbytes` `liger-kernel` → `huggingface_hub[cli]` |
| eval | `vllm` → `agentlab` `browsergym` → `browsergym-{webarena,miniwob,workarena}` → `playwright install chromium` → `playwright install-deps chromium`(실패 시 안내만) → miniwob-plusplus 클론 후 특정 커밋으로 `reset --hard` |

CUDA torch를 먼저 설치하는 이유가 주석에 있다: bert-score가 CPU 빌드를 끌어오지
않게 하기 위함이다. 각 venv 설치 후 import 확인 한 줄을 실행한다.

### 10.3 `download_models.sh` — 52 LOC

`_fetch(repo, dest)`: `dest/config.json`이 있으면 스킵.
`huggingface-cli download --local-dir --exclude "*.pth" "original/*"`를 먼저 시도하고,
실패하면 `hf download ... --exclude "*.pth" --exclude "original/*"`로 폴백한다
(hub 1.0의 Typer CLI는 `--exclude` 하나당 패턴 하나만 받기 때문).

`all` 순서: qwen25_7b → qwen3_8b → qwen35_9b → gemma3_4b.
게이트된 gemma를 마지막에 두어 앞의 셋은 로그인 없이도 받게 했다.

### 10.4 `download_data.sh` — 68 LOC

- `prebuilt`(기본): `WEASEL_TRAIN_JSON`이 이미 있고 비어있지 않으면 스킵.
  아니면 `python -m gdown <id> -O <경로>`(실패 시 `gdown` 명령)로 받고
  레코드 수를 출력한다. 파일 id는 `setup_env.sh`의 `WEASEL_TRAIN_GDRIVE_ID`에 있다.
- `agenttrek`: HF 데이터셋을 받아 `messages` 컬럼만 뽑아
  `[{"messages": [...]}]` 리스트로 변환한다. `messages` 컬럼이 없으면 컬럼 목록과 함께
  종료한다.

**실측 이슈**: `agenttrek` 경로 끝에서 "`weasel.prune_axtree`가 레포에 없어서
`run_select.sh`가 프루닝을 건너뛴다"고 안내한다. 사실이 아니다 —
`weasel/prune_axtree.py`가 존재하고 `run_select.sh`는 `PRUNE=1`(기본)로 이를 실행한다.

### 10.5 `run_select.sh` — 87 LOC

`weasel_activate select` 후 4단계 직렬 실행. 전 과정을 `logs/run_select.log`에 tee.

| 환경변수 | 기본 | 전달 대상 |
|---|---|---|
| `T0` | 3 | `select_greedy --t0-fixed` |
| `LAMBDA` | 1.0 | `select_greedy --lambda-weight` |
| `BATCH` | 64 | `prepare_scores --batch-size` (모듈 자체 기본은 32) |
| `PRUNE` | 1 | 0이면 단계 0 생략 |
| `WINDOW` | 60 | `prune_axtree --window-size` |
| `FALLBACK` | 120 | `prune_axtree --fallback-threshold` |
| `GPUS` / `--gpus` | 0 | 첫 GPU만 `CUDA_VISIBLE_DEVICES`로 사용 |
| `--input` | `$TRAIN_INPUT_JSON` | 입력 경로 |

프룬 산출물 이름은 `$WEASEL_DATA/<입력basename에서 확장자 제거>_pruned.json`이고
통계는 `<같은이름>_stats.json`이다. 단계 3은
`--max-user-chars 40000 --max-examples 10000 --seed 0`을 명시적으로 넘긴다.

주의: 단계 3의 `--dataset`은 `$SELECT_INPUT`(프룬본)이다. 선택 인덱스가 프룬본 기준이며
프루닝이 레코드 수/순서를 보존하므로 정합이 유지된다.

### 10.6 `convert_traindata.sh` — 53 LOC

`INPUTS`(기본 `train_data/*.jsonl` 전부), `MODE`(기본 `both`).
출력은 `$WEASEL_DATA/gemini_steps.jsonl`, `gemini_traj.jsonl`,
`gemini_convert_stats.json`. 끝에 다음 단계 명령을 echo로 안내한다
(실행하지 않는다).

### 10.7 `run_train.sh` — 144 LOC

모델별 레시피(실측):

```bash
model_spec() {   # <베이스 경로> <lr> <epochs> <global batch>
  qwen25)    "$MODEL_QWEN25_7B 2.0e-5 4.0 8"
  gemma3)    "$MODEL_GEMMA3_4B 2.0e-5 2.0 16"
  qwen3)     "$MODEL_QWEN3_8B  1.0e-6 2.0 8"
  qwen35_9b) "$MODEL_QWEN35_9B 1.0e-6 2.0 8"   # Qwen3-8B 레시피 상속
}
```

기본 `MODELS="qwen25 gemma3 qwen3"` — `qwen35_9b`는 명시해야 돈다.

전역 배치 고정 공식:

```bash
accum = gbatch / (ngpu * PER_DEVICE);  accum < 1 이면 1
```

`PER_DEVICE=1` 기준으로 qwen25는 8GPU에서 accum 1(전역 8), gemma3는 8GPU에서
accum 2(전역 16)다. GPU 수를 바꿔도 전역 배치가 유지된다.

플래그 조합:

```bash
QLORA=1 → EXTRA_FLAGS="--load-4bit --liger"
QLORA=0 && LIGER=1 → EXTRA_FLAGS="--liger"
```

`train_args()`가 조립하는 고정 인자: `--lora-r 8 --lora-alpha 8 --cutoff-len $CUTOFF`.

두 실행 모드:

- 기본(순차 DDP): 모델마다 `CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node $NPROC`
- `--parallel`: 모델을 GPU 목록에 라운드로빈 배정해 `nohup python ... &`로 동시 실행,
  마지막에 `wait`. 하나도 못 띄우면 그 사실을 stderr로 알린다.

베이스 모델 디렉터리가 없으면 해당 모델을 건너뛰고 `dl_key()`로 변환한
다운로드 명령을 안내한다(`qwen25`→`qwen25_7b`, `gemma3`→`gemma3_4b`,
`qwen3`→`qwen3_8b`, 그 외는 키 그대로).

`VARIANT`가 `weasel`이 아닌데 `DATA_FILE`이 없으면 오류로 종료한다.

### 10.8 `run_merge.sh` — 41 LOC

`MODELS`(기본 3종) × `VARIANT`(기본 weasel)에 대해
`$OUTPUT_ROOT/<model>/<variant>` → `$MERGED_ROOT/<model>/<variant>`.
어댑터 디렉터리가 없으면 스킵, 모르는 모델 키면 즉시 종료.
`weasel_activate train` venv에서 실행한다.

### 10.9 `serve_vllm.sh` — 42 LOC

`MODEL_KEY`는 첫 위치 인자(기본 `qwen25`). 기본값 `TP=1`, `MAXLEN=32768`,
`VARIANT=weasel`. `$MERGED_ROOT/$MODEL_KEY/$VARIANT`가 없으면 병합 명령을 안내하며 종료.

```bash
CUDA_VISIBLE_DEVICES="$GPUS" vllm serve "$MODEL_PATH" \
  --served-model-name "$VLLM_SERVED_NAME" --host "$VLLM_HOST" --port "$VLLM_PORT" \
  --tensor-parallel-size "$TP" --max-model-len "$MAXLEN" --trust-remote-code
```

### 10.10 `run_eval.sh` — 88 LOC

`RESULTS_DIR="$EVAL_RESULTS_ROOT/$VARIANT"`를 `AGENTLAB_EXP_ROOT`로 export하고,
`OPENAI_API_BASE`/`OPENAI_BASE_URL`을 서빙 엔드포인트로 맞춘다.

벤치마크별 전제조건:

| bench | 검사 |
|---|---|
| `miniwob` | `MINIWOB_URL`에서 `file://`를 뗀 경로 존재 확인 — **경고만** |
| `webarena` / `webarena_lite` | `WA_SHOPPING` 미설정이면 `:?`로 **즉시 종료**, `OPENAI_API_KEY` 없으면 경고 |
| `workarena_l1` / `workarena_l2` | `SNOW_INSTANCE_URL` 미설정이면 즉시 종료, `workarena-install` 실행(실패 시 경고) |
| 그 외 | 종료 코드 2 |

이후 `agentlab_eval.py` → `summarize_results.py` → `miniwob_report.py` 순으로 실행하며,
뒤의 두 개는 실패해도 평가를 실패시키지 않는다(`|| echo`).
산출 파일명: `logs/eval_<variant>_<bench>.log`,
`..._summary.csv`, `..._report.html`.

### 10.11 `run_experiment.sh` — 116 LOC

`--exp full|weasel` 하나에 대해 선택 → 학습 → 병합 → 서빙 → 평가를 한 번에 돈다.
대상 모델은 `MODEL_QWEN35_9B` 고정, 전역 배치는 8 하드코딩.

- `weasel`이면 먼저 `NEWDATA_FULL_JSON`에 `## Goal:`이 있는지 `grep -q`로 확인하고
  없으면 "그룹핑이 무너진다"고 경고한 뒤, `GOALS_SCORES_JSON` /
  `SELECTED_INDICES_JSON` / `WEASEL_TRAIN_JSON`을 실험 전용 경로로 덮어쓴 채
  `run_select.sh`를 호출한다. `run_select.sh`의 가드가 "변수가 이미 있으면
  재-source하지 않음"이라 이 덮어쓰기가 살아남는다.
- 학습은 `run_train.sh`를 거치지 않고 `torchrun`으로 트레이너를 직접 부른다.
  `LR`(기본 1e-6), `EPOCHS`(기본 2), `QLORA`/`LIGER` 환경변수를 읽는다.
- 서빙은 `bash -c "... exec vllm serve ..."`로 띄운다. `exec`를 쓰는 이유가 주석에
  있다 — `SERVE_PID`가 래퍼 bash가 아니라 vllm 프로세스 자신이 되어야
  `trap ... EXIT`가 실제로 GPU/포트를 해제한다.
- 준비 대기: 10초 간격 최대 120회(= 20분) 동안 `curl -fsS $OPENAI_API_BASE/models`를
  폴링하고, 중간에 서버 프로세스가 죽으면 즉시 오류로 종료한다.
- 평가는 `EVAL_RESULTS_ROOT`를 실험 디렉터리로 바꿔 `run_eval.sh --variant $EXP`를 호출.
  최종 결과는 `$EXP_OUTPUT_ROOT/<exp>/qwen35_9b/eval/<exp>`에 놓인다.

### 10.12 `setup_webarena.sh` — 71 LOC

`env`(기본): `$WEASEL_WORK/webarena_env.sh`에 `WA_*` 7종을 쓴다.
포트는 공식 기본값 shopping 7770, shopping_admin 7780, forum 9999, gitlab 8023,
wikipedia 8888, map 3000, homepage 4399.
`up`: docker 유무 확인 후 4개 컨테이너를 `docker run -d`로 띄우고(각각 `|| true`)
환경파일도 쓴다. 이미지 tarball 로드는 사용자 몫이라고 명시한다.

---

## 11. 의존 관계

### 11.1 모듈 간 import

`weasel/` 모듈끼리 서로를 import하지 않는다. 8개 모두 독립 실행 파일이고,
결합은 오직 **파일 포맷 계약**으로만 이뤄진다.

```
prune_axtree      : (없음)
prepare_scores    : bert_score, torch, tqdm(선택)
select_greedy     : (표준 라이브러리)
postprocess_dataset: (표준 라이브러리)
convert_gemini    : (표준 라이브러리)
select_trajectories: (표준 라이브러리)
select_clean      : bert_score(선택), torch(선택)
```

`scripts/*.py`도 마찬가지로 서로를 import하지 않는다.
`train_lora_sft.py`는 `main()` 안에서 torch/datasets/peft/transformers를
**지연 import**한다 — `--help`가 무거운 패키지 없이 뜨게 하려는 구조다.

### 11.2 데이터 포맷 의존(중요도 순)

| 생산자 | 산출물 | 소비자 | 깨지면 생기는 일 |
|---|---|---|---|
| `prepare_scores` | goal 레코드의 `trajectory_groups[].distance_matrix` | `select_greedy` | 키 없으면 `ValueError` |
| `prepare_scores` | `bert_scores_obs_history_norm` 등 별칭 키 | `select_greedy --score-key` | 없는 키면 전부 `0.0`으로 **조용히** 폴백 |
| `select_greedy` | 인덱스(중첩 또는 평탄) | `postprocess_dataset` | 범위 밖이면 `IndexError` |
| `prune_axtree` | 레코드 수/순서 보존 | `postprocess_dataset` | 순서가 바뀌면 잘못된 예제 선택 |
| `convert_gemini` | `_traj_id` 러닝 번호 | `select_trajectories` | 파일 순서가 다르면 잘못된 궤적 선택 |
| `convert_gemini` | `## Goal:` 등 마커 | `prepare_scores` | 마커 없으면 `<NO_GOAL_FOUND>` 한 그룹으로 붕괴 |
| 학습 파일 | `messages`(+`tools`) | `train_lora_sft` | 정규화 실패 레코드는 조용히 폐기, 남은 수를 출력 |
| AgentLab | `summary_info.json` | `summarize_results`, `miniwob_report` | 없으면 study 미탐지 |

**조용한 실패 두 곳**을 특히 주의해야 한다: `segment_importance`의 `0.0` 폴백과
`extract_goal`의 `<NO_GOAL_FOUND>` 그룹 붕괴다. 둘 다 예외 없이 진행되며
결과 품질만 떨어진다.

### 11.3 외부 서비스/자산 의존

| 대상 | 필요한 곳 |
|---|---|
| Hugging Face Hub | 베이스 모델 4종, AgentTrek, BERTScore `roberta-large`(암묵) |
| Google Drive | 사전 선별 10K 학습 파일 |
| GitHub | `miniwob-plusplus`(커밋 고정 클론) |
| OpenAI API | WebArena 계열 성공 판정기(`OPENAI_API_KEY`) |
| ServiceNow 개발 인스턴스 | WorkArena L1/L2 |
| Docker 또는 AWS AMI | WebArena 사이트 |

---

## 12. 미사용·미연결 코드 총괄(실측)

| 항목 | 상태 |
|---|---|
| `import gzip` (`miniwob_report.py:23`) | 파일 내 미사용 |
| `--lang` (`prepare_scores.py`) | 파싱만, `BERTScorer`에 미전달 |
| `fallback_threshold_count` (`prune_axtree.py`) | 증가하는 코드 없음, 항상 0 출력 |
| `--include-prompts`, `--skip-response-distance`, `--phi-field`/`--state-field`/`--response-field`/`--model-type`/`--device` | CLI에만 존재, 어떤 셸 스크립트도 사용 안 함 |
| `select_greedy --flat`, `--annotated-output`, `--t0-mode percentage`, `--score-key` | 동상 |
| `postprocess_dataset --dedupe`, `--preserve-order`, `--stats-output` | 동상 |
| `weasel/select_clean.py` | 셸 스크립트 미연결(수동 전용) |
| `weasel/select_trajectories.py` | 셸 스크립트 미연결(안내 echo만 존재) |
| `scripts/convert_dataset.py`, `inspect_dataset.py` | 셸 스크립트 미연결, EXPERIMENTS.md에서만 안내 |
| `scripts/_resume_eval.py`, `_vllm_qwen35.py`, `_overlay_text_into_mm.py` | 셸 스크립트 미연결(수동 보조 도구) |
| `configs/webarena_lite_tasks.txt` | 코드가 참조하나 레포에 없음 |
| `train_data/` | `convert_traindata.sh`가 참조하나 레포에 없음 |
| 테스트 | 레포 전체에 없음 |
