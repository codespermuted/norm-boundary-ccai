# 사전 등록 증거 체인 (G7 §1, 2026-07-20 작성)

세 개의 독립 타임스탬프 계열(git · MLflow · GitHub)로 "예측 등록 → 실험 실행" 순서를 검증한다.
모든 시각은 KST(+09:00), GitHub API 시각은 UTC 원문을 병기.

## 1. 핵심 순서 (주 증거)

| 사건 | 시각 (KST) | 출처 |
|---|---|---|
| **사전 등록 커밋** `cab17c1` (paper/predictions.md + lps_official.csv + 산출 스크립트) | **2026-07-13 13:20:23** | git (author == committer, 재작성 후에도 보존) |
| **최초 Block-A grid run** `jeju_wind_revin_rlinear_24_0` | **2026-07-13 13:24:36** | MLflow start_time + 동일 시각 `results/g4_errors/*.npy` mtime |
| 간격 | **+4분 13초** | |

최초 60개 grid run의 MLflow `mlflow.source.git.commit` 태그 = `279bb80` — **사전 등록 커밋 자체의
체크아웃에서 실행**되었음을 뜻한다 (아래 해시 대응표).

## 2. 해시 대응표 (history 재작성 전↔후)

2026-07-15 11:15–11:21의 filter-branch(.env 추적 제거)로 커밋 해시가 재작성되었다.
날짜(author·committer 모두)는 전 커밋에서 보존 확인 (감사 2026-07-20, `git log --all` 전수).
대응은 커밋 `228bdcc`(11:21:05)에 기록되어 있다.

| 재작성 전 (MLflow 태그) | 재작성 후 (현 저장소) | run 수 | 최초 run 시각 |
|---|---|---|---|
| `05db7c7` (G2 합성) | — | 2,640 | 07-13 09:34:48 |
| `0ce99db` (G3 sanity; **양쪽 히스토리에 동일 해시로 존재 — 앵커**) | `0ce99db` | 4 | 07-13 10:31:55 |
| `279bb80` **(사전 등록)** | **`cab17c1`** | 60 | **07-13 13:24:36** |
| `7d809f0` | (후속 튜닝 커밋) | 885 | 07-13 13:28:32 |
| `f51a4cc` / `65a56f9` / `010f842` / `af61aab` | 〃 | 219/417/40/47 | 07-13 16:20 ~ 07-14 15:38 |

앵커 `0ce99db`가 재작성 전 MLflow 태그와 현 히스토리에 **같은 해시**로 존재하므로,
재작성이 사전 등록 이전 구간의 시간 축을 건드리지 않았음이 교차 확인된다.

## 3. GitHub 원격 증거 (보조 증거 — 한계 명시)

GitHub Events API (`repos/<ORG>/<REPO>/events`) 추출:

| 사건 | 시각 (UTC / KST) |
|---|---|
| 원격 저장소 생성 | 2026-07-15 02:21:05Z / 11:21:05 |
| **최초 push** (head `ab0d581`, cab17c1을 조상으로 포함) | **2026-07-15 02:42:25Z / 11:42:25** |
| 이후 push 17건 | 07-15 ~ 07-20 |

**한계의 정직한 기술**: 원격 저장소는 grid 시작 **이틀 후**에 개설되었으므로, push 타임스탬프
단독으로는 "등록 → 실행" 순서를 증명하지 못한다. push 증거가 증명하는 것은:
(a) 2026-07-15 11:42 시점에 `cab17c1`의 내용(예측 8건·τ=0.3·grid 설계)이 외부 서버에 고정되었고,
(b) 이는 확장 블록(B-deep/C/D) 완주(07-16), G5 분석(07-16~20), 원고 작성(07-20) **이전**이며,
(c) push된 내용이 분석에 사용된 값과 바이트 단위 일치한다는 것이다.
**순서의 주 증거는 §1의 git 로컬 타임스탬프 + MLflow 태그·시각 + 파일 mtime 삼중 일치**다.

## 4. 검증 재현 명령

```bash
git log --format='%h %aI %cI' cab17c1 -1        # author==committer 2026-07-13T13:20:23+09:00
git show cab17c1:paper/predictions.md | head    # 등록 당시 예측 내용
git show 228bdcc                                 # 해시 대응 기록
uv run python - <<'PY'                           # MLflow 최초 run
import sqlite3; con=sqlite3.connect('mlflow.db')
print(list(con.execute("""SELECT t.value, datetime(MIN(r.start_time)/1000,'unixepoch','+9 hours')
 FROM runs r JOIN tags t ON r.run_uuid=t.run_uuid
 WHERE t.key='mlflow.source.git.commit' GROUP BY 1 ORDER BY 2 LIMIT 4""")))
PY
gh api repos/<ORG>/<REPO>/events --paginate  # PushEvent 시각
ls -l --time-style=full-iso results/g4_errors/jeju_wind_revin_rlinear_24_0.npy
```

## 5. 판정

- 등록(13:20) → 실행(13:24) 순서: **git·MLflow·mtime 3계열 일치로 입증** ✅
- 재작성의 무해성: 날짜 보존 + 앵커 커밋 + 대응 기록으로 입증 ✅
- 원격 push: 보조 증거로서 07-15 이후 확장 블록·분석·원고에 대한 선행성 입증 ✅ (grid 자체에 대해서는 비적용 — 한계 명시)
