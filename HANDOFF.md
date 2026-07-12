# HANDOFF — 현재 상태와 다음 단계

- **최종 갱신**: 2026-07-13
- **완료 Phase**: G0 ✅
- **다음 Phase**: **G1 — 이론 수치화** (RESEARCH_PLAN.md §10, 부록 A의 [G1] 프롬프트로 착수)

## G0 AC 자체 평가

| AC | 판정 | 근거 |
|---|---|---|
| `pytest` 통과 | ✅ | 11 passed (가역성 4 + 데이터 계약·누수 4 + 스모크 2 + RAW identity 1), 1.8s |
| MLflow에 스모크 run 기록 | ✅ | experiment `ini/norm-boundary`, run `etth1_revin_rlinear_96_0`, FINISHED — test_mse **0.4067**, test_mae **0.4193** (ETTh1, L=336, h=96, RLinear+RevIN, GPU1) |
| the environment contract 생성 | ✅ | 환경 규약·MLflow 규약·데이터 계약·코드 규약 수록 |
| (추가) 신규성 스윕 | ✅ | `docs/novelty_sweep.md` — **킬 크라이테리아 비발동**. 단 RevIN 재검토 문헌(arXiv:2603.11869, 2510.04667)이 최근 급증 — 선점 리스크 시간 민감 |

## 구현된 것

- 레포 구조 §9 그대로 (theory/synth/eval은 빈 디렉터리 — G1·G2에서 채움)
- `src/norms/`: `NoNorm`(RAW), `RevIN` (affine, 가역성 테스트 완료) — `forward(x, mode)` 인터페이스, REGISTRY 등록 방식
- `src/models/`: `RLinear`(CI 공유 선형, `individual` 옵션), `NormWrapper`(norm→backbone→denorm 조합)
- `src/data/etth.py`: ETTh1/h2 로더 — 자동 다운로드→Parquet 캐시(`curated/`), 계약 assert, 문헌 표준 border(12/4/4개월), train 통계 z-score, `drop_last=False`
- `src/train.py`: 단일 진입점 (config yaml → 학습 → MLflow). GPU1·결정론 env 기본값 강제
- 스모크 결과 요약: `results/summary.md`

## 환경 확정치 (계획서와의 차이 포함)

- torch **2.11.0+cu128**, GPU: RTX 5070 Ti ×2 (GPU1 사용, CUDA 정상)
- **MLflow는 sqlite 백엔드** (`sqlite:///mlflow.db`) — MLflow 3.x가 file store를 차단하여 계획서의 `./mlruns` 대신 사용. `MLFLOW_TRACKING_URI`로 재정의 가능
- ETTh1 스모크(비튜닝, lr 0.005 고정)는 test MSE 0.4067 — 문헌 범위(0.37~0.41) 상단. lookback·lr 튜닝은 G3 AC에서 수행

## G1 착수 정보

1. `src/theory/closed_form.py`: 선형-가우시안 설정에서 RAW/IN/CN의 닫힌형 MSE (명제 1–3, RESEARCH_PLAN.md §2)
2. 몬테카를로 대조 상대오차 < 1%를 pytest로 고정
3. λ*(w, h, σ) 임계 함수 + Fig 1(지배 영역) → `paper/figures/`
4. 유의: 명제 2의 도출 형태는 §2.3의 부등식 — IN vs RAW는 (수준 드리프트 분산) vs (윈도우 통계 노이즈 σ_z²/w + 외삽 편향), CN vs IN은 λ > λ* 형태로 정리할 것
5. matplotlib은 이미 의존성에 포함됨. 수치 검증 스크립트는 GPU 불필요(numpy로 충분)하나, MC 시뮬레이션이 크면 torch+GPU1 사용 가능
