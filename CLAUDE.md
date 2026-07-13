# norm-boundary — 행동 불변식

연구 마스터 플랜은 `RESEARCH_PLAN.md` (Phase G0–G6). 현재 진행 상태와 다음 단계는 `HANDOFF.md` 참조.

## 환경 규약

- 의존성은 **uv**로만 관리 (`uv sync`, `uv run ...`). Python ≥ 3.11, PyTorch ≥ 2.7 (cu128, Blackwell sm_120).
- **GPU1 고정**: 학습·평가는 항상 `CUDA_VISIBLE_DEVICES=1` (GPU0은 Xorg 예약). `src/train.py`가 기본값으로 강제하지만, 별도 스크립트를 만들면 반드시 동일 규칙 적용.
- 재현성: 시드 고정 + `torch.use_deterministic_algorithms(True, warn_only=True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## MLflow 규약

- Tracking: 기본 `sqlite:///mlflow.db` (`MLFLOW_TRACKING_URI`로 재정의 가능; MLflow 3.x는 파일 스토어를 차단함)
- Experiment: `ini/norm-boundary`
- Run 이름: `{dataset}_{norm}_{backbone}_{h}_{seed}`
- 모든 run에 config 전체를 param으로, git commit 해시를 tag로 기록 (train.py가 자동 처리)

## 데이터 계약 (curated/ Parquet)

- 시간 인덱스: `date` (DatetimeIndex), 단조 증가, 결측 시각 없음 (시간 단위 고정 해상도)
- 값: float, NaN 금지 — 로더(`src/data/`)가 계약을 assert로 검증
- 분할: 시간순 out-of-time. ETT 계열은 문헌 표준 border(12/4/4개월), 전역 z-score는 **train 구간 통계로만** 적합
- DataLoader는 항상 `drop_last=False` (TFB 지적 사항)
- NWP 공변량(G3+): 반드시 리드 매칭된 아카이브 예보 사용, 관측 재분석 사용 금지 (누수)

## 코드 구조 규약

- 정규화는 `src/norms/` 모듈로 분리, 인터페이스는 `forward(x, mode)` with `mode ∈ {"norm", "denorm"}` — 백본과 독립적으로 조합 (`NormWrapper`)
- 새 정규화/백본/데이터셋은 각 `__init__.py`의 REGISTRY에 등록
- 모든 정규화는 **가역성 pytest** 필수 (`denorm(norm(x)) ≈ x`)
- 학습 진입점은 `src/train.py` 하나만: config yaml → run → MLflow

## 이론–실험 정합 규약 (2026-07-13 사용자 지시)

- 이론 검증 실험(G1·G2)의 백본은 **이론의 함수 클래스와 정확히 일치**시킨다: 선형-가우시안 이론 ↔ 단층 선형(RLinear). 은닉층·활성함수 등 용량 추가 금지.
- 정규화 비교에서 백본 용량(깊이·넓이·lookback)은 arm 간 완전 동일 — 달라지는 변수는 정규화 하나뿐.
- 용량 선택은 config yaml에 명시적으로 기록. 용량 민감도는 통제된 ablation으로만.
- 이론과 실험 사이의 매핑은 `docs/theory_g1.md`의 대응표가 기준. 대응이 어긋나면 그 표의 "괴리" 항목에 기록하고 진행 (완벽 대응에 매몰 금지).

## 검증 명령

```bash
uv run pytest            # 전체 테스트 (가역성·누수·스모크)
uv run python -m src.train --config configs/smoke_etth1.yaml   # 스모크 학습
```

## Phase 종료 공통 규칙 (RESEARCH_PLAN.md §10)

1. 해당 Phase의 AC 체크리스트 자체 평가
2. `results/` 요약 갱신
3. `HANDOFF.md`에 다음 Phase 착수 정보 기록
