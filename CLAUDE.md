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

## 검증 명령

```bash
uv run pytest            # 전체 테스트 (가역성·누수·스모크)
uv run python -m src.train --config configs/smoke_etth1.yaml   # 스모크 학습
```

## Phase 종료 공통 규칙 (RESEARCH_PLAN.md §10)

1. 해당 Phase의 AC 체크리스트 자체 평가
2. `results/` 요약 갱신
3. `HANDOFF.md`에 다음 Phase 착수 정보 기록
