# Tab B 초안 — 정보-대등 ablation (전 arm 동일 공변량, 외생 4종)

arm 평균 MSE (dataset·h·seed 평균):

| backbone    |    raw |   revin |      san |      fan |   condnorm |
|:------------|-------:|--------:|---------:|---------:|-----------:|
| lgbmcov     | 0.127  |  0.146  | nan      | nan      |     0.1361 |
| linmix      | 0.2762 |  0.2967 |   0.3191 |   0.4332 |     0.1496 |
| mlpmix      | 0.1978 |  0.2214 |   0.2419 |   0.4081 |     0.1446 |
| patchtstcov | 0.2851 |  0.2889 |   0.3378 |   0.3729 |     0.1434 |
| segrnncov   | 0.1312 |  0.1425 |   0.1822 |   0.2912 |     0.1352 |

DM (revin+cov vs condnorm+cov, (dataset,h) 셀별 시드평균 손실):

| backbone | CN우세 유의 셀 | 전체 셀 |
|---|---|---|
| lgbmcov | 3 | 11 |
| linmix | 10 | 11 |
| mlpmix | 10 | 11 |
| patchtstcov | 11 | 11 |
| segrnncov | 2 | 11 |

주: 블록 간 직접 비교 금지 원칙에 따라 이 표는 covfair 내부 비교만 담는다 (docs/design_audit.md §4).