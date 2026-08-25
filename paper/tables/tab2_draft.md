# Tab 2 초안 — 분산 귀속 + MCS 포함 요약

## (a) log-MSE 분산 귀속 (Block A)

| factor                         |   share_% |
|:-------------------------------|----------:|
| norm                           |      0.73 |
| backbone                       |      0.13 |
| dataset                        |     42.45 |
| h                              |      5.65 |
| normxdataset (interaction)     |     47.12 |
| normxbackbone (interaction)    |      0.31 |
| backbonexdataset (interaction) |      0.44 |

핵심: 정규화-관련 분산(주효과+norm×dataset+norm×backbone = 48.2%) vs 백본-관련(주효과+backbone×dataset = 0.6%). 정규화 주효과가 작은 이유는 효과의 '부호'가 데이터셋에 따라 뒤집히기 때문 — norm×dataset 상호작용(47%)이 바로 본 논문의 연구 대상(적용 경계)이다.

## (b) MCS(α=0.10) 잔존 횟수 (23 (dataset,h) 셀)

| arm | 잔존 셀 수 |
|---|---|
| raw | 0/23 |
| revin | 6/23 |
| san | 6/23 |
| fan | 3/23 |
| condnorm | 11/23 |