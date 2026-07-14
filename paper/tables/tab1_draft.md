# Tab 1 초안 — dataset × normalization (backbone·h·seed 평균, std-scale MSE)

| dataset | raw | revin | san | fan | condnorm |
|---|---|---|---|---|---|
| jeju_wind | 0.6887* | 0.6957* | 0.7030* | 0.7724* | **0.2933** |
| gefcom_wind | 1.1153* | 1.0011* | 0.9755* | 1.1875* | **0.1629** |
| gefcom_load | 0.3592* | 0.3429* | 0.3330* | 0.3678* | **0.1066** |
| gefcom_solar | 0.1828* | 0.1850* | 0.2116* | 0.1833* | **0.0580** |
| etth1 | 0.3962* | **0.3773** | 0.3910* | 0.4114* | 2.2016* |
| etth2 | 0.3868* | 14425.9848* | **0.2869** | 0.3253* | 6.6044* |
| electricity | **0.1523** | 5735.0117 | — | — | 0.1777* |
| weather | 0.1799* | 0.1722* | **0.1666** | 0.1747* | 0.7254* |

`*` = DM 검정(Harvey 보정, (backbone,h) 셀별 시드 평균 손실, Fisher 결합)에서 해당 데이터셋 최적 arm보다 유의하게 나쁨 (p<0.05). **굵게** = 최적 arm.

## MCS (α=0.10) — (dataset, h)별 정규화 arm 잔존 집합

- jeju_wind h=24: {condnorm}
- jeju_wind h=48: {condnorm}
- gefcom_wind h=24: {condnorm}
- gefcom_wind h=96: {condnorm}
- gefcom_wind h=336: {condnorm}
- gefcom_load h=24: {condnorm}
- gefcom_load h=96: {condnorm}
- gefcom_load h=336: {condnorm}
- gefcom_solar h=24: {condnorm}
- gefcom_solar h=96: {condnorm}
- gefcom_solar h=336: {condnorm}
- etth1 h=24: {revin}
- etth1 h=96: {revin}
- etth1 h=336: {revin}
- etth2 h=24: {revin, san}
- etth2 h=96: {san}
- etth2 h=336: {san}
- electricity h=24: {raw}
- electricity h=96: {raw}
- electricity h=336: {raw}
- weather h=24: {san}
- weather h=96: {san}
- weather h=336: {condnorm}