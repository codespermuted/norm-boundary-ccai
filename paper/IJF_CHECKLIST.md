# IJF 투고 체크리스트 (International Journal of Forecasting, Elsevier)

작성: 2026-07-20 (G6). 항목은 IJF Guide for Authors 기준.

## 원고 형식

- [x] elsarticle 문서 클래스 (`preprint,12pt`), author–year 인용 (`elsarticle-harv`)
- [x] 제목·초록(150–220단어)·키워드(≤6, `\sep` 구분)
- [x] Highlights 별도 파일 (`sections/highlights.tex`, 각 항목 ≤85자, 3–5개)
- [ ] 저자·소속·교신저자 이메일 기입 (현재 익명 placeholder — 투고 직전 기입)
- [ ] 표지 서한(cover letter) 작성
- [x] 본문 ~8,000 단어 목표 (현재 본문 섹션 합계 ~8.3k 단어, 부록 제외)
- [x] 그림 5장 (Fig 1–5) + 부록 질적 그림 1장, 표 3개 (Tab 1–3) + 부록 표 3개 — 계획서 §6 산출물 고정과 일치
- [x] 정리 환경: Proposition/Assumption/Remark (amsthm), 증명은 전부 부록

## 연구 무결성 · 재현성 (IJF replicability policy)

- [x] 사전 등록 문서와 커밋 증빙 (paper/predictions.md, commit cab17c1 — 본문 §4·부록 재현성 성명에 기재)
- [x] 재현성 성명 부록 (환경, 시드, 결정론 설정, MLflow 추적, `make figures`/`make tables`)
- [x] 데이터 출처: GEFCom2014(공개), 표준 LTSF(공개), 제주 풍력(data.go.kr) + KMA API 허브 아카이브 예보 — 수집 코드 저장소 포함
- [x] 코드·데이터 공개 저장소 링크 — 2026-07-22 공개 완료 (git 이력 포함, 사전등록 증빙 목적 상시 공개; 논문 §1 각주·재현성 부록에 URL 기재)
- [x] 통계 검정 명세: DM(Harvey 보정, HAC 대역폭 상한), MCS(α=0.10, stationary bootstrap), Fisher 결합
- [x] 이해상충·사사 문구 위치 확보 (투고 시 기입)

## 내용 방어선 (리뷰 대비, RESEARCH_PLAN §7 매핑)

- [x] 방어 1 "RevIN 만능 주장한 적 없다" → §1·§2: 원논문 동기 존중 + 경계 특성화 포지셔닝
- [x] 방어 2 "적응형 정규화가 이미 해결" → §6: SAN·FAN 포함, 외생 그룹 전패 실증
- [x] 방어 3 "피처 엔지니어링일 뿐" → §3 명제 1 + §6 블록 B 정보 대등 (RAW+cov > RevIN+cov)
- [x] 방어 4 "에너지 편중 평가" → §6: GEFCom + 표준 LTSF 문헌 수치 재현 병행

## 남은 TODO (본문 마커 ≤20 규정: 렌더링 0개)

1. `main.tex` frontmatter — 저자·소속 (투고 메타데이터, 주석으로만 표시)
2. (조판 소사항) arXiv 전용 참고문헌 뒤 여분 마침표 — elsarticle-harv .bst의 빈 volume/pages 필드 artifact. 투고 시 .bst 후처리 또는 무시 (의미 영향 없음)

## 빌드

```bash
cd paper && tectonic main.tex   # 또는 저장소 루트에서: make paper
```
