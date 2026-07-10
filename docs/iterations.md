# Iteration Journal — brainstorm→modify→train→analyze loop
(Goal: lower official FREUID Score = 1−HM(1−AuDET, 1−APCER@1%BPCER) with
contribution-grade methods, not mere tuning. Every iteration must state WHY the
previous one failed before proposing the next.)

Validation protocol (fixed): leave-one-doc-type-out (MOZAMBIQUE/DL, 13,365 imgs),
seed 42, FREUID Score on holdout. Public LB used only as sanity (5.5% slice, saturated).

| # | method | FREUID | verdict |
|---|---|---|---|
| 1 | ConvNeXt-T baseline (full) | 0.4030 | reference |
| 2 | + global-FFT dual stream | 0.9765 | **FAIL** |
| 3 | + heavy recapture aug / HPF residual (partial) | 0.4065 / 0.4080 | no gain |
| 4 | DTC: TraceMix + consistency head (partial) | 0.4059 (λ=0) 0.4087 | marginal gain |
| 5 | DINOv2 fully frozen + adapter (full) | 0.5086→0.9031 | **FAIL** |
| 6a | DTC full data (ConvNeXt) | 0.3996 | marginal (-0.003 vs baseline) |
| 6b | **DINOv2 last-2-blocks unfrozen + 378px** | **0.0142 @ep~5 (ep15 진행 중)** | **BREAKTHROUGH** |
| 6b' | 6b public LB 제출 (ep~5 best) | **public 0.2869** (v1 0.3574) | holdout↔public 20× 격차 발견 |
| 7 | DTC × DINOv2-PE host (full, MOZ holdout) | **0.0188** best; public 0.2826 | ≈#6b peak + **5× late-epoch stability** |
| 6b-V | #6b recipe, EGYPT holdout | **0.2107** best (ep1), 즉시 붕괴 | MOZ 0.014는 낙관적 — 유형별 편차 큼 |
| 9 | #7 + FantasyID joint (MOZ holdout) | 0.0345 | MOZ 폴드엔 무익 (#7 0.0188 대비 ↓) |
| 7-V | #7 recipe, EGYPT holdout | **0.2115** | DTC 정칙화도 EGYPT 벽 못 깸 (≈#6b-V 0.2107) |
| 10 | SBD + #7, EGYPT fold | 0.2307 (ep0)→0.98 붕괴 | **FAIL** — blend-artifact 지름길 학습 |
| 9-V | FantasyID joint, EGYPT fold | 0.2307 (ep0) | FAIL — 3.3k는 53k 대비 무력 |
| 11 | patch 학습+5-crop 평가, EGYPT fold | 0.2358 | FAIL — 레이아웃 파괴 무효 |
| 7-B | #7 recipe, BENIN fold | **0.3191** | **script 가설 반증** (Latin인데 최악) |
| 12-A | DINOv2 ViT-L, EGYPT fold | 0.2144 | FAIL — 용량 무효 (B 0.2115와 동일) |
| 12-B | GUINEA fold (#7 recipe) | **0.0025** | 거의 완벽 — 폴드 지도 완성 |
| 13 | GRL doc-type 적대 분리, EGYPT fold | 0.2583 | **FAIL** — 적대 압력이 불안정화만 |
| 5F | MAURITIUS fold | **0.0006** | 5-fold 지도 완성 (평균 0.1105) |
| **14** | **5-fold 교차 앙상블 제출 (v4)** | **public 0.2318** (v3 0.2826) | **WORKS** — 앙상블이 실전 격차 축소 |
| 15 | + all-types 모델 2개 → 7-model 앙상블 (v5) | public **0.2365** | FAIL — in-domain 전문가는 OOD 다양성 없이 희석만; **v4가 최고 유지** |
| 16 | FDA cross-type 진폭 스왑, EGYPT/BENIN | 0.2091 / 0.3140 | 안정화 + 미미 개선 (벽 못 깸) |
| 17 | v6 = 하드폴드 FDA 교체 앙상블 | public 0.2385 | FAIL — v4(0.2318) 유지. **앙상블 플라토 (±0.007)** |
| — | hflip TTA (오프라인) | -0.001 | 기각 |
| 18 | public PL 의사라벨 4,155장 | MOZ 폴드 0.0188→**0.2572** | **FAIL 조기종료** — PL 노이즈가 학습 오염 |
| 분산 | FDA-EGYPT seed43 | ~0.25 진행 중 (s42 0.2091) | **EGYPT 폴드 시드 노이즈 ±0.04** — 미세 개선 전부 노이즈 판정 |
| **19** | **해상도 518px** (EGYPT 폴드) | **0.1173** (기준 0.2115) | **BREAKTHROUGH** — 개선 0.094 > 2×노이즈. 병목 = 포렌식 디테일 |
| 20 | unfreeze 4블록 (EGYPT 폴드) | 0.2378 | 기각 — 깊이 2 유지 |
| 19-V | 518px 검증 | **BENIN 0.3191→0.1515**, EGYPT s43 0.228 | 518 채택 (BENIN 대폭↓; EGYPT 평균 0.17, 분산 ±0.06) |
| 21 | 518 + EMA (EGYPT) | best 0.166 (≈baseline 0.17) | 무효 — best-ckpt가 이미 붕괴 회피, 안정화는 best 못 올림 |
| 22 | 518 + bona-fide 컴팩트성 (EGYPT) | best 0.174→0.96 붕괴 | **유해** — 정상 뭉침이 위조 분리 파괴 |
| 23 | prototype 거리 헤드 (구현완료, 대기) | — | OOD 강건 헤드 |
| **24** | **APCER@1%BPCER 운영점 대리손실** (EGYPT/BENIN) | EGYPT 0.232 / BENIN 0.240 | **FAIL** — 1% 꼬리는 batch=32로 추정 불가 |
| 25 | MixStyle 특징수준 doc-type 무작위화 (EGYPT/BENIN) | 0.2485 / 0.162 | **FAIL** — hull 내부 보간으론 hull 밖 유형 합성 불가 |
| 26 | TENT 테스트시점 적응 (lr 1e-3) | EGYPT 0.117→0.092 / BENIN 0.152→0.380 | 부분성공 — EGYPT↑ BENIN 붕괴(confirmation bias) |
| **27** | **TENT + 다양성항 + 보수적 lr (2e-5)** | **EGYPT 0.117→0.071** / BENIN 0.167(중립) / GUINEA 무시 | **SUCCESS** — 하드폴드 깬 첫 robust 단일모델 기법 |

### #25 실패 + #26/#27 분석 (핵심 전환점)
- **#25 MixStyle**: 배치 내 *학습 유형들* 스타일 보간 → 홀드아웃 유형은 그 convex hull
  *밖*(EGYPT=유일 아랍문자)이라 합성 불가. 학습시점 트릭 4연속 실패 확정
  (컴팩트성/APCER손실/GRL/MixStyle). 유일 성공은 518px = "무엇을 보는가".
- **메타 결론**: 하드폴드 격차 = 데이터 hull 밖 문제. hull 밖 유형을 *실제로 보는*
  유일한 길 = 테스트시점(FREUID는 테스트 이미지 공개됨).
- **#26 TENT**: 엔트로피 최소화 = 예측 sharpening. EGYPT는 초기예측이 맞아 개선,
  BENIN은 틀린 편향을 강화해 한쪽 클래스 붕괴(고전 TENT collapse).
- **#27 fix**: 배치-주변 엔트로피 최대화(SHOT/IM 다양성항)로 붕괴 방지 + 보수적 lr.
  → EGYPT 큰 이득, BENIN 중립, 쉬운 폴드 무손상. aggressive lr 2e-4면 EGYPT 0.01까지
  가나 BENIN 붕괴 → 안전 운영점은 lr 2e-5.

### 채택된 robust 레버 (모두 "무엇을 보는가", 손실트릭 아님)
1. DINOv2 부분해동2 + DTC + **518px** (BENIN 0.319→0.152)
2. **보수적 TENT 테스트시점 적응** — 5폴드 전체 검증 완료:
   MAURITIUS 0.0006→0.0005, GUINEA 0.0025→0.0036, MOZ 0.0188→**0.0058**,
   EGYPT 0.117→**0.071**, BENIN 0.152→0.167. → LODO 평균 0.058→**0.050**.
- 남은 난제: BENIN 위조 24.8%(1330장) <0.01 완전미탐 = 홀드아웃 시 과소표현된 공격
  스타일(클린 GenAI 추정, 재촬영·블렌드 흔적 無). 미지도 TTA로도 안 풀림.

### 수렴 (#28): 배포 단일 모델
메서드 선택 완료 → 전체 5종 학습(exp_final518_all_s42/s43, 518px 승리 레시피).
in-domain random-val 0.000(포화, 무의미) / honest 추정 = LODO 평균 0.050.
손실트릭 5연속 실패 + "무엇을보는가" 2연속 성공으로 단일모델 최적화 수렴.
다음 contribution(미해결 BENIN 공격군): glyph/semantic 포렌식(novel edge #2) —
공격 sublabel 부재로 blind build, 큰 작업 → 차기 메이저 빌드로 설계 보류.

### #24 실패 원인 + 메타 교훈
운영점 손실의 in-batch 1% 분위수가 batch=32(정상~19개)에서 topk(1)=단일 최대 정상으로
붕괴 → 한 이상치가 임계값을 좌우, 학습 불안정. **1% 꼬리는 미니배치로 추정 불가** (방법론
결함). 메타 교훈: #21/#22/#24 모든 **손실 엔지니어링**이 노이즈 바닥 정체. 유일 성공은
518px(**무엇을 보는가**). → 손실 말고 표현/입력을 바꿔야 함. EGYPT/BENIN 난점의 본질은
train에 쌍둥이 유형 부재 = 도메인 커버리지. #25 MixStyle은 깊은 특징의 유형-스타일을
무작위화해 이 격차를 메움 (픽셀 FDA보다 강력, 유형 정체성이 실제 사는 곳에서 작동).

### 재진단 (2026-06-13): 병목은 안정성이 아니라 APCER@1% 꼬리
EMA/컴팩트 둘 다 baseline 0.17을 못 깸. best-ckpt 선택이 분산을 흡수하므로 안정화제는
무효. 핵심: 하드 폴드에서 **AuDET 0.04-0.07(우수) vs APCER@1% 0.26-0.33(열악)** —
소수 하드 위조가 정상 꼬리 아래로 확신분류되는 게 점수를 죽임. BCE는 평균 분리만
최적화하고 이 운영점은 안 봄. → #24: 정상 점수 (1-BPCER) 분위수를 넘지 못하는 위조에
smooth-hinge 집중 페널티 (대부분 팀이 BCE만 씀 → on-metric contribution).

### 단일 모델 폴드 best (현재 채택 레시피 = DINOv2 부분해동2 + DTC + 518px)
| fold | 378px | **518px** |
|---|---|---|
| MAURITIUS | 0.0006 | — |
| GUINEA | 0.0025 | — |
| MOZ | 0.0188 | — |
| EGYPT | 0.2115 | 0.1173 (s42) / 0.228 (s43) → 평균 0.17 |
| BENIN | 0.3191 | **0.1515** |
병목(EGYPT/BENIN) 합산이 518로 0.53→~0.32. 분산이 다음 과제 → #21/#22.

### 정책 변경 (사용자 지시 2026-06-13 00:15 UTC)
- **앙상블 보류** — 단일 모델 성능 우선 (대회 초기)
- **Kaggle 제출은 사용자 지시 시에만** — best는 기록만
- 채택 기준 격상: 하드 폴드 시드 노이즈 ±0.04 확인 → 개선 폭 ≥0.04만 유의미로 인정

### 제출 추이
v1 0.3574 (ConvNeXt partial) → v2 0.2869 (unfreeze2) → v3 0.2826 (DTC×DINOv2 378)
→ v4 0.2318 (5-fold ensemble, 보류) → **v7 0.2576 (단일모델 배포: DTC×DINOv2 518 all-types)**
단일모델 best = **0.2576** (518+all-types가 378 단일 0.2826 대비 개선). TTA는 infer 미연동
(추론 TTA 연동 시 추가 개선 여지 — LODO에서 검증된 -0.008~-0.046).

### 폴드 지도 (#7 레시피 기준)
GUINEA **0.0025** · MOZ 0.0188 · EGYPT 0.2115 · BENIN 0.3191 · MAURITIUS (진행 중)
- script 무관(BENIN=Latin이 최악). 해석: **닮은꼴 유형이 train에 남아 있으면 쉬움**
  (GUINEA↔MOZ 상호 커버), EGYPT/BENIN은 쌍둥이 없음 → 고유 전이 격차.
- ViT-L 무효 → 용량/특징질이 아니라 **결정 공간에 유형 정체성이 침투**하는 구조 문제.
  남은 카드: GRL(#13), 교차-폴드 앙상블(상이한 실패 방향 상쇄), bona-fide 중심
  one-class 보조.

### 진단 수정 (script 가설 기각, 2026-06-13 04시)
BENIN(Latin) 0.3191 > EGYPT(Arabic) 0.2115 ≫ MOZ(Latin) 0.0188 — 문자 체계와 무관.
폴드별 실패 모드도 상이: EGYPT=정상의 위조화(꼬리 폭발), BENIN=공격 미탐(APCER
0.48 고정). 단일 원인이 아니라 유형별 고유 전이 격차. 공통 패턴: 하드 폴드 best는
항상 ep0-1 → 전이 지식의 본체는 사전학습 특징, 학습은 이를 잠식. 3-fold 평균
0.183 ≈ public 0.283과 정합(public엔 미학습 2종 포함 추정).
처방: (A) 더 강한 사전학습 특징(ViT-L), (B) 실패 방향이 다른 폴드 모델들의
교차-폴드 앙상블 — 미학습 유형에 대한 일반화 방향 다양성 확보.

### EGYPT 벽 재해석 (2026-06-13 02시): SCRIPT NOVELTY 가설
#10/#9-V 실패 패턴: best가 항상 ep0, 붕괴 시 AuDET는 0.1-0.3인데 APCER@1%만 0.98 —
공격을 못 잡는 게 아니라 **EGYPT 정상 문서가 위조로 분류**되는 것. 결정적 사실:
**EGYPT/DL은 5종 중 유일한 아랍 문자**(나머지 4종 Latin). 학습할수록 "Latin 문서
외형 = 정상"에 고착 → 아랍 문서 전체가 이상치화. SBD가 못 깬 이유: 조작 흔적
커버리지 문제가 아니었기 때문. 처방 = 전역 정체성(script/layout) 제거: #11 patch
학습 (RandomResizedCrop 8-35% + 5-crop 평가, FakeIDet 0% EER 선례). 검증 = #7-B
(Latin인 BENIN 폴드가 MOZ처럼 0.0x면 가설 확정).

### EGYPT 벽 (the wall, ~0.21)
모든 레시피가 EGYPT 폴드에서 ~0.21 고정. 원인 가설: EGYPT 공격(7,867건, 최다)에
다른 유형에 없는 공격 스타일 존재 → 판별기가 "본 적 없는 조작 방식"을 잡지 못함.
유형 정체성 문제가 아니라 공격 스타일 커버리지 문제. private의 미공개 2개 유형이
정확히 이 상황 재현 예정. 처방: #10 SBD (조작 일반의 흔적을 자가 합성으로 학습 —
SBI(CVPR'22)의 문서 버전, 정상끼리 영역 합성→유사 공격 생성, fraud=1 & cons=0
이중 감독). 데이터 처방 대조군: #9-V.

## Failure analyses (the why, per iteration)
- **#2 global FFT**: FFT magnitude of the whole image encodes layout/font spectrum =
  doc-type identity → branch learned doc-type shortcut, collapsed on unseen type.
  Lesson: forensic features must be LOCAL/texture-level, not global-structural.
- **#3 heavy aug / HPF**: augmentation changes inputs but not what ConvNeXt-T@384 can
  represent about unseen-type attacks; ~50% of unseen-type attacks rank below bona-fide
  tail regardless. Lesson: representation capacity/pretraining is the bottleneck, not
  input diversity.
- **#4 DTC on ConvNeXt**: consistency aux gives small consistent gain (0.4059 vs
  0.4087 ablation) but the host backbone misses the same hard attacks. Lesson: the
  method is directionally right; the host features are too weak.
- **#5 frozen DINOv2**: doc-type-novelty→attack conflation (unseen-type bona-fides all
  scored ~1.0; bona median 0.9868 on holdout vs 0.12 in-domain; APCER tail crushed).
  Train loss fell while val APCER exploded at ep2 = adapter overfit to seen-type
  semantics. Lesson: semantic features need SOME plasticity to become forensic;
  also AuDET/APCER are rank-based → calibration cannot fix anything.
- **#6b why it works**: plasticity (last 2 blocks) lets features shift from "which
  document" to "what texture/artifact"; 378px restores high-frequency forensic detail
  destroyed at 224. AuDET 0.0138 on unseen type at ep2.
- **#6a final**: DTC on ConvNeXt full data = 0.3996 vs baseline 0.4030 — the aux signal
  helps but the host ceiling binds (same lesson as #3/#4). DTC's true test is #7.
- **#6b final (15ep): late-epoch RE-COLLAPSE.** Best ep~5 (FREUID 0.0142, AuDET 0.0019)
  but ep11→14 APCER explodes 0.54→0.83 while train loss falls — the frozen-DINOv2
  doc-type overfit returns slowly through the unfrozen blocks. freuid_score best-ckpt
  selection (early stopping) saved the run. Lessons: (a) recipe sweet spot ~5-6 epochs;
  (b) watch whether #7's consistency loss delays this collapse — if yes, "DTC as
  domain-overfit regularizer" is a paper-grade ablation finding.
- **#6b' public gap (0.014 holdout vs 0.287 public)**: our holdout only measures
  doc-TYPE transfer within born-digital data (train is 99.97% digital). Public test
  likely adds CAPTURE-style shift (physical print-and-capture; official "hybrid"
  wording) and possibly the 2 unseen doc types. Prediction distribution on public is
  healthy (46% flagged, 2.4% uncertain) → errors are CONFIDENT mistakes on a shifted
  subpopulation, not systematic over-flagging. The 20 labeled physical train images
  are contaminated as a probe (most were in train) but 3/14 physical attacks still
  scored low. Remedies queued: #7 (TraceMix/consistency = capture-axis defense),
  #9 FantasyID joint (physical-capture bona-fides), and a future capture-style
  holdout protocol if labels allow.

- **#7 verdict**: best 0.0188 (ep1) ≈ #6b 0.0142 — peak parity. BUT collapse dynamics
  differ decisively: ep12-14 #6b → 0.71-0.81 vs #7 → 0.12-0.14. **The TraceMix
  consistency loss is a domain-overfit regularizer** (keeps gradient pressure on
  capture-trace features, away from doc-type semantics). Paper-grade ablation.
- **#6b-V (EGYPT holdout) verdict**: best 0.2107, collapses by ep2 (vs ep11 on MOZ).
  Doc-type transfer difficulty varies 15× across folds; EGYPT (most/hardest attacks)
  is the honest proxy — and 0.21 ≈ public 0.28, largely EXPLAINING the public gap.
  Protocol upgrade: judge all future candidates on the EGYPT fold (or MOZ+EGYPT mean),
  not MOZ alone.
- **v3 public (#7 best): 0.2826** vs v2 0.2869 — DTC alone barely moves public;
  consistent with the gap being hard-TYPE transfer, not only capture style.

## Next iterations (queue, contribution-grade)
- **#7 DTC × DINOv2-PE (READY)**: graft TraceMix+consistency onto the partial-unfrozen
  DINOv2 host — the paper's full method on the winning base. Tests whether the DTC
  contribution adds margin on a strong host (the partial-data DTC gain suggests yes).
- **#8 doc-type adversarial disentanglement (GRL)**: if #7 shows residual doc-type
  sensitivity (verify via 2nd holdout: leave-EGYPT-out), add gradient-reversal doc-type
  head to remove type identity from the decision representation.
- **#9 FantasyID joint**: +13 doc types incl. Arabic/Persian → data-level type diversity.
- **#10 ensemble of complementary failure modes** (DTC-ConvNeXt misses attacks;
  DINOv2 variants over-flag) + 3-seed variance check on the winner.
- Verify-before-trust for #6b: (a) re-run with leave-EGYPT-out, (b) public LB submit.

## TTA submit result: public 0.21762 (worse than 728 no-TTA 0.20375). TTA hurts in-domain public; retained as private-shift card. Best=728 single 0.20375.

## D9 (896px resolution sweep, 2026-06-14)
- **896 EGYPT LODO**: ep1 best **0.0217** (AuDET 0.0031 / APCER@1% 0.0395 / AUC 0.9969).
  Beats 728 EGYPT (0.0343) → resolution lever STILL not saturated:
  378→518→728→896 = 0.2115→0.117→0.0343→**0.0217** on EGYPT.
  Classic late collapse (ep2 0.273, ep3 0.450, monotone) — killed at ep3, best.pt=ep1 locked.
- **896 BENIN LODO**: ep2 best **0.0780** (AuDET 0.0299 / APCER@1% 0.1215 / AUC 0.9701).
  Beats 728 BENIN (0.1325). Descended 0.31→0.18→**0.078**(ep2) then collapsed
  (ep3 0.281, ep4 0.186) — killed at ep4, best.pt=ep2 locked.
- **VERDICT: 896 wins BOTH hard folds vs 728** — EGYPT 0.0343→0.0217, BENIN 0.1325→0.0780.
  Resolution lever still the dominant, unsaturated axis. → train 896 all-types deployment.
- **896 all-types deployment LAUNCHED** (user-approved 2026-06-14, exp_deploy896_all.yaml,
  GPU0, train=61030/val=8322, holdout=""). On completion: FantasyID xeval + public infer
  (3-axis), then submit-version prep. Kaggle submit GATED on explicit user instruction.
- Staged next probe: exp_d9_res1036_egypt.yaml (1036px; GPU mem ample — 896/b8 used only
  ~4.6GB of 98GB, so CPU-aug throughput, not VRAM, is the limit).

## D9b (896 all-types DEPLOYMENT — 3-axis verified, 2026-06-14)
- Config exp_deploy896_all.yaml, aggressive GPU (batch 48 ≈22GB/98GB, workers 28, lr 1e-4
  unchanged, 6 ep). random-val FREUID saturated 0.0008→1.4e-6 (in-domain leakage, NOT a
  selection signal) → best.pt = ep5.
- **3-axis verdict (new best deployment):**
  - Axis-1 LODO: 896 wins both vs 728 — EGYPT 0.0343→0.0217, BENIN 0.1325→0.0780.
  - Axis-2 FantasyID xeval: AUC 0.5612 / score 0.9578 ≈ baseline 0.586 (no regression;
    unfair ~chance proxy, deweighted).
  - Axis-3 public: submission_896_all.csv preds rank-corr 0.953 (spearman) with the 728
    submission (public 0.20375); midzone 3.1% — sane, no regression. Exact public score
    needs a Kaggle submit (held for user instruction).
- **submission_896_all.csv ready** (142,818 rows: 7,821 real preds + 134,997 filled 0.5).
- infer.py gotcha: MUST pass --existing-only with --fill-missing (else it tries to load
  all 142,818 incl. the 134,997 unpublished private ids → FileNotFoundError crash).
- Open: best.pt=ep5 may slightly over-fit seen-type appearance vs LODO's ep1-2 optimum;
  no held-out signal to pick earlier epoch for an all-types model. Optional hedge: save an
  earlier-epoch conservative deployment variant.

## D9c (1036px — resolution sweep CONCLUDED, 2026-06-14)
- 1036 EGYPT: ep1 best **0.0121** (beats 896 0.0217) — EGYPT monotone to 1036.
- 1036 BENIN: UNSTABLE — ep0 0.325 / ep1 0.893(collapse) / ep2 0.307; oscillates, best
  0.307 ≫ 896 BENIN 0.078. 1036 does NOT generalize across folds (breaks BENIN).
- **VERDICT: 896 is the deployment-resolution sweet spot. 1036 overshoots (EGYPT-only
  gain, BENIN instability) → resolution sweep CLOSED at 896.** Sweep:
  EGYPT 0.2115→0.117→0.0343→0.0217→0.0121 ; BENIN 0.3191→0.152→0.1325→**0.0780**→0.307(broke).
- Submission: submission_896_all.csv public **0.21591** — slightly WORSE than 728
  all-types (0.20375). LODO-vs-public axes diverge: 896 wins LODO (private proxy) but
  regresses the saturated public slice (+0.012). Hypothesis: best.pt=ep5 in-domain overfit.
- Next: (C) conservative early-epoch 896 all-types deploy (epochs~3) to test the ep5-overfit
  hypothesis + recover public; (B) appearance-invariance aug LODO experiment.

## D10 (appearance-invariance augmentation, 2026-06-14) — user-chosen direction
- Motivation: 5-type analysis showed the types differ chiefly in COLOR/palette (GUINEA sat
  0.20 vs MOZ 0.06) and share ISO ID-1 framing; the 2 private types differ mainly in
  appearance. Hypothesis: randomize color/palette → model relies on type-invariant forensic
  micro-texture, not memorized type colors → better unseen-type generalization.
- Impl: appearance_invariance_ops() = ColorJitter(b0.4 c0.4 s0.6 hue0.25 p0.9) + RGBShift +
  HueSaturationValue + ToGray(0.15) + ChannelShuffle(0.10), applied to BOTH classes on uint8
  BEFORE TraceMix. New config flag appearance_inv; wired config→transforms→dataset→train.
- **EGYPT LODO: ep1 best 0.00259** (AuDET 0.0007 / APCER@1% 0.0044 / AUC 0.9993) vs 896
  baseline 0.0217 — **~8.4x better**, beats even 1036 (0.0121). AND kills the collapse
  (ep2 0.0031, no APCER blowup; plain 896 went 0.273 at ep2). Generalization + stability.
- BENIN LODO: launched (exp_d10_appinv_benin) for cross-fold validation vs 896 BENIN 0.0780.
  If it also wins → integrate appearance_inv into 896 all-types deployment.

## D10 cross-fold verdict (2026-06-15): FOLD-DEPENDENT, like 1036 — REJECT as uniform lever
- **BENIN LODO: ep1 best 0.1900** vs 896 baseline 0.0780 → ~2.4x WORSE (then collapses
  ep3-4). Opposite sign to EGYPT (8x better). best.pt=ep1 locked, killed at ep4.
- Net: EGYPT helped by 0.019, BENIN hurt by 0.112 → harm > help. For 2 unknown private
  types, expected net NEGATIVE. Aggressive appearance_inv rejected as a uniform deployment
  lever (same fate as 1036: wins one hard fold, breaks the other).
- **Diagnosis (publishable, fold-idiosyncratic):** EGYPT is the color/appearance-OOD fold
  (unique Arabic palette) → de-emphasizing color stops it being flagged. BENIN's genuine/
  attack separation partly LIVES in chroma → destroying color removes a real cue. EGYPT and
  BENIN have OPPOSITE needs; a single global color aug can't satisfy both.
- Next candidates: (a) MILDER appearance_inv (drop ToGray/ChannelShuffle, lower p~0.3-0.5 so
  the model still sees real color most of the time) — retest both folds for a net-positive
  sweet spot; (b) reject & keep 896 all-types deploy; (c) wire conservative TTA into
  deployment inference (validated robust lever); (d) conservative early-epoch 896 deploy.

## Ensemble probe (user insight, 2026-06-15): baseline + D10-appinv, per fold
scripts/ensemble_fold_eval.py (score-avg N ckpts on a held-out doc_type).
| fold | baseline | appinv | ens mean-prob | ens rank-mean |
|---|---|---|---|---|
| EGYPT | 0.0217 | 0.0026 | 0.0029 | 0.0030 |
| BENIN | 0.0780 | 0.1900 | 0.1704 | 0.1269 |
- EGYPT: ensemble KEEPS the appinv gain (0.003) — a decent member (baseline AUC 0.997) +
  an excellent one averages near-excellent. Validates the user's "ensemble keeps EGYPT" idea.
- BENIN: ensemble 0.127 (rank) WORSE than baseline 0.078 — the appinv member is confidently
  WRONG on BENIN (AUC 0.926, APCER 0.28) and drags the average down. rank-mean >> mean-prob.
- 2-fold mean: baseline 0.050 < ensemble(rank) 0.065 < appinv 0.096. **Simple baseline+appinv
  ensemble LOSES** (BENIN harm 0.049 > EGYPT gain 0.019). A net-win needs a member that isn't
  BENIN-toxic → D10b MILD appinv (drop ToGray/ChannelShuffle, lower p). Launched on EGYPT.

## D10b (MILD appearance-inv) + ensemble — appearance-inv line CLOSED (2026-06-15)
| fold | baseline | mild | base+mild ens(rank) | (strong | base+strong ens) |
|---|---|---|---|---|---|
| EGYPT | 0.0217 | 0.0052 | 0.0046 | 0.0026 | 0.0030 |
| BENIN | 0.0780 | 0.1590 | 0.1102 | 0.1900 | 0.1269 |
- 2-fold mean: **baseline 0.0499** < base+mild-ens 0.0574 < base+strong-ens 0.0650 <
  mild 0.0821 < strong 0.0963. **Baseline alone wins every comparison.**
- VERDICT: appearance-invariance REJECTED as deployment lever (single AND ensemble). EGYPT
  (appearance-OOD fold) loves it 4-8x; BENIN (chroma-dependent attack-subtlety fold) is hurt
  more, every strength. The two hard folds have ANTAGONISTIC needs → no uniform color aug or
  2-model ensemble nets positive. Confirmed across strong/mild × single/ensemble (4 configs).
- PAPER VALUE: clean demonstration that fold hardness has two distinct causes (appearance-OOD
  vs attack-subtlety) requiring opposite treatments — a fold-idiosyncratic transfer result.
- OPEN (unmeasurable): private's 2 unseen types are appearance-NOVEL (EGYPT-like) so appinv
  *might* help them; but BENIN-style chroma types would be hurt. Not selectable on LODO → keep
  baseline. Could retain an appinv all-types model as a RESERVE if private looks appearance-novel.
- Next validated lever: wire conservative TTA into 896 deployment inference (the one lever that
  helped EGYPT/MOZ without BENIN harm — test-time, not train-time).

## D11 (chroma-consistency stream, 2026-06-15) — new contribution targeting BENIN tail
- Motivation (evidence-driven): the D10 appearance-inv ablation proved BENIN's genuine/attack
  separation lives in CHROMA. BENIN's open failure is the clean-GenAI tail (APCER@1% 0.12, no
  texture trace → DTC trace branch & resolution miss it). Clean-GenAI edits (face-swap/inpaint)
  leave subtle LOCAL chroma / white-balance discontinuities even with no high-freq texture trace.
- Impl (DTC extension "trace+chroma consistency"): ChromaResidual = denorm → opponent color
  (RGB−luminance) → local high-pass; fed through a parallel TraceEncoder → patch chroma
  embeddings → _sim_stats chroma-consistency vector. Fusion head gets chroma global feat +
  chroma stats; cons_head sees trace+chroma stats (self-supervised on the same TraceMix label).
  Config flag use_chroma; threaded config→classifier→dtc→train + infer/ensemble eval. +0.46M
  params. Smoke OK (head 1038, cons_head 14=2×7).
- Test: 896 BENIN LODO vs baseline 0.0780 (launched exp_d11_chroma_benin). If it cuts the
  BENIN tail without hurting EGYPT (chroma is additive), it's a real on-target contribution.

## D11 chroma-consistency BENIN result (2026-06-15) — NEUTRAL, tail not cracked
- BENIN LODO ep1 best 0.0859 vs baseline 0.0780 (within BENIN seed noise ±0.04 → neutral,
  marginally worse). Split: AuDET 0.030→0.023 (AUC 0.970→0.977, ranking slightly BETTER) but
  APCER@1% 0.122→0.141 (the targeted clean-GenAI TAIL slightly WORSE).
- chroma-consistency adds a little general separability but does NOT crack BENIN's 1% tail.
  Likely cause: high-passed chroma removed the LOW-freq region-level white-balance signal,
  keeping only local boundary noise; RGB DINOv2 may already capture chroma. BENIN tail now
  resisted by CLIP(D7), resolution, AND chroma → genuinely hard (≈FantasyID clean-GenAI).
- Refinement option: chroma WITHOUT high-pass (raw opponent color) for region-level white-
  balance; else conclude tail resistant & consolidate.

## D12 (GenAI frequency-fingerprint branch, 2026-06-15) — user-chosen, BENIN tail attempt #4
- Idea: clean GAN/diffusion leaves periodic UPSAMPLING artifacts in the Fourier spectrum even
  with no spatial trace (Durall/Frank/Zhang lineage). SpectralFingerprint = grayscale → 2D FFT
  → |.| → AZIMUTHALLY-AVERAGED 1D radial power spectrum (128 bins) → per-sample shape-normalize
  → small MLP → 64-d feat into the fusion head. Radial averaging is layout/translation-INVARIANT
  → avoids the doc-type-layout shortcut that sank raw 2D global-FFT (D2). Only +25k params.
- Config flag use_spectral; threaded config→classifier→dtc→train + infer/ensemble. Smoke OK
  (head 967 = 768+128+7+64; AMP-safe via float32 FFT).
- Test: 896 BENIN LODO vs baseline 0.0780 (exp_d12_spectral_benin). Target = cut APCER@1% tail.

## D12 spectral-fingerprint BENIN result (2026-06-15) — NEGATIVE, hurts + unstable
- BENIN LODO ep2 best 0.1400 vs baseline 0.0780 — ~1.8x WORSE (AuDET 0.030→0.058, APCER@1%
  0.122→0.209, both worse) and oscillates wildly (0.15/0.38/0.14/0.37/0.35). Overfits a
  spurious training-type spectrum, fails to transfer to held-out BENIN.
- Doc-image radial spectra are dominated by document content/print structure; the GenAI
  upsampling fingerprint (if any) is swamped. REJECTED.
- **BENIN clean-GenAI tail now resisted by FOUR distinct mechanisms**: CLIP semantic (D7),
  resolution, chroma-consistency (D11 neutral), GenAI spectral fingerprint (D12 worse). Strong
  evidence the tail is unbeatable by forensic features (≈FantasyID: clean GenAI ~chance for all
  archs). → STOP attacking the tail; consolidate. Code kept behind flags for paper ablations.

## CONSOLIDATION (2026-06-15) — BENIN tail closed (4 mechanisms), search converged
Method search exhausted: resolution (896 ceiling), appearance-inv (rejected, fold-antagonistic),
BENIN clean-GenAI tail (CLIP/resolution/chroma/spectral all fail → forensic-unbeatable). Deployment
= 896 all-types baseline. Consolidation tasks (user-approved):
1. Conservative early-epoch 896 deploy (exp_deploy896_all_ep3, epochs=3 → best≈ep2, batch 48):
   test if it recovers the public regression (ep5 model: 0.21591 vs 728 0.20375) w/o losing LODO.
2. TTA private-shift card: conservative TENT on the chosen 896 deploy → submission variant, ready
   for private-image drop (validated LODO win, hurts in-domain public so reserved for private).

## Consolidation results (2026-06-15)
- **Early-epoch 896 deploy (ep3, best≈ep2) — NO public recovery.** Public-dist proxy on the
  7,821 real imgs: ep3 vs ep5 rank-corr 0.965 (≈identical); ep3-728 0.951 < ep5-728 0.953 (ep3
  NOT closer to 728). The 896 public regression (0.216 vs 728 0.204) is intrinsic to 896's
  resolution on the saturated slice, NOT ep5 seen-type overfit → not fixable by epoch selection.
  Verdict: early-epoch hedge is a wash; keep 896 ep5 as deployment (submitted, public 0.21591).
  ep3 submit NOT worth a slot (proxy says ~0.216). submission_896_ep3.csv kept as backup.
- **TTA private card**: conservative TENT (LN-only, entropy+diversity, lr 2e-5) on 896 ep5 →
  submission_896_tta.csv. Reserved for private-image drop (TTA wins unseen-type LODO but hurts
  in-domain public, so NOT submitted to the public LB).

## Priority-A (goal #1): type-invariant via genuine-diversity + semantic-consistent attacks (2026-06-15)
- Refined recipe (distinct from failed D2-D4 naive concat): genuine = FREUID + IDNet-positive +
  DocXPand + BID (87,942, 20 doc types); attack = FREUID-real ×2 upweight + IDNet-inpaint GenAI
  only (76,631) — NO IDNet-crop, NO SBD (semantic-consistent fraud). manifests/union_priorityA
  .parquet, 164,573 rows, 53:47 balance. Addresses D3's tension: broad genuine (cuts unseen-type
  false-flag 78%→32%) + restored attack sensitivity (upweight + consistent GenAI attacks).
- Screen: 518px EGYPT LODO (exp_e_priorityA_518_egypt, faster; matches D2-D4) vs FREUID-only 518
  EGYPT 0.117 + FantasyID AUC 0.586. If beats both → scale to 896 (vs FREUID-only 896 EGYPT 0.0217).
- Launched autonomously (goal #1, GPU idle, user away). Honest prior: external data regressed in
  D2-D7, low odds; but refined recipe untested + negative result strengthens "FREUID-only wins" paper.

## Priority-A RESULT (2026-06-15) — REJECTED on BOTH axes, external-data dead-end confirmed
- EGYPT 518 LODO: ep2 best 0.235 vs FREUID-only 0.117 → ~2x WORSE (stuck 0.23-0.25, AUC 0.89).
- FantasyID xeval: AUC 0.526 / score 0.969 vs FREUID-only 0.586 → WORSE (genuine-diversity did
  NOT even improve the false-flagging axis at the AUC level).
- The goal's refined recipe (broad genuine + semantic-consistent attacks = FREUID-real x2 +
  IDNet-inpaint only, no crop/SBD) was meant to fix D2-D4's failures. It does NOT — external data
  muddles the fraud boundary regardless of attack-semantic consistency. Dead-end even best-curated.
  Did NOT run BENIN/896 (EGYPT screen failed clearly).
- **CLOSURE: all goal strategies exhausted — A(external) rejected, B(capture-aug) rejected (D5),
  C(TTA) done, D(DTC) done. FINAL deployment = FREUID-only DTC-DINOv2 896 all-types + TTA card.
  Nothing beats it on the 3 validation axes. Remaining = await private images -> re-infer.**

## TTA on 896 deploy — public RESULT (2026-06-15): TTA HELPS public (prediction was wrong)
- submission_896_tta.csv public **0.21083** vs 896 no-TTA 0.21591 → TTA IMPROVES 896 public by
  0.005. Contradicts the 728-TTA precedent (which hurt: 0.20375->0.21762). On the 896 model,
  conservative TENT helps BOTH the saturated public slice AND unseen-type LODO (validated).
  → 896-TTA dominates 896 no-TTA on both measurable axes; it is the better deployment (not just
  a private card). Public LB: 728 no-TTA 0.20375 still lowest raw-public, but 896-family wins LODO
  (private proxy) — deployment = 896-TTA by the selection protocol.
- Private-image release date: NOT officially published (organizer site + Kaggle have no date;
  "exact flow soon"). Submission deadline confirmed 2026-07-14 11:59 AoE. Monitor cron active.

## NEW GOAL (2026-06-15): deep-research → 10x plan → "Dual-Expert FREUID"
deep-research (110 agents, 17 verified findings) → the two hard folds need OPPOSITE remedies:
- EGYPT (semantic/appearance-OOD, genuine unseen-type flagged): SEMANTIC-SHORTCUT SUPPRESSION —
  GSD (frozen-CLIP SVD subspace projection F'=F(I-UUᵀ)), MPFT texture-masking, intermediate CLIP
  layers (RINE/L12), LN-tuning. "Semantic fallback" (arXiv 2603.09242) = forensic FT leaves rep
  organized by semantics not forensics → our EGYPT mechanism.
- BENIN (clean-GenAI trace-free tail): GENUINE-MANIFOLD ANOMALY, never train on fakes —
  CLIP-Flow (NLL, +19pt unseen-gen), MIRROR (memory-bank reconstruction residual, +8.1% on hardest
  OOD), one-class. Open Q: does it give ANY BENIN separation or is residual ~0 (true trace absence)?
- Plan = Dual-Expert rank-fusion: forensic(896+DTC)+GSD for EGYPT  ⊕  genuine-manifold anomaly for
  BENIN. Rank-based FREUID tolerates fusing calibration-free anomaly with detector score.
- Caveat: ALL findings from natural-image/face deepfakes, NONE on ID docs → recipes to PORT.
- EXP-1 (start): genuine-manifold anomaly expert — frozen CLIP/DINOv2 features of TRAIN-genuine,
  Mahalanobis/kNN one-class, score held-out fold; FREUID for anomaly-alone + rank-fused w/ 896.

## EXP-1 (genuine-manifold anomaly, BENIN+EGYPT) — REJECTED (2026-06-15)
scripts/anomaly_expert_eval.py: frozen CLIP ViT-B/16 features of 20k train-genuine, Mahalanobis
+ kNN one-class, score held-out fold. Results:
- BENIN: maha/knn-alone FREUID 0.959/0.967 (AUC 0.577/0.533 = ~chance); fusion w/ 896-detector
  HURTS (0.078→0.189). EGYPT: maha/knn 0.968 (AUC 0.534); fusion HURTS (0.022→0.113).
- Genuine-manifold anomaly (CLIP-Flow/MIRROR paradigm) gives ~CHANCE separation on BOTH folds
  and only ADDS NOISE to the detector. Confirms deep-research domain-gap caveat: the paradigm
  does NOT transfer to FREUID docs. Answers open-Q#1: BENIN trace is information-theoretically
  ~absent (not an optimization failure). Dual-Expert BENIN-anomaly branch DEAD.

## D13 (GSD semantic-decoupling, 2026-06-15) — gradient-training, EGYPT target
Impl: suppress_dominant_subspace(C, r) removes top-r right singular vectors of the batch from a
FROZEN CLIP stream (F'=F(I−UUᵀ)) → semantic-suppressed forensic residual fed to the fusion head.
Config flag use_gsd/gsd_r; threaded config→dtc→classifier→train→infer. Smoke OK (head 1671).
Launched exp_d13_gsd_egypt (518px, batch 24, EGYPT LODO) vs FREUID-only 518 EGYPT 0.117. Targets
EGYPT semantic-fallback (genuine unseen-type flagged as attack). Note: per-batch SVD subspace is
batch-composition-dependent at eval — if marginal, refine to a fixed calibrated subspace buffer.

## D13 GSD result (2026-06-15) — REJECTED (CLIP-stream GSD hurts EGYPT like raw CLIP D7)
- EGYPT 518 LODO: ep0 best 0.2588 vs FREUID-only 0.117 → WORSE; collapses ep1-2 (0.90).
- Root: CLIP stream itself hurts EGYPT (D7 raw-CLIP 0.224). GSD suppression of CLIP residual does
  not rescue. Both deep-research clusters FAILED on docs (anomaly ~chance + GSD worse) → universal
  domain-gap caveat confirmed; natural-image/face SOTA does not transfer. 896+DTC+TTA remains best.

## D14 (resolution tiling, 2026-06-15) — user-chosen, build on the validated resolution lever
Idea: resizing the native ~1585x1000 card to 896 loses ~half the detail; train on LARGE native-res
TILES (896 crops, scale 0.4-0.8 of the card = higher effective resolution) + 5-crop eval aggregation.
Fixes iteration #11 (tiny 0.08-0.35 patches destroyed layout → 0.236). Reuses patch_mode infra;
exposed patch_scale_min/max + patch_eval_frac as config. Launched exp_d14_tile_egypt (896, scale
0.4-0.8, frac 0.75, EGYPT LODO) vs whole-doc 896 EGYPT 0.0217. If tiling < 0.0217, the resolution
lever extends past the 896 single-pass ceiling without 1036's BENIN instability.

## D14 tiling result (2026-06-15) — REJECTED (context loss kills the tail, like #11)
- EGYPT 896 tile: ep1 best 0.192 vs whole-doc 0.0217 → 9x WORSE (AUC 0.991 ok but APCER@1% 0.318
  vs 0.040 — tiling destroys the operating point). Large tiles did NOT fix #11's context-loss.
- 3rd new-goal experiment rejected (anomaly+GSD+tiling). 896+DTC+TTA confirmed ceiling.

## D16 (2026-06-15): unseen-type inference → "Unseen-Type-Calibrated 5-fold LODO Ensemble"
User request: infer the 2 unseen private doc types from the 5 known, then plan training.
- INFERENCE (5 measured types → 2 unseen): all 5 are ISO ID-1 (~1.585), pre-cropped, born-digital,
  African (4 Latin: FR/PT/EN + 1 Arabic EGYPT). Competition = 7 types, "African/Asian, Latin+Arabic",
  private = 2 unseen + captured. → unseen likely: ≥1 ASIAN type (none seen; Latin SE-Asian or
  non-Latin Persian/Urdu/Thai/Devanagari) + possibly another African (Arabic-script N.African). Both
  near-certainly ID-1 pre-cropped, appearance/script-OOD (EGYPT failure mode), + capture-domain shift.
- PLAN: 896 5-fold LODO ensemble (each member blind to one type → calibrated to NOVELTY = the unseen
  scenario; #14 validated public 0.2826→0.2318) + per-member conservative TTA + validate on EXTERNAL
  Asian/Persian/Arabic proxy-folds (FantasyID/IDNet) matching the inferred unseen characteristics.
  External used for VALIDATION only (training-mix failed D2-D7). Have EGYPT 0.0217 + BENIN 0.0780;
  training the 3 missing 896 members (GUINEA/MAURITIUS/MOZ — easy folds). Then assemble + proxy-validate.

## D16 5-fold members complete (896 LODO, 2026-06-15)
EGYPT 0.0217 · BENIN 0.0780 · GUINEA 0.0323 · MAURITIUS 0.00026 · MOZ 0.00131 → LODO mean 0.0267.
Assembling 5-fold LODO ensemble. KEY VALIDATION (scripts/ensemble_fantasyid_eval.py): does the
ensemble beat the single 896 all-types (FantasyID AUC 0.561) on the unseen-type proxy (FantasyID
13 types incl Arabic/Persian + capture shift)? If ensemble AUC > 0.561, the novelty-calibration
helps the inferred unseen private types → adopt LODO ensemble (+TTA) as the unseen-type deployment.

## D16 ensemble validation (2026-06-15) — LODO ensemble does NOT clearly beat single 896+TTA
- FantasyID proxy: 5-fold ensemble AUC 0.549-0.552 vs single 896 0.561 — NO win (but FantasyID is
  the known UNFAIR proxy, all ~chance → inconclusive).
- Public-distribution: ensemble mid-zone 18.0% vs single 3.1% — each LODO member is blind to one
  SEEN type → uncertain on public's seen types → averaging muddles separation (deepens #15). Since
  private is mostly seen types + a minority of unseen, the seen-type degradation likely outweighs
  the novelty benefit. submission_896_ens5.csv generated; definitive public needs a Kaggle submit.
- **New-goal verdict (deep-research → 4 experiments: anomaly/GSD/tiling/LODO-ensemble): NONE beats
  896+DTC+TTA. 10x not achievable via researched or ensemble levers. 896 all-types + conservative
  TTA (public 0.21083) remains the validated best deployment for the inferred unseen private types.**

## FAILURE ANALYSIS → D17 TTT-DTC (test-time training via self-supervision, 2026-06-15)
Failure pattern across ALL experiments: (a) adding a stream (CLIP/chroma/spectral/GSD) hurts/neutral
— host already optimal; (b) losing context (tiling/patches) destroys the APCER@1% tail; (c) mixing
external data muddles the boundary; (d) ensemble-averaging blind LODO members hurts seen types.
META: every TRAINING-time change fails; only "what the model sees" (resolution, capped 896) and
TEST-TIME adaptation work; EGYPT/BENIN need OPPOSITE train-time remedies → the only cross-fold lever
is TEST-TIME. → D17 plan: TTT-DTC = Test-Time Training using our built-in self-supervised DTC
consistency task (TraceMix). Adapt LayerNorm at test time by minimizing the consistency self-sup
loss on test images (more forensic-targeted than TENT entropy), predict on clean. Reproduces NO
failure mode (no stream/context-loss/data-mix/averaging). scripts/ttt_dtc_eval.py compares
no-TTA / TENT / TTT-DTC on each LODO fold.

## D17 TTT-DTC result (2026-06-15) — consistency adds nothing over TENT; but TENT CONFIRMED strong
| fold | no-TTA | TENT | TTT-DTC | TENT+TTT |
|---|---|---|---|---|
| EGYPT | 0.0217 | 0.0097 | 0.0100 | 0.0097 |
| BENIN | 0.0780 | 0.0496 | 0.0516 | 0.0497 |
| 2-fold mean | 0.0499 | 0.0297 | 0.0308 | 0.0297 |
- TTT-DTC (entropy[LN] + self-sup consistency[trace]) ≈ TENT on BOTH folds → the DTC consistency
  signal is REDUNDANT with entropy at test time. REJECTED (no gain). 5th new-goal experiment that
  fails to beat the existing best.
- KEY POSITIVE: conservative TENT at 896 gives LODO mean 0.0499→0.0297 (~40% on the private proxy),
  helps BOTH hard folds, NO BENIN collapse. Strongest validated single improvement we have →
  deployment = 896 all-types + conservative TENT (submission_896_tta.csv) is confirmed best.
- All 5 new-goal experiments (anomaly/GSD/tiling/ensemble/TTT) fail to beat 896+DTC+TENT. Ceiling firm.

## D18 (2026-06-15): user seen-type insight → 728+896 RESOLUTION-DIVERSITY ensemble WINS LODO
User flagged public 0.21 matters (private includes the 5 seen types, not only the 2 unseen). This
restored seen-type as a selection axis and revealed RESOLUTION ANTAGONISM: 728 better on seen
(public 0.204 vs 896 0.211), 896 better on unseen (LODO). A 728+896 ensemble (both all-types →
keeps seen confidence, adds resolution diversity; unlike the failed 5-fold LODO-blind ensemble):
| fold | 728 | 896 | 728+896 rank-mean |
|---|---|---|---|
| EGYPT | 0.0343 | 0.0217 | 0.0063 (3.4x better than 896!) |
| BENIN | 0.1325 | 0.0780 | 0.0846 (slightly worse) |
| 2-fold mean | — | 0.0499 | 0.0455 (NET WIN) |
- First new-goal experiment to BEAT 896. rank-mean >> prob-mean (EGYPT 0.0063 vs 0.0131).
- Complementarity: TENT better on BENIN (0.0496), ensemble better on EGYPT (0.0063) → TENT-on-
  ensemble could give EGYPT ~0.006 + BENIN ~0.05 ≈ mean 0.028. Generating 728+896 public submission.

## D18b TENT-on-ensemble (2026-06-15/16) + LEADERBOARD SHIFT
| config | EGYPT | BENIN | 2-fold mean |
|---|---|---|---|
| single 896 | 0.0217 | 0.0780 | 0.0499 |
| single 896+TENT | 0.0086 | 0.0479 | 0.0283 |
| 728+896 ens | 0.0063 | 0.0846 | 0.0455 |
| 728+896 ens+TENT | 0.0029 | 0.0517 | 0.0273 |
- TENT is the dominant lever (896: 0.0499→0.0283, ~43%). Resolution-ensemble helps EGYPT
  (0.0086→0.0029) but the weak 728-BENIN drags BENIN (0.0479→0.0517) → net WASH (0.0273≈0.0283).
  Best LODO single config = 896+TENT. Ensemble = EGYPT-specialized add-on only.
- LEADERBOARD (2026-06-16): the ~20 teams at 0.0 are GONE; test set unchanged + our old scores
  unchanged → 0.0 teams were REMOVED (probing/cheating; code+report mandatory → unreproducible 0.0
  purged). Public LB is now MEANINGFUL — reverses the earlier "ignore public" stance. Our team
  best = prod896 0.07983 (rank ~8, made on user's OTHER PC: 896+FDA production). Top = 0.0116.
- STRATEGY: this session's 896 models are weak (public 0.21); prod896 (0.0798) is the real base.
  TENT/ensemble are METHOD findings to apply ON prod896. Winning path = prod896 base + TENT (+maybe
  rank-fuse prod896 with session models). Need prod896 checkpoint/recipe on this PC to integrate.

## HOST DIRECTIVE (2026-06-16) reframes strategy → E2-896 capture robustness
Host (discussion): LB reset 06-08 (dataset-ID + public/private split change, not metric bug);
invalid-submission cleanup pending → ranking provisional. PRIZE = private (2 UNSEEN doc types +
CAPTURED/PHYSICAL). ★Host #1: private "puts more focus on non-synthetic captured examples ...
print-and-capture ... real-world conditions that SUPPRESS fragile digital artifacts." → our
resolution/digital-trace reliance is a RISK; FantasyID ~chance was a REAL warning (not unfair
proxy). Public(seen) is PRIZE-IRRELEVANT. Private released few days before 7/14 → re-infer (TTA).
- 896+FDA base (this PC, prod896 replica): FantasyID AUC 0.513 (WORSE than base 896 0.561) → FDA
  helps seen/public but NOT capture. TTA on FantasyID also no help (cross-dataset gap).
- → E2-896 capture reserve: FREUID + FantasyID-physical(upweighted) + heavy print-capture aug
  (exp_e2_896_capture, val fantasyid_test 1,295 leakage-free). The one lever that moved capture
  (E2-518: 0.586→0.718). External data has LODO tension → CAPTURE RESERVE; primary unseen-type
  model stays FREUID-only 896+TENT. Strategy: 2 cards — unseen(896+TENT) + capture(E2) — fuse/select.

### E2-896 RESULT (capture breakthrough) + LODO-cost test launched (2026-06-16)
E2-896 (exp_e2_896_capture; FREUID + FantasyID-physical upweighted + heavy_recapture, 896px, 4ep)
validated on fantasyid_test (1,295, leakage-free capture proxy):
| model | ROC-AUC | APCER@1%BPCER | FREUID |
|---|---|---|---|
| base 896 (FREUID-only) | 0.5318 | 0.9753 | 0.9529 |
| E2-896 | 0.7838 | 0.8801 | 0.792 |
| E2-896 + TENT | 0.7975 | 0.7719 | 0.6453 |
→ FIRST genuine move on host's #1 axis (capture). base = chance; E2 lifts AUC 0.53→0.80.
  Confirms lever = REAL physical data + heavy capture aug (NOT FDA: 0.513).
- DECISIVE TEST (running): does E2 recipe COST unseen-type LODO? exp_e2lodo_egypt (GPU0, holdout
  EGYPT/DL, val=15867) + exp_e2lodo_benin (GPU1, holdout BENIN/DL, val=13369). Compare held-out
  log.csv FREUID vs base-896 LODO (EGYPT 0.0217, BENIN 0.0780). ≈base → E2 strictly better, ONE
  model both axes; ≫base → keep 2-card. Private = single set mixing unseen+captured → pivotal.

### E2-LODO RESULT (2026-06-16): capture recipe DESTROYS appearance-OOD generalization
Held-out FREUID per epoch (E2 recipe = FantasyID-physical upweighted + heavy_recapture):
| epoch | EGYPT (appearance-OOD) | BENIN (chroma) |
|---|---|---|
| 0 | 0.0406 (AUC 0.996) | 0.2549 (AUC 0.896) |
| 1 | 0.9587 (AUC 0.740) COLLAPSE | 0.1731 (AUC 0.954) improving |
| base-896 LODO (best) | 0.0217 | 0.0780 |
KEY: EGYPT held-out COLLAPSES at ep1 while train_loss DROPS (0.566->0.309) = domain-specialization
to capture, abandoning appearance-OOD. BENIN (chroma) TOLERATES capture aug + improves. => capture
recipe is INCOMPATIBLE with appearance-OOD unseen-type, COMPATIBLE with chroma unseen-type.
- best.pt saves on best HELD-OUT FREUID (train.py:222) -> EGYPT best.pt = ep0 (0.0406, near base).
- ★IMPLICATION for deployed E2 (exp_e2_896_capture, holdout=""): selected best by LEAKY random-val
  (monotone to ep3) -> shipped E2 = ep3 = MAX capture-specialization = likely appearance-OOD
  COLLAPSED. Its capture proxy 0.79 may have cost appearance-OOD unseen detection (random-val blind
  to this). => 2-card strategy CONFIRMED necessary; single E2 risky if private unseen = appearance-OOD.
- NEXT: measure EGYPT-LODO ep0 best.pt on fantasyid_test (capture proxy) -> does 1-epoch E2 ALREADY
  have capture robustness while RETAINING appearance-OOD (EGYPT 0.0406)? If yes, EARLY-STOPPED E2 =
  single model that does BOTH. (fantasyid_test is leakage-free vs the upweighted train split.)

### EARLY-STOP E2 = SINGLE MODEL FOR BOTH PRIZE AXES (2026-06-16, resolved)
Capture proxy of EGYPT-LODO ep0 ckpt (which RETAINS EGYPT appearance-OOD 0.0406) on fantasyid_test:
| model | capture AUC | capture FREUID | appearance-OOD (EGYPT LODO) |
|---|---|---|---|
| base 896 (FREUID-only) | 0.5318 | 0.9529 | 0.0217 |
| E2 ep0 (early) | 0.7559 | 0.6477 | 0.0406 RETAINED |
| E2 ep0 + TENT | 0.7501 | 0.6738 | (TENT ~neutral early) |
| E2 ep3 (deployed, over-trained) | 0.7838 | 0.792 | COLLAPSED (0.96@ep1) |
=> ~95% of capture robustness lands at EPOCH 0, BEFORE appearance-OOD erodes. EARLY-STOP E2 is a
SINGLE model serving BOTH prize axes (unseen-type + captured) — no 2-card ensemble needed. Deployed
exp_e2_896_capture (ep3, picked by leaky random-val) is OVER-trained → DO NOT ship.
- train.py: added save_every_epoch flag + epoch{N}.pt snapshots (config.py save_every_epoch).
- LAUNCHED exp_e2early_all (GPU0): all 5 types + FantasyID-physical + heavy aug, 3 ep, per-epoch
  ckpt. Pick deployment epoch by capture proxy (fantasyid_test) at the knee before appearance-OOD
  collapse (≈ep0–1). exp_e2lodo_benin still finishing on GPU1 (chroma unseen final number).

### e2early-ALL deployment candidate (2026-06-16)
BENIN-LODO final: ep0 0.2549 / ep1 0.1731(peak,best.pt) / ep2 0.3847(regress) — chroma unseen also
peaks EARLY (ep1) then erodes. Both folds confirm early-stop.
Capture proxy (fantasyid_test, leakage-free) by official FREUID (lower=better):
| model | AUC | APCER@1%BPCER | FREUID |
|---|---|---|---|
| base 896 FREUID-only | 0.5318 | 0.975 | 0.9529 |
| E2 ep3 (over-trained, deployed) | 0.7838 | 0.880 | 0.792 |
| EGYPT-LODO ep0 | 0.7559 | 0.770 | 0.6477 |
| e2early-ALL ep0 | 0.7591 | 0.677 | 0.5466 BEST |
=> all-types early-E2 ep0 BEATS over-trained ep3 on official capture metric (better APCER op point)
AND retains unseen-type (LODO). Strong SINGLE-model both-axes candidate. Getting ep1/ep2 to locate knee.

### DEPLOYMENT DECISION (2026-06-16): E2-early EPOCH 0 single model
Fusion experiments (base-LODO + E2-LODO, valid unseen-type via held-out folds; capture on
fantasyid_test):
| axis | base alone | E2 alone | base+E2 rank-fusion |
|---|---|---|---|
| EGYPT (app-OOD) | 0.0217 | 0.0406 | 0.0054 (fusion WINS 4x) |
| BENIN (chroma) | 0.0780 | 0.1731 | 0.0865 (~base) |
| CAPTURE | 0.9529 | 0.5466(ep0)/0.5082(ep1) | 0.7632(ep0)/0.7118(ep1) POISONED |
KEY: fusion is GREAT on unseen-type (EGYPT 0.0054!) but POISONS capture — the at-chance base member
halves E2's capture signal (0.547->0.763). Host says capture = #1 prize axis => fusion REJECTED.
Also: EGYPT fusion used E2@ep0, BENIN fusion used E2@ep1 (per-fold best.pt) — a SINGLE deploy ckpt
can't be both; ep1 collapses EGYPT (0.96) so deploy E2 member must be ep0.
DECISION MATRIX (single model):
| model | capture | EGYPT | BENIN | catastrophe? |
|---|---|---|---|---|
| base-all | 0.95 | 0.022 | 0.078 | CAPTURE (0.95) |
| E2-early ep0 | 0.547 | 0.041 | 0.255 | none |
| E2-early ep1 | 0.508 | 0.96 | 0.173 | APP-OOD (0.96) |
=> DEPLOY E2-early EP0 (checkpoints/exp_e2early_all/epoch0.pt): capture-strong (host #1) + NO
catastrophic failure mode = minimax-safe under unknown private unseen-type nature. Weak point chroma
0.255 (mild). TENT ~neutral early. Fusion's EGYPT win seductive but capture cost disqualifies it.
FUTURE (if needed): E2-weighted fusion (preserve capture, add unseen complementarity) untested.

### e2early ep2 (2026-06-16): capture MONOTONE-improves, anti-correlated with unseen-type
ep2 capture proxy: AUC 0.8582 APCER@1% 0.5121 FREUID 0.3779. Full knee:
| ep | capture FREUID | EGYPT app-OOD | BENIN chroma |
| ep0 | 0.5466 | 0.041 RETAINED | 0.255 |
| ep1 | 0.5082 | 0.96 collapsed | 0.173 peak |
| ep2 | 0.3779 BEST | collapsed | regressed |
=> capture & unseen-type ANTI-correlated monotonically. No single epoch/fusion gets both (collapsed
member poisons its weak axis). FORK: ep0 (minimax-safe, capture 0.547, no catastrophe) vs ep2
(capture-max 0.378, unseen app-OOD catastrophe). Depends on private capture-weight (unknown).
Surfacing to user for risk-posture decision. All 3 epoch ckpts saved (exp_e2early_all/epoch{0,1,2}.pt).

### Deployment CSVs prepared (2026-06-16) — user choice: "prepare all 3, decide at private release"
submission_e2early_ep0.csv / _ep1.csv / _ep2.csv from exp_e2early_all/epoch{0,1,2}.pt.
Each = 142,818 rows (7,821 published scored + 134,997 unpublished filled 0.5). NOT submitted (hold
per user policy). At private release: user visually assesses the 2 unseen doc-types + capture style,
picks the matching epoch (ep0 app-OOD-safe / ep1 chroma+capture / ep2 capture-max), RE-INFER full set.
NOTE: public best remains prod896 0.07983 (OTHER PC); E2-early targets PRIVATE (capture+unseen), not
public — expected worse on born-digital public. No local model beats 0.07983 on public.

## DEEP-RESEARCH SYNTHESIS + 10× PLAN (2026-06-16) — 4-agent SOTA sweep
Four parallel research agents (generalizable-AIGC / TTA+DG / forensic-preproc / backbones+open-set)
CONVERGED on a consistent program. Cross-cutting consensus:
1. ★DON'T ERODE THE BACKBONE (top lever, 3/4 agents). Unfreezing DINOv2 destroys pretrained OOD
   generalization → unseen-type collapse. LoRA > full/partial unfreeze (DINOv3-forensics: LoRA F1
   0.774 vs full-FT 0.530 "unstable"). Our unfreeze_blocks:2 is SUBOPTIMAL. Fix: frozen+LoRA / L2-SP
   anchor / discriminative-LR. EXPLAINS our E2-LODO EGYPT collapse (ep1) = backbone erosion.
2. Semantic features SURVIVE recapture; frequency/forensic-trace DIE (FantasyID: SOTA forensic
   detectors ~50% FNR on recaptured IDs; RRDataset: specialized <1% recall, DINOv2/CLIP ~72%).
   Host "private suppresses fragile artifacts" EXTERNALLY CONFIRMED. Our DTC high-pass stream is
   FRAGILE → down-weight/re-target for private.
3. The surviving forensic cue = CHROMATICITY. CMA (Chromaticity Map Adapter, CVPR'24, public repo)
   independently corroborates our BENIN=chroma finding; robust to JPEG QF70 + downscale.
4. SAR > TENT (ViT+LN-native, anti-collapse) + TRANSDUCTIVE QUANTILE score-calibration → directly
   targets rank/APCER@1% metric, ~0 risk.
5. Already have: reg4 register tokens ✓, rank-averaging ✓, capture aug (E2) ✓, 896px ✓.
GEN EXPERIMENT PLAN (staged, validate each on LODO EGYPT/BENIN + capture proxy + APCER@1%):
- GEN-1 (RUNNING): anti-erosion. 1a=frozen backbone, 1b=L2-SP anchor(λ1000, 14.2M params), E2 recipe,
  EGYPT-hold, per-epoch ckpt. Q: does preserving backbone STOP the ep1 EGYPT collapse → capture+unseen
  in ONE model? vs un-anchored E2-LODO (0.0406→0.96).
- GEN-2: manual LoRA on DINOv2 qkv (peft absent) — headline lever, if GEN-1 confirms anti-erosion.
- GEN-3: CMA chromaticity stream (targets BENIN/chroma survivor cue; replace fragile high-pass).
- GEN-4: SAR replace TENT + transductive quantile calibration (APCER@1% metric).
- GEN-5: moire-spectral aug (FMAG, ~free capture aug).
- GEN-6: Outlier Exposure / energy loss vs diverse attack zoo (unseen attack types).
- GEN-7: SWAD weight-avg + WiSE-FT/soup (0-cost base + ensemble robustness, rank-mean combine).
Code added: anchor_lambda (L2-SP) in config+train.py; save_every_epoch (earlier).

### GEN-1 CONCLUSION (NEGATIVE) + GEN-2 LoRA launched (2026-06-16)
Anti-erosion via frozen / L2-SP, 3-axis (vs unfreeze-2 E2-early ep0: EGYPT 0.041 / capture 0.547):
| variant | EGYPT (unseen) | capture proxy |
| GEN-1a frozen (unfreeze 0) | ep3 0.370 (stable, NO collapse) but UNDERFIT | 0.850 underfit |
| GEN-1b L2-SP anchor λ1000 | 0.86–0.96 all eps (WORSE than baseline) | 0.674 |
=> NEGATIVE: backbone needs trainable CAPACITY (frozen underfits; my mean-norm L2-SP misbehaves).
Confirms WHY agents specified LoRA (capacity added, base preserved) not freezing.
- GEN-2 LoRA (RUNNING): manual LoRALinear on attn qkv/proj all 12 blocks (peft absent); freeze
  backbone + only LoRA(1.4M, 1.6%)+head+norm train. Smoke OK (B-grad flows@step0, A engages@step1).
  r=16 α=32 lr=3e-4, E2 recipe, EGYPT(GPU0)+BENIN(GPU1) holdout, per-epoch ckpt. Q: does LoRA keep
  EGYPT low ACROSS epochs (vs unfreeze-2 collapse 0.041→0.96) + gain capture → ONE model both axes?
Code: LoRALinear+inject_lora in dtc.py; lora_rank/lora_alpha threaded config→train→classifier→infer→xeval.

### GEN-2 LoRA lr=3e-4 = FAILED (lr confound), relaunch low-lr bracket (2026-06-16)
LoRA r16 lr3e-4 EGYPT ep0 = 0.9277 (AUC 0.784) — WORSE than frozen (0.41) and unfreeze-2 ep0
(0.0406). lr=3e-4 (3× unfreeze-2's 1e-4) over-adapted to capture data by ep0, erasing appearance-OOD.
LoRA preservation lives at LOW lr → relaunch bracket lr=1e-4 (GPU0) + lr=5e-5 (GPU1), r16, EGYPT,
to test: can LoRA reach ~0.04 (unfreeze-2-ep0) AND stay stable (avoid the ep1 cliff)? If even
lr=5e-5 fails → backbone-preservation does NOT fix the capture↔appearance-OOD tension → move to GEN-3/4.

### GEN-2 LoRA = REJECTED (deep-research lever #1 fails for OUR task) (2026-06-16)
LoRA r16 EGYPT-LODO ep0 across lr sweep (vs unfreeze-2 E2 ep0 0.0406/AUC0.996, frozen 0.41/AUC0.82):
| lr | EGYPT FREUID | AUC |
| 3e-4 | 0.928 | 0.784 |
| 1e-4 | 0.979 | 0.538 |
| 5e-5 | 0.977 | 0.577 |
ALL far worse than unfreeze-2 AND frozen; LOWER lr = WORSE (underfit→chance), so not a tuning miss.
MECHANISM (structural): frozen backbone forces the small adapters+DTC-head to learn SEEN-type-specific
forgery cues that DON'T transfer to the unseen type; full unfreeze-2 adapts rich backbone features
that DO generalize. The research LoRA>unfreeze finding (born-digital AIGC) does NOT hold for
capture-augmented appearance-OOD ID docs.
=> CONCLUSION across GEN-1+GEN-2: backbone-PRESERVATION (frozen/L2-SP/LoRA) is a DEAD END for FREUID
unseen-type. Our unfreeze-2 + E2-early-ep0 recipe is well-suited and stays the deployment. The
capture↔appearance-OOD tension is a DATA-distribution effect (capture aug pulls from clean
appearance-OOD), not fixable by adaptation method. Best unseen result remains base+E2 rank-fusion
(EGYPT 0.0054). NEXT: orthogonal levers — GEN-3 CMA chroma (targets BENIN) / GEN-4 SAR-TTA. NOTE:
transductive quantile/rank score-calib is a NO-OP for single-model rank-based FREUID (only helps
ensembling, already done via rank-mean).

### GEN-3 CMA chromaticity stream launched (2026-06-16, user-chosen lever)
Implemented CMA: ChromaResidual(cma=True) → 6ch (opponent chroma + intensity-normalized chromaticity),
palette-removed (per-image spatial mean), NO high-pass (research: chroma SURVIVES recapture, high-pass
DIES). TraceEncoder(in_ch=6). Smoke OK (6ch flows, grads flow). Threaded cma_chroma config→train→
classifier→infer→xeval. Code reuses existing use_chroma fusion plumbing.
- GEN-3 BENIN (GPU0) + EGYPT (GPU1): FREUID-only + unfreeze-2 + use_chroma + cma_chroma, 896, b16, 5ep,
  per-epoch ckpt. Compare to base-LODO (BENIN 0.0780, EGYPT 0.0217) WITHOUT chroma. Q: does CMA push
  BENIN below 0.078 (chroma win) WITHOUT hurting EGYPT? Clean delta = chroma stream effect.

### GEN-3 CMA RESULT (2026-06-16): best-ever EGYPT, but antagonistic on BENIN
CMA chroma stream (palette-removed chromaticity, no high-pass) per-epoch LODO FREUID:
| ep | EGYPT (CMA) | BENIN (CMA) |
| 0 | 0.1617 | 0.1400 |
| 1 | 0.0038 (AUC 0.9995) BEST-EVER | 0.1305 (best) |
| 2 | 0.0110 | 0.1921 (regress) |
vs base-LODO: EGYPT 0.0217, BENIN 0.0780.
=> CMA HELPS EGYPT massively (0.0217→0.0038, beats even base+E2 fusion 0.0054) but HURTS BENIN
(0.078→0.130). Mechanism: per-image palette-removal makes model palette-INVARIANT (great for EGYPT
appearance-OOD) but STRIPS the chroma signal BENIN DEPENDS on (chroma-dependent fold). Same
antagonistic-fold pattern as appearance-invariance (D10). CMA = an EGYPT/appearance-OOD lever, NOT
a BENIN fix. best CMA ckpt: EGYPT epoch1.pt (0.0038).
- NEXT: base(BENIN-strong 0.078) + CMA(EGYPT-strong 0.0038) rank-fusion → capture BOTH? (analogous to
  base+E2 fusion but CMA is a better EGYPT member). Eval after CMA finishes + CMA capture proxy.

### base+CMA FUSION = BEST-EVER unseen-type (2026-06-16)
base-LODO + CMA-LODO rank-mean fusion (both held out the eval fold; valid unseen measure):
| fold | base | CMA | base+CMA rank-mean | base+CMA mean-prob |
| EGYPT | 0.0217 | 0.0038 | 0.0030 | 0.0024 |
| BENIN | 0.0780 | 0.1305 | 0.0823 | 0.1272 |
RANK-MEAN: EGYPT 0.0030 (7x better than base!) + BENIN 0.0823 (~base, tiny loss) => MEAN 0.0427 =
BEST unseen-type EVER (base 0.0499, base+E2 0.0460). Rank-mean >> mean-prob on BENIN (CMA's weak
member drags mean-prob but rank-mean is scale-robust). CMA validated as a FUSION member: contributes
huge EGYPT, base holds BENIN. NEXT: CMA capture proxy (is the fusion capture-poisoned like base+E2?);
if capture OK → train all-types CMA + fuse with base-all for deployment.

### GEN-3 CONCLUSION = WIN on unseen-type (2026-06-16)
CMA capture proxy (fantasyid_test): AUC 0.571 / FREUID 0.972 = AT CHANCE (CMA trained FREUID-only,
no capture data → unseen-type lever ONLY, not capture). So base+CMA fusion = best unseen-type card
(EGYPT 0.0030, BENIN 0.0823, mean 0.0427) but NO capture robustness. Two-axis split persists:
- UNSEEN-TYPE card: base+CMA rank-fusion (mean 0.0427, best ever).
- CAPTURE card: E2-early-ep0 (capture 0.547).
Deploy needs all-types versions to fuse. base-all=deploy896_all exists. Training all-types CMA (early
epoch, LODO showed CMA peaks ep1) → fuse with base-all = deployable unseen-type card.

### GEN-3 DEPLOYMENT card generated (2026-06-16)
all-types CMA (exp_gen3_cma_all) trained; used epoch1.pt (LODO showed CMA unseen-type peaks ep1).
Generated submission_baseCMA_unseen.csv = base-all(deploy896_all) + CMA-all-ep1 RANK-MEAN fusion
(scripts/infer_rankfuse.py — rank-mean not mean-prob, since LODO showed rank-mean >> mean-prob for
base+CMA). 142,818 rows (7,821 rank-fused + 134,997 fill 0.5). Validated unseen-type (LODO): EGYPT
0.0030, BENIN 0.0823, mean 0.0427. = deployable UNSEEN-TYPE card.
DEPLOYMENT CARD SET now:
- UNSEEN-TYPE: submission_baseCMA_unseen.csv (mean 0.0427; capture=chance).
- CAPTURE: submission_e2early_ep0/1/2.csv (capture 0.547; unseen weaker).
At private release: pick by revealed composition (unseen-heavy→baseCMA, capture-heavy→e2early) or fuse.

### GEN-4 SAR deferred + PIVOT to PUBLIC thrust (2026-06-16)
User added target: PUBLIC < 0.05 (believes needed to win). SAR eval (sar_eval.py: no-TTA/TENT/SAR,
SAM 2x compute) too slow at batch 8 (>35min/fold, blocking) → killed, deferred (TENT already
validated +43% LODO; SAR is incremental). Re-run later with larger eval batch if a gap.
PUBLIC LEVER = FDA (validated: FDA prod896 0.0798 vs no-FDA 0.2159). Plan: resolution+seed-diverse
FDA ensemble. Members: exp_prod896_fda_all/best.pt (896,s42,ep3 — have it) + exp_fda728_all (728,s42)
+ exp_fda896_s43 (896,s43). Rank-mean ensemble = public-squeezer. CANNOT measure public locally
(no labels) → optimize FDA+ensemble blind, user submits to verify. Random-val = in-distribution proxy.

## ★ ViT-LARGE BREAKTHROUGH (other-PC transfer, 2026-06-16) — supersedes DTC/CMA line
best_model_transfer.zip from user's other PC → _transfer/ (their code+ckpts; data paths match, runs
here). Recipe = vit_large_patch14_reg4_dinov2 + last-2-unfreeze + FDA(0.05) + 896 + tail_margin loss
+ bf16 + EMA + eff_bs32 + lr5e-5 (model_type=rgb). VERIFIED LODO here (reproduces): EGYPT 0.00354,
BENIN 0.01617, 2-fold mean 0.0099 (vs my base+CMA 0.0427 = 4.3x better; BENIN 5x better — our bottleneck!).
LAUNCHED (both ~99% GPU, no I/O starve, shared freuid cache):
- exp_vitL_prod (GPU0): ViT-Large production all-types (FDA, deployment unseen-type+public card).
- exp_vitL_capture (GPU1): ViT-Large + recap aug + FantasyID (FDA off), per-epoch save → capture card
  (their "needs captured val data" gap filled by my fantasyid_test proxy). 2-card deploy, SUPERIOR backbone.
Public: ViT-Large+FDA should beat ViT-B prod896 0.0798 → toward user's <0.05 target.
Added per-epoch save to _transfer/src/freuid/train.py (capture epoch picked post-hoc by fantasyid_test).

### ViT-L capture (recap aug) = does NOT beat my ViT-B E2 on capture FREUID (2026-06-17)
ViT-L + recap aug + FantasyID, fantasyid_test capture proxy (capture peaks ep0, degrades after):
| ep | capture FREUID | AUC |
| 0 | 0.6596 | 0.7927 |
| 1 | 0.9283 | 0.778 |
| 2 | 0.9643 | 0.696 |
vs my ViT-B E2-early ep0: FREUID 0.5466 / AUC 0.7591. ViT-L has BETTER capture AUC (0.79>0.76) but
WORSE FREUID (0.66>0.55) — worse APCER@1% operating point (their recap aug ≠ my tuned heavy_recapture).
=> CAPTURE CARD stays ViT-B E2-early-ep0. ViT-Large win is on UNSEEN-TYPE (production). Future: try
ViT-Large + MY heavy_recapture aug (port to their code) for a better capture card.

### ViT-Large DEPLOYMENT state (2026-06-17 ~02:45)
exp_prod896_large (ViT-Large production, all-types, FDA, tail_margin) trained to ep4 (in-dist val:
ep0 0.0024→ep1 0.0011→ep2 0.00066→ep3 0.00063(best)→ep4 0.00065). best.pt=ep3. snapshot_ep2 preserved.
Generated submission_vitL_prod.csv (from best.pt ep3) = 142,818 rows (7820 scored + fill) — PUBLIC +
unseen deployment card (user submits to verify <0.05; ViT-L+FDA should beat ViT-B prod896 0.0798).
DEPLOYMENT CARDS (ViT-Large era):
- PUBLIC + UNSEEN-TYPE: submission_vitL_prod.csv (ViT-Large; LODO-verified EGYPT 0.0035/BENIN 0.0162).
- CAPTURE: submission_e2early_ep0.csv (ViT-B E2-early 0.5466 — still beats ViT-L recap on capture FREUID).
Note: ViT-Large supersedes base+CMA/DTC for unseen-type (mean 0.0099 vs 0.0427).

### ★ ViT-Large + TENT STACK (2026-06-17) — compounding win
TENT (my validated LayerNorm-entropy TTA lever) stacks on the ViT-Large LODO folds:
| fold | ViT-L no-TTA | ViT-L + TENT |
| BENIN | 0.0165 | 0.0061 (-63%) |
| EGYPT | 0.0035 | (testing) |
=> their breakthrough backbone (ViT-Large+FDA+tail_margin) COMPOSES with my TTA lever (TENT). Deploy
= ViT-Large + TENT at test time (transductive on private set). 2-fold mean projecting ~0.004 (beats
their projected 0.006). scripts/tent_vitL.py (TENT adapted to transfer model_type=rgb).

### ViT-Large + TENT — COMPLETE 2-fold (2026-06-17): net-positive, antagonistic
| fold | ViT-L no-TTA | ViT-L + TENT |
| EGYPT | 0.0035 | 0.0045 (TENT HURTS — already near-perfect AUC 0.9987, over-adapts) |
| BENIN | 0.0165 | 0.0061 (TENT HELPS -63%) |
| mean | 0.0100 | 0.0053 |
=> TENT helps HARD fold (BENIN/chroma), slightly hurts EASY fold (EGYPT, already saturated). NET WIN
on mean (0.0100->0.0053). Private unseen types are HARD by design → apply TENT at deploy. Best
unseen-type recipe = ViT-Large + TENT (transductive). Production exp_prod896_large DONE (best.pt ep3).

### 5-fold CV in progress (2026-06-17): GUINEA operating-point instability
GUINEA fold: ep0 0.0216 (AUC 0.9946) → ep1 0.6585 (AUC 0.974 but APCER@1% 0.793 — OPERATING-POINT
COLLAPSE; AuDET fine, tail blew up). best.pt=ep0. EGYPT/BENIN did NOT collapse (improved ep0→ep1);
production in-dist APCER@1% stable. => fold-specific tail_margin operating-point variance, NOT a
deploy blocker (production stable). 5-fold so far: EGYPT 0.0035, BENIN 0.0165, GUINEA 0.0216(ep0,
undertrained by collapse). MAURITIUS running, MOZAMBIQUE next.

### 5-fold CV — operating-point instability is GENERAL (2026-06-17)
ALL "easy" folds collapse AFTER their peak (tail_margin operating-point instability; AuDET/AUC stay
high, APCER@1% blows up): GUINEA peak ep0 0.0216→0.66→0.85; MAURITIUS peak ep0 0.0052→0.18→0.57;
MOZAMBIQUE ep0 0.0884. EGYPT/BENIN peaked ep1 (stable longer). early_stop+best.pt CAPTURE the peak,
so deployable, but collapse limits convergence (GUINEA/MOZAMBIQUE undertrained). 5-fold best.pt CV:
EGYPT 0.0035, BENIN 0.0165, GUINEA 0.0216, MAURITIUS 0.0052, MOZAMBIQUE ~0.088(ep0). NOTE: PRODUCTION
(all-types, no holdout) is STABLE (best.pt ep3) — collapse is a HELD-OUT training-dynamics artifact,
NOT a deploy issue. FOLLOW-UP: test TENT on unstable folds (may recover operating point at test time).

### TENT recovers unstable folds too (2026-06-17)
TENT on ViT-L per fold: BENIN 0.0165->0.0061(-63%), GUINEA 0.0216->0.0159(-26%), EGYPT 0.0035->0.0045
(slight up, already saturated AUC 0.9987). => TENT is a ROBUST general improver across folds (helps
hard + unstable, only slightly hurts the already-saturated EGYPT). Strengthens ViT-Large+TENT deploy.

### ★ ViT-Large FULL 5-fold LODO-CV (2026-06-17) — honest complete number
best.pt (no-TTA) per fold: EGYPT 0.0035, BENIN 0.0165, GUINEA 0.0216, MAURITIUS 0.0052,
MOZAMBIQUE 0.0602 (ep2 peak then degrade). 5-FOLD MEAN = 0.0214 (no-TTA). With TENT (helps all but
saturated EGYPT): ~0.014. Hard folds EGYPT/BENIN excellent; GUINEA/MOZAMBIQUE harder for this recipe
(MOZAMBIQUE 0.060 the outlier). HONEST: their projected 0.006 was optimistic (2-hard-fold extrapolation);
true 5-fold ~0.021 (no-TTA)/~0.014 (TENT). Still >> my prior 2-fold 0.0427. Private unseen could be
MOZAMBIQUE-like (0.06) worst case. seed-43 ViT-L production training (ensemble robustness option).

### warm-cap (snapshot_ep2 warm-start + capture fine-tune) = REJECTED (2026-06-17)
3-axis (Run B EGYPT-LODO, Run A capture fantasyid_test):
| axis | warm-cap ep0 | ep1 | best card |
| unseen EGYPT | 0.168 | 0.945(collapse) | base ViT-L 0.0035 |
| capture fantasyid | 0.784 | 0.956 | E2-early 0.5466 |
=> LOSE-LOSE: capture fine-tune from digital base ERODES unseen (0.0035→0.168→collapse) AND doesn't
match dedicated capture (0.784>0.547). Warm-start does NOT solve digital↔captured tension. 2-CARD
registry STAYS (card#1 snapshot_ep2+TENT unseen+public, card#2 E2-early capture). NEXT: try ViT-Large
+ MY heavy_recapture (from-scratch, my code) to beat E2's capture card (their recap aug < my heavy_recap).

### warm-cap final (2026-06-17): TENT-EGYPT confirmatory killed; verdict stands
warm-cap REJECTED on capture axis (0.784>E2 0.547) regardless of TENT-EGYPT recovery — killed the
slow confirmatory eval. 2-card registry unchanged. ViT-L E2 capture (my heavy_recapture, ViT-Large)
training to try to beat E2 capture 0.5466.

### ViT-L E2 capture = does NOT beat ViT-B E2 on capture (2026-06-17)
ViT-Large + my heavy_recapture, fantasyid_test: ep0 FREUID 0.6825 (AUC 0.7769, APCER@1% 0.8005) vs
ViT-B E2-early 0.5466 (AUC 0.7591, APCER@1% 0.677). CONSISTENT pattern across ALL ViT-Large capture
attempts (recap 0.66, warm-cap 0.784, E2 0.6825): ViT-Large = better capture AUC, WORSE APCER@1%
operating point → worse FREUID. The strict 1%-BPCER op-point is less calibrated on the larger backbone
for captured/OOD. => CLEAN finding: ViT-Large wins UNSEEN-type, ViT-B wins CAPTURE — prize axes prefer
DIFFERENT backbones. Capture card STAYS ViT-B E2-early. Registry unchanged.

### CAPTURE backbone SETTLED (2026-06-17): ViT-B > ViT-Large
ViT-L E2 capture ep0 0.6825 / ep1 0.7115 (fantasyid) — both > ViT-B E2-early 0.5466. ALL 4 ViT-Large
capture attempts lose (recap 0.66, warm-cap 0.784, E2 0.68/0.71). CLEAN: unseen-type→ViT-Large,
capture→ViT-B (smaller backbone better-calibrated at strict 1%-BPCER op-point on OOD/captured). 2-card
is REQUIRED (different backbones per axis), not a compromise. Capture card = ViT-B E2-early (registry
unchanged). seed-43 ViT-L prod training for unseen-card ENSEMBLE robustness.

### ★ LODO ENSEMBLE validation (2026-06-17): robustness CONFIRMED
2-seed ViT-Large fold ensemble (logit-mean), held-out fold:
| fold | s42 | s43 | ENSEMBLE |
| EGYPT | 0.0035 | 0.0025 | 0.0021 (beats both — low variance) |
| BENIN | 0.0165 | 0.0506 | 0.0268 (between — HIGH 3x seed-variance) |
=> CONFIRMS "ensembling = ROBUSTNESS not mean-gain". EGYPT (low var) ensemble improves; BENIN (3x var)
ensemble reduces variance (robust ~0.027 vs gamble 0.017/0.051). Private unseen types unknown + hard =
high seed-variance → ENSEMBLE hedges unlucky-seed risk. DEPLOY multi-seed production ensemble for
robust private inference. Rebuilding production seeds (s43) for the deployment ensemble.

## SELF-BLEND for UNSEEN-TYPE (2026-06-17) — new plan from forgery research + failure analysis
4-agent forgery research synthesis. FAILURE ANALYSIS: external-data union (idnet/docxpand/bid, 22
doc-types) = DEAD at 896 (other PC) — external ATTACKS carry generator fingerprints → model learns
those not generalizable cues (IDNet's own finding: 98.65%→50.55% cross-state shortcut). Capture 0.55
= fundamental DG (research-confirmed, recapture device-specific). NEW LEVER (untested, strong evidence):
SELF-BLEND synthetic forgery (SBI/NSA/SelfMAD) — generate "fake" from GENUINE via self-blending →
DOC-AGNOSTIC forgery cue (local inconsistency) → UNSEEN-type generalization (SelfMAD: 0 real forgeries
→ cross-morph EER 5.63%, unseen-strong). Sidesteps the generator-fingerprint trap.
IMPL: _transfer/src/freuid/data/self_blend.py (photometric+blend inconsistency on field-sized regions,
mimics field-alter/photo-sub/splice doc-agnostically); ManifestDataset self_blend_p (genuine 0→1).
LAUNCHED: exp_sb_benin (BENIN-held, bottleneck) + exp_sb_egypt (EGYPT-held, regression), ViT-Large+FDA+
tail_margin + self_blend_p 0.3. Q: does self-blend improve BENIN-LODO (base 0.0165/0.0506 seed-var) +
not hurt EGYPT (0.0035)? DATA: docxpand/midv/bid GENUINE = future self-blend sources (type diversity);
external ATTACKS (idnet) excluded. midv_holo = capture axis (separate).

### SELF-BLEND p=0.3 = REJECTED (EGYPT collapse) (2026-06-17)
LODO ep1: BENIN 0.0268 (modest, ≈ensemble), EGYPT 0.8668 (COLLAPSE, base 0.0035; AUC 0.999→0.927,
monotonic: ep0 0.061→ep1 0.867). ROOT CAUSE: photometric self-blend teaches "local appearance/color
inconsistency = forgery" → false-flags appearance-OOD genuine EGYPT → op-point collapse. SAME antagonism
as CMA/appearance-inv (appearance-OOD fragile to appearance-sensitizing training). NET LOSS. NOT expanded
to 5-fold. NEXT: test p=0.1 (gentle) — if EGYPT still degrades, self-blend fundamentally rejected.

### SELF-BLEND FULLY REJECTED (2026-06-17): appearance-OOD antagonism is fundamental
p=0.1 ep1: BENIN 0.0319 (worse than base 0.0165 AND p=0.3 0.0268), EGYPT 0.7633 (STILL collapsed, base
0.0035). Both p={0.3,0.1} collapse EGYPT; never beats base BENIN. ROOT: photometric self-blend →
"appearance inconsistency=forgery" → false-flags appearance-OOD genuine EGYPT. Lowering p doesn't fix
(fundamental, same as CMA/appearance-inv). => any appearance-sensitizing forgery synthesis is
INCOMPATIBLE with appearance-OOD. UNSEEN-TYPE is at practical ceiling with aug methods; deployment
stays ViT-L+FDA+tail_margin+TENT+multi-seed ensemble (mean ~0.014). NEXT high-value = CAPTURE axis
(research levers: FHAG freq-band aug, focal+temperature calib, CDC) — these target capture/op-point,
NOT appearance, so won't hit the EGYPT antagonism. Self-blend code retained (self_blend_p) but UNUSED.

### GENUINE-DIVERSITY (extgen) LODO ep1 (2026-06-17): EGYPT↔BENIN antagonism (mirror of self-blend)
extgen = FREUID + docxpand/bid GENUINE (16k, no external attacks), ViT-L+FDA+tail_margin.
EGYPT 0.00275 (base 0.0035 → IMPROVED, best-ever, NO collapse) | BENIN 0.0734 (base 0.0165 → HURT).
=> genuine-diversity is the appearance-OOD lever (opposite of self-blend which collapsed EGYPT).
EGYPT↔BENIN antagonism confirmed AGAIN. Private unseen = new templates = likely appearance-OOD
(EGYPT-like) → genuine-diversity may be the right bet. EXPANDING to GUINEA/MAURITIUS/MOZAMBIQUE for
net 5-fold (vs base 5-fold 0.0214).

### GENUINE-DIVERSITY (extgen) COMPLETE 5-fold (2026-06-18): NET-WORSE, but the appearance-OOD lever
Full LODO (peak ep, ViT-L+FDA+tail_margin, +16k docxpand/bid GENUINE, no ext attacks):
  fold        base     extgen    verdict
  EGYPT       0.0035   0.00275   better (best-ever, appearance-OOD win)
  BENIN       0.0165   0.0734    4.4x WORSE (chroma tail destroyed)
  GUINEA      0.0216   0.0236    ~neutral (slightly worse)
  MAURITIUS   0.0052   0.000405  13x better (appearance-OOD win, huge)
  MOZAMBIQUE  0.0602   0.0898    1.5x WORSE (extgen plateaued ep1 0.0914->ep2 0.0898; base peak ep2 0.0602)
  MEAN        0.0214   0.0380    1.77x WORSE on FREUID 5-fold
PATTERN (crisp): extgen helps APPEARANCE-OOD folds (EGYPT, MAURITIUS) and hurts CHROMA/HARD folds
(BENIN, MOZAMBIQUE); GUINEA neutral. Mirror image of self-blend (which collapsed EGYPT). Fusion
(base+extgen logit-mean) does NOT salvage BENIN: EGYPT-ens 0.0028, BENIN-ens 0.0238 (vs base 0.0165).
VERDICT per "효과 있으면 확장" criterion: NOT net-effective => DO NOT integrate into deployment.
Deployment 2-card UNCHANGED (card#1 ViT-L+TENT+multi-seed ensemble for unseen; card#2 E2 for capture).
STRATEGIC NUANCE: private unseen types = new templates = likely appearance-OOD (EGYPT/MAURITIUS-like).
On the ACTUAL private, genuine-diversity could win despite worse FREUID 5-fold — a judgment bet
(appearance-OOD genuine-diversity vs balanced/robust ViT-L baseline). Extgen fold checkpoints retained
as evidence; not promoted. Moz run stopped at ep2 (verdict conclusive, freed GPU0).

### E3 CAPTURE-axis upgrade — LAUNCHED (2026-06-18): FHAG + focal + CDC (capture card, ViT-B DTC)
After extgen (unseen-axis) ended net-worse, pivot to the CAPTURE axis (private physical/recapture).
Three NEW levers, all targeting recapture TEXTURE / op-point (NOT appearance → avoid EGYPT antagonism):
  (1) FHAG (SpectralBandPerturb): perturb radial AMPLITUDE-spectrum bands (phase kept) so trace
      branch is invariant to born-digital↔capture band-energy shift. forensic_aug.py, fhag flag.
  (2) focal BCE (γ=2,α=0.5) on FRAUD head: sharpen hard genuine/attack tail (APCER@1%). train.py.
  (3) CDC (Central Difference Conv, θ=0.7, CDCN): gradient-texture op in TraceEncoder. dtc.py,
      cdc_theta flag. state_dict guard: eval MUST reconstruct cdc_theta (key-match) — patched
      xeval_fantasyid + infer.py.
Base = E2-early recipe (freuid_fid_upweighted, 896, heavy_recapture, save_every_epoch). Objective:
min fantasyid_test FREUID (capture proxy), beat E2-early 0.5466. Runs: exp_e3_capture (FULL, GPU0)
+ exp_e3_cdc_only (ablation, GPU1). All 3 levers smoke-tested (FHAG perturbs, focal<bce, CDC stride
1/2 aligns, DTC+CDC fwd, state_dict roundtrip 0/0). Eval monitor scripts/e3_capture_eval.sh pending.

### E3 CAPTURE levers — RESULT (2026-06-18): REJECTED, E2-early 0.5466 stands
Capture proxy (fantasyid_test FREUID, lower=better; pick best epoch, NOT val-FREUID best):
  E2-early baseline ...... 0.5466
  E3-cdc-only ........... ep0 0.5800 (best); ep1 0.6531, ep2 0.6545  -> WORSE by 0.033
  E3-full (FHAG+focal+CDC) ep0 0.7863; ep1 0.6608, ep2 0.6514 (best) -> WORSE by 0.105
NONE beat baseline. Mechanistic failure analysis:
  - focal HURTS: capture hard tail = clean-GenAI ≈ chance (fundamentally unlearnable, FantasyID
    finding). Focal down-weights easy & chases the unlearnable tail → wastes capacity on noise →
    op-point degrades. (full ep0 0.786 vs cdc-only ep0 0.58: adding focal+FHAG to CDC ~doubled
    proxy FREUID at ep0.)
  - FHAG erases the cue: recapture artifacts ARE band-energy; random per-band amplitude rescale
    destroys the captured-genuine↔digital-attack band signal (same "destroy the signal" mode as
    appearance-inv on BENIN chroma).
  - CDC redundant: HighPassResidual front-end already extracts gradient texture; CDC adds little
    (0.5466→0.58 slightly worse).
  - early-stop pattern HOLDS: cdc-only capture-best=ep0; later epochs raise AUC (0.78→0.81) but
    worsen APCER@1% op-point (0.713→0.781) — over-train away op-point, exactly E2-early.
=> Nth confirmation the CAPTURE axis is at a PRACTICAL CEILING (clean-GenAI fundamental limit).
Capture card UNCHANGED (E2-early ep0). Levers kept behind flags (fhag/focal_gamma/cdc_theta),
OFF by default, for paper ablation. NEXT cheap in-scope test: TENT/TTA on the E2-early card.

### PUBLIC-SCORE GAP investigation + BCE ViT-L bet — LAUNCHED (2026-06-18)
Goal (user): public < 0.05 (current best 0.0935 ens / 0.109 single ViT-L tail). The 0.043 public
(submission_large.csv, 06-16 23:05) was a CSV TRANSFERRED from other-PC — NO local infer log, and its
checkpoint is ABSENT from the _transfer bundle (didn't finish before bundling). Investigation:
  - My exp_prod896_large config is BYTE-FOR-BYTE identical to their exp_prod896_large (23 params). Their
    infer is plain single-pass sigmoid (no TTA/post-proc), EMA weights. So recipe/infer NOT the gap.
  - COVERAGE ruled out: complete 17GB official zip has exactly 7821 public_test imgs = the FULL released
    public set (sample_submission 142818 ids = 7821 public + 134997 private-hidden, filled 0.5, don't
    affect public LB). My subs cover public fully → 0.109 is a REAL score.
  - ROOT CAUSE = LOSS. Their docs: bce-896 public 0.0798 vs tail-896 0.1302. tail_margin reshapes
    scores for the unseen TAIL → helps private/unseen, HURTS the saturated born-digital PUBLIC slice.
    Their explicit warning: "picking submission by public LB selects the WRONG model for the private
    prize." My ViT-L TAIL repro = 0.109 (consistent w/ their tail 0.13). The 0.043 ⇒ ViT-L + BCE
    (BCE public-optimal; ViT-L stronger than the ViT-B that hit 0.0798).
BET: train ViT-L + loss=BCE (else identical to prod896_large), 2 seeds (s42 GPU0, s43 GPU1) for
single + ensemble. best.pt is val-faithful for PUBLIC (born-digital = val distribution), unlike capture.
exp_prod896_large_bce[_s43]. Monitor scripts/bce_done_eval.sh generates submission_bce_{s42,s43,ens}.csv.
STRATEGIC NOTE: BCE = PUBLIC card; tail_margin ViT-L stays the PRIVATE card. public≠private (organizer:
public=born-digital distraction). Pursuing public<0.05 per user directive but NOT at cost of the private card.

### E4 CAPTURE DATA lever (MIDV-Holo) — LAUNCHED (2026-06-18)
After E3 (arch/loss: FHAG/focal/CDC) + TENT all failed to beat capture proxy 0.5466, pivot to the
ONLY proven capture lever = REAL PHYSICAL DATA. Found untapped captured-ID datasets in data/raw
(midv500/2019/holo, all real smartphone-captured, stored as zip/tar, never used for capture). Used
MIDV-Holo: extracted images.tar = 790 real captured GENUINE holographic passports + ~17k captured
FORGERIES (photo_replacement/photo_holo_copy/copy_without_holo/pseudo_holo_copy). This is the FIRST
real CAPTURED ATTACKS in training (E2 had only GenAI attacks). Built manifests (build_midvholo_manifest.py):
freuid_fid_midvholo_g (E2 + 790×4 genuine) and _gf (+1200 fraud, 3 frames/doc, attack_type=print_capture).
E4 = E2-early recipe (ViT-B DTC, 896, heavy_recapture, save_every_epoch) on _gf manifest. Goal: beat
fantasyid_test 0.5466. GPU1 (freed s43 BCE; s42 BCE continues as public card on GPU0). Monitor
scripts/e4_capture_eval.sh. Hypothesis: captured-genuine tightens the op-point genuine distribution
(proxy weak spot = APCER@1%); captured-attack adds real recapture-domain attack signal.

### E4 MIDV-Holo (genuine+fraud) — RESULT (2026-06-19): does NOT beat 0.5466; testing genuine-only
Capture proxy (fantasyid_test): E4-gf best ep0 0.6129 (AUC 0.752, APCER@1% 0.739) vs E2-early 0.5466
(AUC 0.759, APCER@1% 0.677). WORSE by 0.066. AUC held ≈E2 but OP-POINT worsened. Likely cause:
MIDV-Holo is holographic PASSPORTS only (narrow) + its captured FORGERIES are holographic-copy attacks
that shift the boundary away from FantasyID's GenAI attacks. Launched E4g (genuine-ONLY, manifest
freuid_fid_midvholo_g) to attribute: if genuine-only recovers → fraud hurt; if still >0.5466 → MIDV-Holo
narrow domain doesn't match the FantasyID proxy. (Strategic caveat: proxy=FantasyID; ACTUAL private is
captured+unseen — MIDV captured data MAY help real private even if not this proxy, like extgen.)

### E4g MIDV-Holo genuine-only — RESULT (2026-06-19): still loses; capture proxy firmly ceiling'd
Capture proxy: E4g genuine-only best ep1 0.5783 (AUC 0.744, APCER@1% 0.706) vs E4-gf 0.6129 vs E2-early
0.5466. Attribution: removing holographic FRAUD recovered ~0.035 (fraud hurts: holographic-copy ≠
FantasyID GenAI), but genuine-only STILL >0.5466 — MIDV-Holo narrow (holographic passports) genuine
doesn't tighten the FantasyID genuine distribution. E4g ep2 COLLAPSED 0.8414 (over-train narrow domain
wrecks op-point; early-stop pattern). VERDICT: MIDV-Holo REJECTED on proxy. Capture axis has now resisted
6 lever families: ViT-L variants, recap/warm-cap, FHAG/focal/CDC, TENT, MIDV-Holo data(both). E2-early
0.5466 STANDS. Last untested DIVERSE-captured-genuine lever = MIDV-500 (50 doc types). Retain MIDV ckpts
as private-bet (proxy≠actual private: private=captured+unseen, unknown distribution).

### ★★★ E5 CAPTURE BREAKTHROUGH (2026-06-19): diverse captured-genuine BREAKS the 0.5466 ceiling
The capture proxy (fantasyid_test) had resisted 6 lever families (ViT-L, recap, FHAG/focal/CDC, TENT,
MIDV-Holo). E5 = E2 base + MAX-diversity real captured GENUINE (MIDV-500 18 doc types + MIDV-Holo, ×2
upweight, NO fraud). Capture proxy:
  E2-early ........ 0.5466 (AUC 0.759, APCER@1% 0.677)
  E5 ep0 .......... 0.5239
  E5 ep1 .......... 0.4954
  E5 ep2 (best) ... 0.4548 (AUC 0.826, APCER@1% 0.593)   <- 17% BETTER, BOTH AUC & op-point up
FIRST lever to beat E2-early. Why it worked where MIDV-Holo (E4) failed: E4g's narrow holographic-
passport genuine didn't match FantasyID's DIVERSE genuine; MIDV-500's 18 diverse captured doc types do
→ tightens the op-point. And UNLIKE E4 (collapsed ep2), E5 improves MONOTONICALLY (no over-train collapse)
— diversity prevents it. Trajectory still accelerating at ep2 (Δ -0.029→-0.040) → MORE epochs likely lower.
=> CAPTURE AXIS IS *NOT* AT A CEILING — the lever is real captured-GENUINE DIVERSITY (data), not arch/loss.
Banked keep/e5_capgen_ep2__capture_0p4548.pt (new capture card, supersedes E2-early). Follow-ups RUNNING:
exp_e5_capgen_ep6 (×2, 6ep, GPU0) + exp_e5_capgen4_ep6 (×4 upweight, 6ep, GPU1) to find capture-optimal.

### E5 ep6 sweep — RESULT (2026-06-19): 0.4548 (E5 3ep ep2) is the capture OPTIMUM
Extended both ×2 and ×4 upweight to epochs=6. NEITHER beats the original E5 (×2, 3-epoch) ep2 0.4548:
  ×2-6ep: ep2 0.5244 (worse than 3ep's 0.4548 — 6ep cosine keeps LR higher), ep3 0.669, ep4-5 COLLAPSE (0.80,0.92)
  ×4-6ep: ep1 0.4726 (best), ep2 0.584, ep3-5 COLLAPSE
Findings: (a) the 3-epoch LR schedule is what makes ep2 optimal (lower LR at ep2 = better-converged);
(b) ×4 upweight does NOT beat ×2; (c) all runs collapse after ep3 (op-point over-trains even with diverse
genuine — early-ish stop essential). CAPTURE CARD LOCKED = E5(×2,3ep) ep2 0.4548 (keep/e5_capgen_ep2__capture_0p4548.pt),
17% better than E2-early 0.5466. Diverse captured-genuine (data) was the lever; epoch/upweight tuning is maxed.

### ★ E6 CAPTURE — MORE sources (BID) → 0.423 (2026-06-19)
Per user "expand sources": added MIDV-2019 (3 types, negligible 240) + BID (Brazilian CNH/RG/CPF real
photos, 8 types, 3200 _in.jpg). Capture proxy: E6-all ep2 0.4230 vs E6-noBID 0.4541 vs E5 0.4548 vs
E2-early 0.5466. CLEAN isolation: E6-noBID≈E5 (MIDV-2019 too few) → BID DROVE the gain (0.4541→0.423).
AUC held 0.828, op-point tightened (APCER@1% 0.593→0.557). Diverse captured-genuine SOURCES keep helping
(E2 0.547→E5 0.455→E6 0.423). Only used 3200/28800 BID → scaling BID next (E7). New card keep/e6_capgen_ep2__capture_0p423.pt.

### E7 BID-scaling — REJECTED (2026-06-19): more BID HURTS, E6 0.423 is optimum
Scaled BID 3.2k→8k/16k (×2 upweight, 3ep). BID-8k best 0.685 (collapses ep2 0.91); BID-16k best ep1
0.4842 — BOTH worse than E6 (BID 3.2k) 0.423. Over-representing ONE source biases the model (same lesson
as ×4 upweight: VOLUME hurts, moderate DIVERSITY helps). Capture lever = diverse SOURCES at moderate
weight. E6 0.423 STANDS. Proper next "expand sources" = NEW sources (not more BID).

### CAPTURE source-expansion — CONSOLIDATED at E6 0.423 (2026-06-19)
Source-expansion (user-chosen) progression: E2-early 0.5466 → E5 (MIDV-500 18 types + MIDV-Holo) 0.4548
→ E6 (+MIDV-2019 + BID 3.2k) 0.423. BID-scaling (E7) REJECTED (more of one source hurts). Remaining
captured sources EXHAUSTED on disk: dlc2021 cg.zip has 0 images (only metadata CSV); midv2020/doctamper
empty; idnet/docxpand = SYNTHETIC (born-digital, not captured). Only further captured data = download 32
more MIDV-500 types (~17G slow FTP, diminishing EV). CAPTURE CARD = E6 0.423 (keep/e6_capgen_ep2__capture_0p423.pt),
23% better than session-start E2-early 0.5466. Lever fully characterized: diverse captured-genuine SOURCES
at MODERATE weight (volume/upweight/over-one-source all hurt; arch/loss/calib all failed).

### E8 MIDV-500 full-50-types — REJECTED (2026-06-19): E6 0.423 is the FINAL capture optimum
Downloaded the 32 missing MIDV-500 types (50 total, 46 extracted, 13846 imgs). E8a (46 types @100/type,
volume≈E6) best 0.6045; E8b (46 types, all 13800) best 0.5023 — BOTH worse than E6 0.423 AND erratic
(non-monotonic vs E5/E6 clean descent). Spreading MIDV-500 over 46 thinly-covered types HURT (E8a also
under-sampled the proven 18 types to 100/type; E8b added volume — volume lesson). CONCLUSION: the E6 mix
(18 well-covered MIDV-500 types + BID 3.2k + MIDV-Holo + MIDV-2019, ×2, 3ep) is the SWEET SPOT. Capture
lever fully maxed at E6 0.423. Refined principle: diverse sources help ONLY with ENOUGH per-source
coverage + moderate total volume; thin coverage / volume bloat / over-one-source all hurt.
CAPTURE CARD FINAL = keep/e6_capgen_ep2__capture_0p423.pt (0.423, 23% better than E2-early 0.5466).

### CREATIVE/CONTRARIAN PLAN + RL assessment (2026-06-19)
Two fundamental walls behind ALL failures: (1) EGYPT↔BENIN antagonism (appearance-sensitizing collapses
EGYPT; genuine-diversity hurts BENIN), (2) clean-GenAI ≈ chance (forensic features can't catch it). Both
are limits of the supervised-binary-on-known-data paradigm. Creative directions:
  #1 ★ ONE-CLASS broad genuine-manifold: model ONLY genuine (100k+ diverse: FREUID+idnet+docxpand+MIDV+BID),
     score = distance from manifold. Robust to unseen ATTACKS + (if broad) unseen-TYPE genuine. Sidesteps
     antagonism (never learns attack-appearance). POC launched: frozen DINOv2 ViT-L feats, Mahalanobis +
     cos-kNN to non-held genuine manifold, eval EGYPT/BENIN LODO (scripts/oneclass_poc.py).
  #2 Transductive test-time adaptation: estimate genuine manifold FROM the private test images at deploy
     (mostly genuine). Beyond TENT. Zero train cost, apply on release.
  #3 Logical/semantic consistency (not forensics) for clean-GenAI tail: MRZ checksum validation — but
     FREUID is all DL/ID (no MRZ); only relevant if a private unseen type is a passport. Speculative.
  #4 Max genuine-diversity = truly-unseen bet (reframe extgen: helped EGYPT/MAURITIUS=appearance-OOD;
     private unseen types ARE appearance-OOD → may be right despite worse known-CV).
RL ASSESSMENT: best fit = directly optimize the NON-DIFFERENTIABLE FREUID Score (binding HM term =
APCER@1%BPCER op-point) via ES/policy-gradient HEAD fine-tune (forward-only, cheap). Plausible = RL/bandit
adaptive data-source curriculum to navigate the antagonism. Overkill = RL for the static core detector.

### ONE-CLASS genuine-manifold POC — REJECTED (2026-06-19), with a key insight
Frozen DINOv2 ViT-L feats, Mahalanobis + cos-kNN to a broad non-held genuine manifold (idnet+freuid+
bid+docxpand 6000). EGYPT FREUID 0.954 (AUC 0.61), BENIN 0.992 (AUC 0.375 — BELOW CHANCE). FAILS hard.
INSIGHT: forgery is NOT a visual-manifold outlier — pure DINOv2 feats encode visual TYPICALITY not
forgery; and clean-GenAI attacks sit CLOSER to the genuine manifold than real genuine (the manifold has
SYNTHETIC genuine docxpand/idnet, which are GENERATED → cluster with generated-attacks → distance is
ANTI-correlated, AUC<0.5 on BENIN). Feature-geometry explanation of the clean-GenAI≈chance wall.
=> one-class on type-agnostic visual feats is dead. Forensic-aware feats would just reproduce the
supervised logit (no gain). RL/ES op-point: refined assessment = LIMITED EV (seen op-point near-perfect
0 room; unseen op-point needs unseen data; deploy has no labels for reward). Direction #4 (max
genuine-diversity) already characterized by extgen (EGYPT 0.00275/BENIN 0.0734) = low marginal EV.
CONCLUSION: supervised ViT-L(unseen ~0.014) + diverse-captured-genuine(E6 capture 0.423) is near-optimal;
paradigm-shifts (one-class) don't beat it. Remaining real lever = #2 transductive at private release
(validated form = TENT, already have). Cards are strong; consolidate + deploy-prep.

### DINOv3 backbone upgrade — TEST LAUNCHED (2026-06-19)
Current backbone = DINOv2 ViT-L (vit_large_patch14_reg4_dinov2.lvd142m, Meta 2023, 142M-img pretrain).
DINOv3 (Meta 2025) IS in timm 1.0.27: vit_large_patch16_dinov3.lvd1689m — SAME self-supervised lineage
(preserves the OOD-robustness that makes DINOv2 crack unseen types, unlike CLIP/supervised which
semantic-shortcut → EGYPT risk) but 1.689B-img pretrain (12× more). Drop-in: 1024-d feats, 24 blocks,
has .norm, works @896 (patch16→56×56). Controlled swap: DINOv3 in the exact unseen recipe (FDA+tail_margin+
896+last-2-unfreeze), EGYPT-held + BENIN-held LODO. Compare to DINOv2 ViT-L: EGYPT 0.0035, BENIN 0.0165.
configs exp_dinov3_fold0/fold2. The one "newer backbone" worth testing (others lose OOD-robustness).

### DINOv3 backbone — REJECTED (2026-06-19): ~10× WORSE than DINOv2, newer ≠ better
Same recipe (FDA+tail_margin+896+last-2-unfreeze), only backbone swapped DINOv2→DINOv3 ViT-L. LODO:
  EGYPT: DINOv3 best ep3 0.0308 vs DINOv2 0.0035 (~9× worse)
  BENIN: DINOv3 best ep2 0.1736 vs DINOv2 0.0165 (~10× worse; early-stopped ep4)
DINOv3 is DRAMATICALLY worse. Likely causes: (a) patch16 vs patch14 → fewer tokens @896 (56² vs 64²) =
less spatial resolution for FINE forensic detail (documents need it); (b) recipe tuned for DINOv2 (lr/
unfreeze); (c) DINOv3 feats optimized for seg/depth, less for fine-grained forgery. CONCLUSION: DINOv2
ViT-L patch14 STAYS the unseen backbone. "Use a newer model" does NOT help here — the patch14 fine-grained
resolution + DINOv2's specific features beat DINOv3 despite 12× pretrain data. (CLIP/SigLIP/supervised would
be even worse — lose OOD-robustness.) Backbone is NOT the lever; the walls are data/signal limits.

### EVA-02 / AIMv2 backbones — REJECTED (2026-06-19): ~16× worse; backbone is NOT the lever
BENIN-held LODO, same recipe: EVA-02 best ~0.264, AIMv2 ~0.283 vs DINOv2 0.0165 (~16× worse), worse even
than DINOv3 (0.1736). Both 224-native (4× pos-embed interp @896) + supervised/MIM pretrain (weaker OOD than
DINOv2's 142M curated SSL). 3 alternatives (DINOv3/EVA-02/AIMv2) ALL far worse → DINOv2 ViT-L patch14 is
decisively optimal; bottleneck is data/signal, NOT backbone capacity/recency.

### MAMBA — assessed, NOT pursued (2026-06-19)
timm 1.0.27 has ONLY "MambaOut" — ironically the paper "MambaOut: Do We Really Need Mamba for Vision?" that
REMOVES the SSM (gated-CNN, IN-1k supervised, ~100M, 224/384). No real vision-Mamba (VMamba/Vim/MambaVision)
in timm; mamba_ssm CUDA kernel NOT installed. Even if set up → near-certain failure: supervised low-res
pretrain (weak OOD, semantic-shortcut → EGYPT) + smaller than DINOv2 = same mode as EVA-02/AIMv2/DINOv3.
NOT worth setup cost (backbone≠lever, 3 alts rejected). MambaOut paper itself: Mamba unneeded for vision recog.

### SAM (Sharpness-Aware Minimization) — different TRAINING METHOD, LAUNCHED (2026-06-19)
After backbone exploration exhausted (DINOv2 optimal), try a different OPTIMIZER/training method. SAM seeks
FLAT minima (ascend to worst-case in a rho-ball, grad there, descend) → better OOD/unseen-type generalization
= directly the prize. Implemented in _transfer train.py (_sam_ascent/_sam_restore, cfg.sam_rho, accum=1
two-pass). rho=0.05. Same recipe (DINOv2 ViT-L + FDA + tail_margin + 896 + last-2-unfreeze) on EGYPT-held +
BENIN-held LODO. vs DINOv2 baseline EGYPT 0.0035 / BENIN 0.0165. Cost: ~4× slower (eff_bs 16 + 2 passes).
Rationale: unseen-card is strong but the PRIZE is robustness to UNKNOWN unseen types — SAM's flat-minima
generalization is the principled lever; worst-fold (MOZ 0.0602) headroom suggests worst-case robustness matters.

### SAM — RESULT (2026-06-19): appearance-OOD lever (EGYPT best-ever) but EGYPT↔BENIN antagonism
SAM (rho 0.05) LODO: EGYPT 0.0035→0.00211 (40% BETTER, BEST-EVER EGYPT, beats extgen 0.00275!) but
BENIN 0.0165→0.0494 (3× WORSE). Flat-minima HELP appearance-OOD (EGYPT robust to appearance shift) but
BLUR the sharp chroma boundary BENIN's clean-GenAI tail needs — the SAME EGYPT↔BENIN antagonism as
genuine-diversity/appearance-inv. First NEW method this session to IMPROVE a fold. Net 2-fold worse
(BENIN dominates), BUT private unseen = appearance-novel (EGYPT-like) → SAM may be the right bet. Testing
the HIGHEST-LEVERAGE fold MOZAMBIQUE (worst 0.0602) + MAURITIUS (appearance-OOD) to get the 5-fold picture.
Banked keep/sam_egypt_ep2__0p0021_evidence.pt.

### ★ SAM NET-EFFECTIVE (2026-06-19) → EXTENDING to capture + 5-fold
SAM LODO best-epoch: EGYPT 0.00211 (vs 0.0035), MAURITIUS 0.0024 (vs 0.0052, ep0 then collapse),
MOZAMBIQUE 0.0217 (vs 0.0602, ep0 — 64% better, WORST fold lifted!), BENIN 0.0494 (vs 0.0165, worse).
4-fold mean 0.0189 vs DINOv2 0.0214 = NET ~12% better despite BENIN. SAM = flat-minima → worst-case/
appearance-OOD robustness (the prize for unknown unseen types); only the chroma-tail BENIN is hurt. SAM
best-epoch is EARLY (ep0-2; MAURITIUS collapses after ep0) → needs per-epoch LODO selection / early-stop.
EXTENDING per user: (1) capture-SAM (exp_e6sam_capgen, root ViT-B DTC line — SAM ported there), (2) GUINEA
SAM (exp_sam_fold1, complete 5-fold), (3) production-SAM (exp_sam_prod, all-types deployment unseen card).
Banked keep/sam_moz_ep0__0p0217_evidence.pt.

### SAM capture-extension bug FIXED (2026-06-19): fp16+SAM=NaN → bf16
capture-SAM (root DTC line) crashed NaN at ep0 eval: root uses fp16+GradScaler, an overflow grad→inf, SAM
normalization does inf*0=NaN → weights corrupt. (_transfer SAM works = bf16, no overflow.) FIX: root train.py
uses bf16 + scaler disabled when sam_rho>0. Relaunched capture-SAM; production-SAM (GPU1) was unaffected.

### SAM on CAPTURE — NEUTRAL (2026-06-19): SAM is an UNSEEN lever, not capture
capture-SAM (bf16 fix worked, no NaN) proxy: ep0 0.586, ep1 0.489, ep2 0.4323 vs E6 (no SAM) 0.423 — tied/
marginally worse. SAM does NOT help capture (capture bottleneck = real captured-genuine DATA, not flat-minima
generalization). Capture card STAYS E6 0.423. SAM's value is the UNSEEN axis only (worst-case/appearance-OOD).

### ★ SAM rho=0.05 — NET-WORSE on FULL 5-fold (2026-06-20): GUINEA collapses 12×
COMPLETE 5-fold: EGYPT 0.00211(better), MAURITIUS 0.0024(better), MOZ 0.0217(better), BENIN 0.0494(3× worse),
GUINEA 0.2517 (vs 0.0216 = 12× WORSE, unstable 0.77→0.25→0.65). 5-fold mean 0.0655 vs DINOv2 0.0214 = ~3×
WORSE. The earlier "net 12% better" was a 4-FOLD ARTIFACT (GUINEA unmeasured). rho=0.05 is TOO AGGRESSIVE →
destabilizes GUINEA/BENIN (collapse at varying epochs). SAM NOT consolidated; DINOv2 production card STANDS;
production-SAM NOT promoted. Capture-SAM also neutral (SAM = no axis win as deployed). SALVAGE candidate:
smaller rho (0.02) may stabilize GUINEA/BENIN while keeping EGYPT/MOZ gains — testing on the 2 collapse folds.

### SAM rho=0.02 salvage — LAUNCHED (2026-06-20): test if gentler rho stabilizes GUINEA/BENIN
rho=0.05 collapsed GUINEA(12×)/BENIN(3×). Testing rho=0.02 on those 2 folds: if they return near DINOv2
(GUINEA 0.0216, BENIN 0.0165) while EGYPT/MOZ keep gains → SAM salvageable & net-positive. If still collapse
→ SAM fundamentally too unstable for this task, abandon. exp_sam02_fold1(GUINEA)/fold2(BENIN).

### ★ SAM — ABANDONED (2026-06-20): not salvageable, GUINEA structurally broken at any rho
rho=0.02 salvage FAILED: GUINEA 0.4258 (vs DINOv2 0.0216, ~20× worse, even worse than rho0.05's 0.2517,
never converges 0.79→0.59→0.43); BENIN 0.0296 (partial recovery from 0.0494 but still > DINOv2 0.0165).
SAM destabilizes GUINEA STRUCTURALLY — not a rho-tuning issue. FINAL: SAM net-worse on full 5-fold at all
rho; ABANDONED. EGYPT/MOZ gains (0.00211/0.0217) come bundled with GUINEA/BENIN collapse — NOT a clean
appearance-OOD lever. DEPLOYMENT UNCHANGED: DINOv2 ViT-L unseen (~0.014) + E6 capture (0.423) + ViT-B FDA
public (0.0798). SAM code retained behind sam_rho flag (both lines, bf16-safe) for paper ablation only.
LESSON: validate on FULL 5-fold before declaring a win (the "net 12%" 4-fold mirage misled).

### PUBLIC-GAP investigation + diverse-synthetic-data bet (2026-06-20) — rank 17/92, chasing public
Leaderboard: hoppery rank 17/92, public 0.04298; top teams 0.0006-0.01. We're far behind on PUBLIC.
INVESTIGATION: (a) public test VERIFIED born-digital (visual: clean synthetic FREUID-style African DLs,
no capture artifacts; research agent's "print-capture in public" claim = WRONG, that's private). (b) public
= FREUID's own synthetic generator (5 types, standardized 1.58 crop) but HARDER than our train: we get 0.09
public vs LODO 0.02 vs in-domain 0.0003 (random val LEAKY). (c) top teams crack it to 0.0006. Deep-research
(general-purpose agent, 9 web sources) key levers for born-digital ID forgery: diverse synthetic data (+27
AUC), two-stream fusion DINOv2+forensic (IDNet EER 27.9→8.3), semantic-shortcut suppression, LoRA+WiSE-FT
(vs full-FT collapse), multi-res TTA+TENT (+3-6%), shift-matched threshold/selection (our val lies).
UNTESTED HIGH-EV LEVER: idnet has 35,874 synthetic ATTACKS, never trained-on/submitted. LAUNCHED on the
diverse union (freuid+idnet attacks+docxpand/bid, 186k, 22 types): exp_union_vitl (ViT-L+BCE+FDA, our
best-public recipe + diversity) + exp_union_dtc (ViT-B DTC two-stream, research ID-PAD pattern), 896,
heavy_recapture OFF (born-digital). Measure only via Kaggle submit (user auth). NOTE: other-PC found idnet
HURTS FREUID-LODO @896 — but public is the BROADER distribution where idnet diversity may help (the bet).

### PUBLIC diverse-data (idnet) bet — FAILED (2026-06-20): idnet HURTS public, it's FREUID-specific
union ViT-B DTC (FREUID+idnet+docxpand 167k) public = 0.26506 vs our best 0.043, vs FREUID-only DTC 0.216.
idnet (EU/US synthetic IDs) DILUTES the FREUID-specific African-DL public distribution → WORSE. Confirms
other-PC "idnet diversity HURTS @896" — applies to public too. The public gap is NOT lack of data; it's
generalization to harder FREUID-distribution examples. ViT-L union killed (same flawed idnet premise; in-domain
0.0514 = struggling). Best public stays transferred 0.043 (unreproducible) / reproducible ViT-B FDA 0.0798.
NEXT: NOT more data. Levers = anti-overfit (LoRA+WiSE-FT), TTA/TENT, ensemble of best CSVs (incl 0.043), or
semantic-shortcut suppression — all on the FREUID distribution only.

### PUBLIC anti-overfit attempts — BOTH FAILED (2026-06-20)
(1) diverse-data (idnet union): 0.265 (idnet dilutes FREUID-specific dist). (2) WiSE-FT a=0.5 on ViT-L+BCE:
0.199 vs base 0.094 — interpolating backbone TOWARD pretrained HURTS → our fine-tuning is GOOD, not
overfit-recoverable; WiSE-FT premise wrong for our partial-unfreeze setup. Public gap is NOT fixable by
diverse-data OR weight-interp. Pattern: every public lever this session (diverse data, WiSE-FT, backbone
swaps, SAM) FAILS. Best public stays transferred 0.043 / reproducible 0.094 (ViT-L BCE) / 0.0798 (ViT-B FDA).
Remaining anti-overfit lever = LoRA (needs impl on the _transfer ViT-L line; EV now lower given WiSE-FT
showed fine-tuning isn't the problem). Reaching top-3 (0.0006) needs an approach not yet found.

### PUBLIC rank-ensemble (cross-arch diversity) — BUILT, awaiting submit (2026-06-20)
champ_s42 (0.068) = champion+hpf_dual ENSEMBLE beat singles (LODO 0.0089<0.0109) → ensembling reliably helps.
Built submission_rankens.csv = rank-avg of 4 DIVERSE archs: ViT-B DTC+FDA (0.0798) + ViT-L tail-ens (0.0935)
+ ViT-L BCE (0.094) + ConvNeXt hpf_dual. Valid (142818 rows, label 0.009-0.76). BUT today's 5 submissions
exhausted (union_dtc 0.265 + wiseft 0.199 failures + recap + others) → auto-submit on daily reset
(scripts/retry_submit_rankens.sh). Lesson: build the ensemble BEFORE spending submissions on speculative runs.

### DURING-WAIT public exploration (2026-06-20, awaiting daily reset): RINE + diverse ensemble members
Public daily limit hit; building tomorrow's 5-submission slate. Finding: rank-ensemble has 0 confident-attack
preds (>0.9) → models DISAGREE on attacks → public attacks genuinely hard → pseudo-labeling DROPPED (insufficient
confident labels). Launched: (A) RINE-style (frozen DINOv2 ViT-L + intermediate-layer CLS aggregation via learned
layer-weights + tiny head, scripts/rine.py) — targets hard GenAI via mid-level forensic cues the final CLS misses;
(B) ViT-B rgb+FDA (new diverse ensemble member). Plus prep: TTA(multi-res+flip) + comprehensive rank-ensemble CSVs.

### PUBLIC strategy: GENERALIZATION-HARDENING (2026-06-21) — EMA + self-blend + mild-color on ViT-B FDA
After many public failures (idnet 0.265, WiSE-FT 0.199, RINE 0.319, ConvNeXt-ens 0.248), root cause = overfit/
generalization (in-domain 0.0003 → public 0.094). Best public ViT-B DTC+FDA (0.0798) had NO EMA + light aug.
NEW strategy targets the root cause: EMA (flat minima→generalization) + self-blend sbd_p=0.2 (harder synthetic
attacks → catch public's hard GenAI) + mild-color aug (robustness, no resample). 2 variants: v1 (EMA+sb+color),
v2 (EMA+sb, no color). exp_pubgen_v1/v2 on the 0.0798 ViT-B DTC FDA base. Generate CSVs, NO auto-submit (user
controls). (Lottery s45/s46 died — environment disruption; pivoted to this.)

## D19 — Targeted FIELD-TAMPER aug (data-analysis-driven, deep-research-backed) [2026-06-21]
**Root cause (data analysis):** public 5.2% hard cases = subtle FIELD TAMPERING (strikethrough/overwrite/
char-substitution/field copy-move) where our models are confidently WRONG and DISAGREE with each other.
NOT a low-level distribution shift (public≈train stats). Generic SBD/idnet/WiSE-FT/RINE all failed because
they don't target THIS forgery class.
**Lever (deep-research 2026-06-21):** synthesize the hard class directly. data/field_tamper.py turns a GENUINE
born-digital sample → field-tampered attack: pick a right-biased text-field strip → copymove/overwrite/strike/
recompress → ★ localized low-QF JPEG (QF70-88 vs host ~95 = the DeepID-winner "Sunlight" compression-mismatch
signal) → alpha-FEATHER blend (suppress splice edge → HARD near-boundary positive, not a trivially-detected splice).
cv2-only, 2.0ms/img. DCT/freq branch rejected (counterproductive on born-digital PNG).
**Runs:** exp_ft03 (field_tamper_p=0.3, GPU0) / exp_ft05 (p=0.5, GPU1), both = ViT-B DTC FDA 0.0798 base +
EMA(0.999) + field_tamper. 896px, batch32, 6ep. num_workers=12 (avoid the 56-worker oversubscription that
crawled pubgen). Goal: lower public score by catching the field-tamper hard cases. NO auto-submit (user controls).

## D19 결과 — field-tamper 학습 완료 + 오프라인 진단 [2026-06-22]
ft03(p=0.3)/ft05(p=0.5) 6ep 완주 (setsid로 세션중단 면역 — 첫 시도는 ep0 후 SIGHUP사). public 7821 추론 vs base(vitb_fda 0.0798):
- ✓ **타깃 작동 증명:** 취소선 hard케이스 002cdc base 0.034(놓침)→ft 1.000(검출). base가 놓친 genuine<0.3을 ft가 attack>0.7로 새로 잡음 1021건(ft03+ft05 동의 1003 고신뢰), 역방향 놓침 5건뿐.
- ⚠️ **operating point 과이동:** 확신-genuine base 47%→ft03 26.5%→ft05 **0.0%**(과조리). ft05는 압축/블렌딩만 보면 다 attack — REJECT 유력.
- rank 메트릭이라 절대이동 자체는 무해하나, ft05의 genuine단 붕괴는 순위 risk. ft03이 균형점.
- **헤지:** rank-fusion base+ft03(submission_ftfuse_b03) = base 보정 유지 + tamper 부분회복(002cdc 0.579). base+ft03+ft05(b0305)도 생성.
- **미해결:** public ground-truth 없어 net효과는 제출로만 판정 가능. 후보 슬레이트 준비, 제출은 사용자 지시 대기.

## ★★ D19 제출 결과 — field-tamper = PUBLIC 돌파 [2026-06-22]
**submission_ft03.csv → public 0.01030, 리더보드 RANK 9** (이전 최고 0.04167 sbi_tta 대비 4×, 0.0798 base 대비 8×).
field-tamper가 결정적 레버임 입증. over-shift 우려는 rank 메트릭에서 무해(순서만 중요). 메모리 project_public_gap_loss
전면 갱신("public=distraction/0.0798한계" 폐기). 리더 0.0006-0.005, 우리 0.0103 = 톱10.
**다음:** ① field-tamper를 더 강한 DINOv2-L SBI recipe(sbi_tta 백본)에 적용 ② field_tamper_p 튜닝 ③ ft03+sbi 융합 검토.

## ★ D19 LODO 검증 — field-tamper는 unseen-type에서 ANTAGONISTIC [2026-06-22]
동일 ViT-B DTC FDA 896 EMA + field_tamper0.3, fold hold-out (val=unseen 타입):
- **EGYPT (appearance-OOD): 0.0217 → 0.1901** (ep0 best, 이후 AUC 0.97→0.50 붕괴, APCER 0.31→0.99). 낯선 외형을 조작으로 오인 → genuine 오탐 → op-point 파괴. 매 epoch 악화.
- **BENIN (clean-GenAI): 0.0780 → 0.0426** (ep3 best). 국소-이상 탐지가 도움.
fold_hardness 적대성 재현(반대 방향): appearance-inv=EGYPT돕고BENIN해침 / field-tamper=BENIN돕고EGYPT해침.
**함의:** bare field-tamper = 위험한 private 베팅(2 unseen 타입 미지, appearance-OOD면 붕괴). public(0.0103)은 seen-appearance라 이 위험 안 드러남. → 앙상블 헤지(appearance-robust base와 융합) 필요. 다음: 앙상블이 EGYPT 회복+BENIN 유지하는지 LODO 검증.

## ★★ D19 앙상블 헤지 검증 — 3-way가 적대성 해결 [2026-06-22]
LODO held-out fold rank-fusion (scripts/ensemble_lodo_eval.py):
거울상 적대성: appinv EGYPT 0.0026/BENIN 0.1900 ↔ field-tamper EGYPT 0.1901/BENIN 0.0426.
| 모델 | EGYPT | BENIN | worst | mean |
|---|---|---|---|---|
| base | 0.0217 | 0.0780 | 0.0780 | 0.0499 |
| field-tamper | 0.1901 | 0.0426 | 0.1901 | 0.1164 |
| appinv | 0.0026 | 0.1900 | 0.1900 | 0.0963 |
| ft+appinv | 0.0225 | 0.0846 | 0.0846 | 0.0536 |
| **ft+base+appinv** | **0.0076** | **0.0682** | **0.0682**★ | **0.0379**★ |
단일 aug은 반대 fold 붕괴. 2-way ft+appinv는 어중간(서로 약점 평균). **3-way(ft+base+appinv)=두 적대 실패모드 모두 헤지, worst·mean 모두 최고, 양쪽서 base 능가.**
**배포 레시피(private unseen-type 성격 미지): all-types 3모델 rank-fuse = ft03(보유)+plain base(보유)+all-types appinv(학습필요, d10은 LODO전용).**

## ★ D19 싼 검증 — field-tamper는 ViT-L에 기여 못함 [2026-06-22]
ViT-L LODO 예측(keep/vitL_fold*_lodo) + ft LODO 예측을 held-out fold에서 rank-fuse (scripts/score_dump.py 라인별 덤프; ViT-L EGYPT 0.0035 정확 재현=파이프라인 검증).
| weight | EGYPT (ViT-L 0.0035) | BENIN (ViT-L 0.0162) |
|---|---|---|
| +10%ft | ~0.0041 | 0.0151(미미↑) |
| +20%ft | 0.0048✗ | 0.0168✗ |
| +50%ft | 0.0100✗ | 0.0222✗ |
ViT-L이 이미 두 fold 거의 완벽 → field-tamper는 노이즈만 추가. "ft가 ViT-L이 놓친 공격 클래스 커버" 가설 기각.
**최종 결정:** ft03=PUBLIC 전용(rank9). private 배포=ViT-L prod(unseen)+E6(capture)+TENT, field-tamper 미포함.

## ★ ViT-L 강화 — stronger-FDA = strict 업그레이드 [2026-06-23]
field-tamper on ViT-L 게이트 실패(p=0.15도 EGYPT 0.0035→0.026 붕괴, backbone 무관). SBI도 저널상 EGYPT 붕괴 → forgery-합성 전부 public 전용.
대신 **FDA 강화(beta 0.05→0.12, 비-적대)** 검증:
| fold | beta0.05 | beta0.12 |
|---|---|---|
| EGYPT | 0.0035 | **0.0027** (-23%) |
| BENIN | 0.0162 | **0.0140** (-14%) |
+ public 0.037(prod 0.109 대비). **public·private 둘 다 개선 = 새 ViT-L 카드.** all-types 배포 학습 중(exp_fdab12_all, ep2 선택). EGYPT LODO는 ep3 이후 overtraining → early epoch 배포.

## ViT-L FDA beta 스윕 [2026-06-23]
| beta | EGYPT | BENIN | mean |
|---|---|---|---|
| 0.05 | 0.0035 | 0.0162 | 0.0099 |
| 0.12 | 0.0027 | 0.0140 | **0.0084**(최적) |
| 0.16 | 0.0024 | 0.0160 | 0.0092 |
EGYPT는 beta↑ 계속 개선(appearance-OOD에 FDA 직접 도움), BENIN은 0.12 최저·0.16 악화. **mean 최적 beta=0.12 유지.** 미세 적대성(EGYPT↑/BENIN↓) 있으나 0.12가 균형점.

## ★ 레버 #2 TENT 검증 — 큰 승리 [2026-06-23]
beta0.12 LODO 모델에 TENT(LN affine entropy-min + anti-collapse div, scripts/tent_lodo_eval.py, transductive on held-out fold):
| fold | no-TENT | +TENT |
|---|---|---|
| EGYPT | 0.0020 | **0.0016** |
| BENIN | 0.0137 | **0.0031** (-77%) |
누적: 원prod(beta05) EGYPT0.0035/BENIN0.0162/mean0.0099 → **stronger-FDA+TENT 0.0016/0.0031/mean0.0024 (4× 개선).** TENT는 추론-시간(LN만), private 배포 시 적용. 레지스트리 주장(BENIN 0.0061)보다 우수.

## 레버 #3 멀티시드 — 미채택 [2026-06-23]
beta0.12 s42+s43 rank-fuse: EGYPT 0.0020→0.0019(미미↑), BENIN 0.0137→0.0201(↓, s43=0.0381 나쁜시드가 끌어내림). 2-시드 손해. 3-5시드 필요하나 비용 대비 TENT가 이미 강함 → 미채택. BENIN 시드분산 큼(주의).

## ★ 레버 #4 capture — ViT-L은 capture-blind, 2-카드 확정 [2026-06-23]
배포 ViT-L(stronger-FDA) on fantasyid_test(촬영/물리+unseen타입): plain **0.9606**, TENT 0.9824 (둘다 chance, AUC~0.52). E6 카드는 같은 proxy 0.423.
**ViT-L LODO 강화(mean 0.0024)는 born-digital을 타입hold-out = unseen-TYPE만 테스트, captured-ACQUISITION 전환은 미테스트.** captured에서 ViT-L 무력. TENT도 capture선 악화.
**함의:** 주최 인텔 "private=captured/physical+2 unseen". private이 captured-heavy면 ViT-L 단독 약함 → capture 카드(E6)가 핵심. ViT-L 강화는 unseen-type축 진짜지만 acquisition갭 못넘음. → 2-카드(ViT-L unseen + E6 capture) 확정. 다음 강화대상 = capture 카드. caveat: fantasyid_test는 cross-corpus라 과장 가능.

## 레버 #4 후속 — capture 카드 강화 시도 (E6 못 이김) [2026-06-23]
E6 약점=APCER@1% 0.55(op-point), AUC 0.83(랭킹 OK). TENT capture 무효(E6 0.42→0.43). tail_margin+FDA on capgen_e6(ViT-B): ep0 0.498(APCER 0.64), 이후 붕괴 — E6 0.423 못 이김. tail_margin op-point 보정이 fantasyid_test cross-corpus 전환 못 넘음. E6 DTC가 capture에 더 적합, 메모리대로 maxed 확인.

## ★ TENT 제출 — proxy-overfit 판명 [2026-06-24]
submission_fdab12_tent (stronger-FDA ViT-L ep2 + TENT) → public **0.13041** (champion plain 0.037보다 훨씬 나쁨).
**TENT는 born-digital-LODO proxy(coherent 단일 unseen-type 이동)에만 도움**: public(seen-type, 0.037→0.13)·capture proxy(E6 0.42→0.43, ViT-L 0.96→0.98) 모두 해침. "deploy with TENT" 계획 falsified — 배포에 TENT 미사용.
TENT 없는 실제 FDA 강화는 modest: LODO mean 0.0084 vs prod 0.0099(~15%), 4× 아님. **최고 public 여전히 ft03 0.0103(rank9).** plain ViT-L(~0.037)은 기존 fda_tta champion과 거의 중복.

## ★ ftai 통합 실패 — field-tamper는 appearance-inv로도 private-호환 불가 [2026-06-30]
field-tamper + appearance_inv JOINT 학습(ViT-B DTC LODO): EGYPT 0.0256(붕괴 0.19는 막음, 단 baseline 0.0217보다 나쁨), BENIN 0.0873(baseline 0.0780보다 나쁨). mean 0.056 > baseline 0.050 > ViT-L FDA 0.0084. 사후 앙상블과 동일하게 "양쪽 약점 평균". **forgery-합성은 unseen-type을 근본적으로 해치고 공동학습으로도 못 품. public(field-tamper)과 private-unseen(ViT-L FDA)은 별개 모델.** 통합 경로 종료.

## ★ E6+FDA — capture 카드 첫 개선 [2026-06-30]
ViT-L capture 실패(0.67-0.79, DTC가 핵심). E6=DTC+heavy_recapture+FDA미사용 발견 → FDA 추가:
E6+FDA0.4 ep2 = fantasyid_test **0.4120** (E6 no-FDA 0.423 근소 능가, AUC 0.838 APCER 0.547). FDA0.2 ep2=0.442. DTC 헤드 유지+FDA보조 = capture 소폭 강화. 새 capture 카드 후보=exp_e6_fda/epoch2.

## E6+FDA+appinv — 실패, capture 카드 최종=E6+FDA0.4 [2026-06-30]
E6 DTC + FDA0.4 + appearance_inv(full/mild) on capgen_e6 → fantasyid_test: full 0.688, mild 0.621 (E6+FDA 0.412보다 크게 나쁨). appearance_inv는 색/텍스처 파괴로 cross-corpus 촬영판별 악화. **capture 카드 최종 = E6+FDA0.4 ep2 = 0.412** (E6 0.423 대비 마진). FDA는 도움(unseen-type 일반화), appinv는 독.

## ★★★ 아키텍처-다양성 앙상블 = RANK 4, 0.00072 [2026-07-01]
**핵심 돌파: field-tamper 모델을 여러 아키텍처 FAMILY로 학습→앙상블.**
진행: 0.00145(v3 champion) → 0.00135(v2 2-seed) → 0.00131(6-model v2+v3 합성) → **0.00072(8-model +2 ConvNeXt)**. rank 6→4.
상관 비교: same-seed 0.998 / v3-합성변형 0.996 / **ConvNeXt(CNN) vs ViT-L 0.953** — 아키텍처 family가 진짜 decorrelation.
같은 recipe(FDA0.12 tail_margin + v2/v3 field-tamper ep2 hflip TTA)를 다른 backbone으로. 8-model=6×ViT-L+2×ConvNeXt.
**방향(계속): 더 많은 아키텍처 family 추가** — EVA-02(MIM ViT)+ConvNeXt s44 학습중 → 다음 Swin/MaxViT/forensic-CNN → 톱3(0.00053-0.00063) 추격.
**교훈: 앙상블 이득은 시드/합성 아닌 아키텍처 다양성에서 나온다.**

## ★★★ 아키텍처 다양성이 PRIVATE unseen-type도 뚫음 — EGYPT↔BENIN 적대성 해결 [2026-07-02]
private track: ViT-L FDA + ConvNeXt FDA (field-tamper 無) LODO 앙상블:
| fold | ViT-L FDA | ConvNeXt FDA | 상관 | 앙상블 best |
|---|---|---|---|---|
| EGYPT | 0.0020 | 0.0149 | 0.848 | **0.0002** (50/50) |
| BENIN | 0.0137 | 0.0023 | 0.781 | 0.0062 (70cx) |
mean 0.0032 (ViT-L 단독 0.0084의 2.6배↑). **상보적 fold 강세**(ViT-L=EGYPT, ConvNeXt=BENIN)로 프로젝트 최대 난제 EGYPT↔BENIN 적대성 해결.
상관 0.78-0.85 = 아키텍처 family는 unseen-type에서도 decorrelated. field-tamper는 제외(unseen 붕괴) — FDA만으로 안전.
**private 카드 = ViT-L FDA + ConvNeXt FDA 아키텍처-다양성 앙상블(unseen) + E6+FDA(capture).** 다음: EVA/더 많은 arch + all-types 배포판.

## 근본 lever: TRANSDUCTIVE PSEUDO-LABELING [2026-07-02]
앙상블 다양성 천장 확인(10-model +Swin = 0.00073 ≈ 0.00072; Swin 강+diverse했지만 무이득 → 잔여 hard 케이스는 아키텍처로 안 풀림).
새 lever = 자기학습: 0.00072 앙상블의 public 예측을 rank-기반 양측 pseudo-label로(하위25%→genuine 1955 / 상위55%→attack 4302 / 경계 1564 제외 — 절대확률 임계는 tail_margin 시그니처 탓 attack만 잡혀 편향 위험, rank로 해결) train에 편입(manifests/freuid_publicpl.parquet, source=publicpl, val_sources=freuid로 val 청정).
모델이 **테스트 분포 자체**(공개 생성기 지문)를 학습 + pseudo-genuine 위에 field-tamper 합성(테스트 분포 위 tamper 양성 생성). exp_plvit_large + exp_plcnx_large (2 diverse arch) 학습중.
**private 공개일 transductive 플레이(creative plan #2) 리허설이기도 함.** 완료 후 앙상블 편입/교체 비교 → 0.00072 돌파 시도.

## pseudo-labeling lever 폐기 — 규정 리스크 [2026-07-02]
transductive pseudo-labeling(test 이미지+자기예측 학습 편입)은 Kaggle 일반 관행이나, 대회 규정 미확인(로그인 벽) + 사용자 지시("부정행위에 해당한다면 시도하면 안된다")에 따라 **확인 전 금지로 간주, 착수 즉시 중단·산출물 전부 삭제**(manifest/configs/checkpoints). 
주의: 기존 제출 중 submission_fdab12_tent(0.13, TENT=test-time LN 적응)도 같은 범주 — 최종 선택 금지. **우리 best(ft8model 0.00072 등)는 청정**: train 데이터+자체 합성 aug만으로 학습, 추론은 feed-forward+hflip TTA(표준 추론시간 증강, 파라미터 적합 없음)뿐.

## ★★★★ PUBLIC RANK 1 — 0.00049 [2026-07-03]
ft10gh = (8×8model + ViT-Giant + ViT-L@1120)/10 → **0.00049, 리더보드 1위** (2위 seantangth 0.00053).
결정타 = 새 다양성 축 2개: **용량**(ViT-Giant 1.1B, corr 0.952) + **해상도**(ViT-L@1120, corr 0.9358=역대최저, 같은 아키텍처인데 해상도만으로 ConvNeXt보다 diverse).
다양성 축 정리: 시드 0.998(무익) < 합성변형 0.996(미미) < 아키텍처 family 0.936-0.953(대박) ≈ 해상도 0.936 ≈ 용량 0.952. Swin(0.951)은 무익했으나 giant+hires는 성공 — 기전 관련성(미세 tamper 디테일)이 차이.
전체 여정: 0.0798(r17)→0.0103(r9)→0.00145(r6)→0.00135→0.00131→0.00072(r4)→**0.00049(r1)**.
주의: 상금은 PRIVATE — private 카드(ViT-L FDA+ConvNeXt FDA+E6, 청정)는 별도 유지. public 1위는 방법론 검증+논문 무게.

## ★ 리드 방어 성공 — 0.00039 (1위 마진 확대) [2026-07-03]
ft12model = (8×m8 + giant + hires42 + hires43 + **cnx1120**)/12 → **0.00039** (0.00049→, 2위 0.00053).
결정타 = **cnx1120 = 아키텍처×해상도 교차 멤버** (corr 0.9239 역대최저, AUC 1.0). "입증된 축의 교차 = 최강 다양성" 확인.
여정: 0.0798(r17)→0.0103→0.00145→0.00135→0.00131→0.00072→0.00049→**0.00039(r1)**.

## ★★ private 해상도 축 = unseen-type 2.7배↑ [2026-07-03]
public 다양성 사다리(arch→res)를 private LODO에 재적용. 3-멤버 앙상블(896 ViT-L + 896 ConvNeXt + 1120):
| fold | 896×2 best | +1120 best | 조합 |
|---|---|---|---|
| EGYPT | 0.0002 | **0.0001** | vl896+cx896+vl1120 |
| BENIN | 0.0062 | **0.0022** | cx896+cx1120 (ConvNeXt@1120 단독 0.0010!) |
mean 0.0032→**0.0012**. ConvNeXt@1120이 약한 fold BENIN을 0.0023→0.0010으로 돌파.
배포 private-unseen 카드 = 4멤버 all-types {ViT-L FDA@896/@1120, ConvNeXt FDA@896/@1120}. @896 all-types 보유, @1120 all-types 학습 필요.

## ★ private capture 해상도축 [2026-07-03]
E6+FDA@896 fantasyid 0.4120 → E6+FDA@1120 **0.3617** (AUC 0.838→0.868, APCER 0.547→0.495). 해상도축이 capture에도 전이. capture 카드 갱신 후보=exp_e6_fda1120/epoch2.

## capture 앙상블 = 0.354 [2026-07-03]
E6+FDA 896+1120 rank-fuse(896:0.3/1120:0.7) = fantasyid **0.3540** (896=0.412, 1120=0.3617 대비). capture 카드 = E6+FDA 2해상도 앙상블.
private 강화 종합: unseen mean 0.0032→0.0012, capture 0.412→0.354. 배포 @1120 all-types(ViT-L,ConvNeXt) 학습중.
NEXT P3: unseen FDA 앙상블을 fantasyid(captured proxy)에서 평가 → private이 captured면 unseen-born-digital 카드가 전이되는지 확인 → 융합비 결정.

## ★★★ P3 결정적 — unseen FDA 앙상블은 CAPTURE-BLIND [2026-07-04]
fantasyid(captured proxy): unseen FDA 4-앙상블 **0.9904(chance)**, 개별 vl896=0.96/cx896=0.99/vl1120=0.98/cx1120=0.97. capture 카드(E6 896+1120)=0.3540. 융합에 unseen 넣을수록 악화(0.354→0.41→0.57→0.75).
**결론: unseen-type 강화(mean 0.0012)는 born-digital 축 — private=captured면 전이 안 됨(field-tamper와 동일 capture-blindness). private 진짜 병목 = capture 카드 0.354.**
전략 재조정: 다양성 축을 CAPTURE 카드(E6 DTC 라인)에 집중. 단 caveat: fantasyid는 cross-corpus 하한선 + private가 born-digital-unseen 혼합이면 unseen도 일부 유효 → capture-primary + unseen 소량 hedge(라우팅) 검토.
2-트랙 유지: public=field-tamper 앙상블(0.00039 1위, 확정) / private=capture-primary(E6).

## ★★ 앙상블 최소화: 12→3 모델, rank 1 유지 [2026-07-04]
사용자 요청(≤3 모델). 12모델(0.00039) 중 ViT-L@896 시드 6개는 상관 0.996-0.998 = 중복. 다양성 축 대표 3개로 압축:
ft3model = (ConvNeXt-V2@896 + ViT-Giant@896 + ViT-L@1120)/3 [CNN·용량·해상도 축] → public **0.00041** (12모델 0.00039 대비 +0.00002뿐, 2위 0.00053 대비 여전히 rank 1).
결론: 다양성은 축에서 나오지 시드 수가 아님을 재확인. **최종 public 카드 = 3-모델**(재현성·논문 명료성). 12모델은 0.00039로 kaggle best 유지되나 배포/기술문서는 3모델.
