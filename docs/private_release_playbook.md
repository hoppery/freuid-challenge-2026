# PRIVATE RELEASE-DAY PLAYBOOK (마감 ~2026-07-15 23:59 AoE)

공식 절차(freuid2026.microblink.com): private test set이 공개되면 **자체 인프라에서 추론 → 예측 파일 제출**.
"exact flow soon" — 제출 채널(Kaggle 재제출 vs 별도 업로드)은 공지 대기. 일일 감시로 포착.

## ✅ RELEASE-READINESS 검증 완료 (2026-07-04)
- env 오버라이드 추가: `FREUID_TEST_DIR`(이미지 dir) + `FREUID_SAMPLE_SUB`(id 목록 csv) — 코드 수정 없이 private로 전환.
  두 라인 어댑터(src/ + _transfer/src/ freuid.py) 모두 반영. 동작 검증됨(가짜 dir→0, 기본→7821).
- capture 카드 앙상블 추론(ROOT DTC, exp_e6_fda+exp_e6_fda1120 prob-mean) → 142,818행 유효 CSV 생성 검증됨.
- 배포 스크립트 `scripts/deploy_infer.sh` 작성·검증 (2-트랙 CSV 생성).

## 0. 공개 감지 시 즉시
- [ ] `kaggle competitions files ...`로 새 디렉토리 확인 → 다운로드 (get_official.sh aria2 경로 사용)
- [ ] 이미지 수·형식 확인 + 새 sample_submission(private id 목록) 확보
- [ ] Kaggle Discussion + 공식 사이트에서 "exact flow" 공지 확인 (public/private id 분리 방식)
- [ ] 실행: `export FREUID_TEST_DIR=data/raw/freuid/<priv_img_dir>; export FREUID_SAMPLE_SUB=data/raw/freuid/<priv_sample_sub.csv>; bash scripts/deploy_infer.sh`
      → submission_deploy_capture.csv (capture, private-primary) + submission_deploy_ftens.csv (field-tamper, public/born-digital)

## 1. 최종 CSV 구성 (2-트랙)
- **public 7,821행** = `submission_ft12model.csv`의 해당 행 그대로 (public 1위 0.00039)
- **private 134,997행** = private 카드 추론 (아래 §2), P3 융합 실험 결과에 따라 구성
- 병합: id 기준 merge, 누락 0.5 fill 금지(전 행 실예측), 행수 142,818 검증

## 2. private 카드 (P3 실험으로 최종 확정; 현 기본값)
- **capture-primary**: `checkpoints/exp_e6_fda/epoch2.pt` (fantasyid proxy 0.412) — @1120 버전(exp_e6_fda1120)이 이기면 교체
- **unseen 앙상블**: `exp_fdab12_all/epoch2.pt`(ViT-L FDA) + `exp_cnxfda_all/epoch2.pt`(ConvNeXt FDA) (+@1120 배포판 완성 시 추가)
- 융합비: P3(fantasyid_test 실측)에서 결정 — E6-단독 vs E6×w + FDA-ens×(1-w)
- 추론: ROOT 라인은 `python3 -m freuid.infer`(DTC), _transfer 라인은 `scripts/infer_tta_ens_transfer.py`(hflip TTA)
  ⚠ private 이미지 디렉토리를 읽도록 build_freuid_test_index 확장 필요할 수 있음(경로 하드코딩 확인)

## 3. 제약·정책
- ⛔ test-데이터로 파라미터 적합 금지(TENT/pseudo-label — feedback_no_testdata_training). hflip TTA만.
- 제출 전 사용자 확인(제출 정책). GPU2 사용 금지.

## 4. 코드 공개 패키지 (수상 요건, 모델 동결 후)
- training/inference 스크립트 + configs 전부, OSI 라이선스(MIT 권장), short technical report
- 재현 지침: manifest 빌드 → 12모델 학습 config 목록 → ep2 선택 → TTA 앙상블 스크립트
