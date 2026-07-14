# Reproducibility reply — pinned Kaggle discussion thread

Post EXACTLY ONE top-level reply on the pinned Kaggle discussion thread by **2026-07-15 23:59 AoE**
(same as the final Kaggle submission deadline). No follow-ups, no separate threads. Keep it short —
the README already covers Docker build/run, weights, external data, and hardware.

Paste the block below (plain text) with the [bracketed] fields filled in. It matches the official
`REPLY_TEMPLATE.txt` verbatim.

---
FREUID Challenge 2026 - Reproducibility Package
---

Team name: MobilityAI
Kaggle usernames: suyongan
Final Kaggle submission: submission_full.csv (2026-07-14 00:09:56 UTC, ref 54665177)

Repository (this should be public git repository): https://github.com/hoppery/freuid-challenge-2026
Commit SHA: 993dbc25e4dcf684eeac1c1a4b7b59447a3b754d
Technical report (PDF): https://github.com/hoppery/freuid-challenge-2026/blob/master/docs/technical_report.pdf

We confirm this repository at the stated commit reproduces our selected final
submission and complies with the competition rules.

Signed (team captain): suyongan
Date (UTC): 2026-07-14

---

## Fill-in notes (do NOT post these)
- **Repository URL**: the public GitHub repo after `git push` (target `github.com/hoppery/freuid-challenge-2026`).
- **Commit SHA**: FILLED to `993dbc25e4dcf684eeac1c1a4b7b59447a3b754d`, the final frozen commit (code + Docker +
  weights manifest + finalized technical report). The rules require the solution/weights in the public repo by
  2026-07-13 and forbid weight/architecture/training changes after; documentation/packaging commits after are
  allowed if weights are unchanged. This SHA builds the exact submitted Docker and contains the final report.
- **Technical report URL**: already set to the GitHub blob URL of `docs/technical_report.pdf` on `master`
  (renders in-browser). Swap to a SHA permalink if you prefer to pin it to the exact frozen commit.
- **Final Kaggle submission**: FILLED = submission_full.csv (2026-07-14 00:09:56 UTC, ref 54665177), public 0.01853.
  ** You MUST still SELECT this submission as final in the Kaggle UI ** (Submissions page). If you do not,
  Kaggle auto-selects your best PUBLIC score, which is a different (field-tamper) model, not this one.
- **Date (UTC)**: set to 2026-07-14 (UTC). Update if you post on a different UTC day.
- Reply = this block only; organizers follow the repo README, not a second copy on the forum.
