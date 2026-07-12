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
Kaggle usernames: [comma-separated Kaggle handles of all team members]
Final Kaggle submission: [label / date-time as shown on Kaggle]

Repository (this should be public git repository): https://github.com/hoppery/freuid-challenge-2026
Commit SHA: [40-char hash of the final HEAD — weights unchanged since the 2026-07-13 freeze]
Technical report (PDF): https://github.com/hoppery/freuid-challenge-2026/blob/master/docs/technical_report.pdf

We confirm this repository at the stated commit reproduces our selected final
submission and complies with the competition rules.

Signed (team captain): [Kaggle username]
Date (UTC): [YYYY-MM-DD]

---

## Fill-in notes (do NOT post these)
- **Repository URL**: the public GitHub repo after `git push` (target `github.com/hoppery/freuid-challenge-2026`).
- **Commit SHA**: `git rev-parse HEAD` of the final pushed commit; paste the full 40 chars. The rules require
  the solution/weights to be in the public repo by 2026-07-13 and forbid weight/architecture/training changes
  after; documentation and Docker packaging commits after 2026-07-13 are allowed as long as the weights are
  unchanged. So the declared SHA may post-date 07-13 (e.g. a report/packaging commit) — it need not be dated 07-13.
- **Technical report URL**: already set to the GitHub blob URL of `docs/technical_report.pdf` on `master`
  (renders in-browser). Swap to a SHA permalink if you prefer to pin it to the exact frozen commit.
- **Final Kaggle submission**: label/date-time of the selected final submission as shown on Kaggle.
- Reply = this block only; organizers follow the repo README, not a second copy on the forum.
