# Reproducibility reply — pinned Kaggle discussion thread

Post EXACTLY ONE structured reply on the pinned reproducibility thread by **2026-07-15 23:59 AoE**.
Fill the bracketed fields, then paste as the reply.

---

**Team name:** [your Kaggle team name]
**Kaggle usernames:** [user1, user2, ...]
**Final Kaggle submission label / date-time:** [e.g. submission_ft3model.csv — 2026-07-06 14:30 UTC (public); private = CAPTURE-card Docker]
**Repository URL (public, OSI-licensed):** [https://github.com/<org>/<repo>]
**Commit SHA (40-char, frozen):** [<full 40-character commit hash after code freeze on/before 2026-07-13>]
**Technical report (PDF) URL:** [link to docs/technical_report.pdf in the repo, or a hosted copy]

**Compliance confirmation:** We confirm the repository is public under an OSI-approved license (MIT),
frozen at the commit SHA above with no model-weight or training-code changes after the code-freeze date;
the Docker image runs with `--network none` (all weights embedded, no runtime downloads) and writes only
to `/submissions/submission.csv`; and all reported results are reproducible from the listed commands.

**Team captain:** [name] · **Date (UTC):** [YYYY-MM-DD]

---

## Remaining actions before posting (only these need a human)

1. **Freeze + push the code** to a public git repo (no remote is configured yet), on/before **2026-07-13**.
   Capture the 40-char commit SHA. Do NOT change model weights / training code after the freeze.
2. **Build + verify the Docker** on your hardware (Docker is not installed in the dev sandbox):
   ```bash
   docker build -t freuid-repro:local .
   mkdir -p out && docker run --network none -v /path/to/test/images:/data:ro -v "$(pwd)/out":/submissions freuid-repro:local
   head out/submission.csv
   ```
   If the build's CUDA differs from this host, adjust the `Dockerfile` base tag + `torch --index-url`.
3. **Host the technical report PDF** (`docs/technical_report.pdf`) at a stable URL (repo or elsewhere).
4. **Fill the bracketed fields** above and post the single reply.
