# SIH 2026 Submission Checklist

This repository follows the structure published in the [NSUT SIH 2026 reference repository](https://github.com/NSUT-SIH-26/NSUT-SIH-DEMO).

## Repository content

- [x] Actual source code is present.
- [x] `README.md` includes PS ID, full title, problem, solution, features and technology stack.
- [x] Setup, launch and verification commands are documented.
- [x] Team responsibilities are listed without inventing member names.
- [x] Important prototype screenshots are stored under `assets/screenshots/`.
- [x] Architecture and technical documentation are stored under `docs/`.
- [x] Presentation and demo-video status pages exist under `submission/`.
- [x] Secrets, virtual environments, caches and generated run databases are ignored.
- [ ] Replace the presentation status with the final SIH PPT/PPTX before submission.
- [ ] Record the final demo video and replace the status in `submission/DEMO.md`.
- [ ] Verify repository and external links while logged out or in a private browser.
- [ ] Replace member numbers with final names when the team roster is confirmed.

## Final pre-submission verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
cd frontend
npm ci
npm run build
```

Then launch `launch_simulation.cmd`, execute at least one `DEMO_*` scenario, export its evidence and confirm every image and relative link in the root README loads on GitHub.

## Do not commit

- passwords, tokens, API keys or private credentials
- `.env` files containing secrets
- `.venv`, `node_modules`, caches or local SQLite run history
- misleading claims that experimental AI or unimplemented hardware is operational
