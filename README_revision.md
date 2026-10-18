# Revision artifact — Sudoku-Guided GF(2^8) SPD image encryption (IJIES 20265806)

Regenerates every table (2–14) and figure (2–10) of the revised manuscript.

## Contents
* `code/sudoku_gf_cipher.py` — reference implementation (authors' file; changes: closed the module docstring so it imports, corrected `key_space_report()`).
* `code/spd_ext.py` — bit-identical compiled diffusion (Numba), hardened SPD-N, random-selection baseline, chosen-plaintext attack.
* `code/metrics.py` — entropy, correlation, PSNR/SSIM, NPCR/UACI, Wu et al. critical values.
* `code/run_experiments.py` (Tables 2–7, 10, 12, 13), `code/run_extended.py` (Table 9, 11), `code/run_attack.py` (Table 14),
  `code/summarize.py`, `code/make_paper_figs.py` (Figs. 2–8), `code/make_figs.py` (Figs. 9–10).
* `results/*.json` — machine-readable per-image outputs incl. every perturbation location and per-trial NPCR/UACI; `results/figs/`.

## Data
* `imgs/core/`: baboon (512×512 Mandrill from the repo below, Lanczos-resized to 256×256), couple, house, airplane, boats, barbara (256×256 versions supplied by the authors).
* 512×512 images: https://github.com/mohammadimtiazz/standard-test-images-for-Image-Processing (clone to /home/claude/data/sti or edit paths).
* Kodak 24: https://github.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite (clone to /home/claude/data/kodak).

## Run
pip install numpy scipy scikit-image matplotlib numba pillow
python3 code/run_experiments.py && python3 code/run_extended.py && python3 code/run_attack.py
python3 code/summarize.py && python3 code/make_paper_figs.py && python3 code/make_figs.py

Test key K = SHA-256("IJIES-20265806-revision-test-key"); experiment seed 20265806.
