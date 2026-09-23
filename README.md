# thptqg26-analyst

Nationwide examination score distribution analysis and dashboard for Vietnam's THPT 2026 exam reform.

This project is mostly used for education *and my school program*. While this contains support from AI, I still warn you that this contains *heavy* bad codes and should not be proceeded to look at.

---

## What is this

An analysis project for Vietnam's **2026 High School Graduation Examination (THPTQG 2026)** under the **GDPT 2018 General Education Program** curriculum reform.

It extracts candidate scores, maps candidate registration numbers (`sbd`) to 63 provinces and regions (Miền Bắc, Miền Trung, Miền Nam), generates subject score distribution charts, calculates university admission block totals (A00, A01, B00, C00, D01), and extracts top candidate lists (Thủ khoa).

---

## Source

- Raw candidate scores are loaded from `data/raw_diem_thpt_2026.csv`.
- The dataset is mirrored from public examination lookup mirrors released following the 2026 exam results.

---


## Deploy & Run

### Local Setup

Install requirements:

```bash
pip install -r requirements.txt
```

Run the pipeline:

```bash
python run_analysis.py
```

Launch the notebook:

```bash
jupyter notebook THPTQG_2026_Nationwide_Score_Distribution_Analysis.ipynb
```

### GitHub Actions

A GitHub Actions workflow (`.github/workflows/analyze.yml`) automatically runs whenever changes are pushed to `eoleun` (or another branch of your choice, if you’re handling the setup, as I assume you understand what I’m working on). It executes `run_analysis.py`, generates the updated Jupyter Notebook, and uploads the resulting charts and CSV reports as GitHub Actions artifacts.

## Credits

- **[anhdung98/diem_thi_2026](https://github.com/anhdung98/diem_thi_2026)** for providing the raw THPT 2026 examination score dataset.
