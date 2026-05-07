# perceptual — provenance

Three rendered PDF page images (grayscale PNG) used by
`tests/unit/test_dedup_perceptual.py` to validate
`PerceptualHashLevel` (the imagehash-based page-image dedup level).
Each image is a single-page render of a U.S. federal-government
document — public-domain by 17 USC §105.

Per `docs/oss/50-data-and-fixtures/provenance-policy.md`.

## Per-file provenance

| File | Source | License | Retrieved | SHA-256 |
|---|---|---|---|---|
| `page_gpo.png` | Single-page render of a U.S. Government Publishing Office report (rendered via kaos-pdf at default DPI). Sister fixture lives at `kaos-pdf/tests/fixtures/gpo_report.pdf`. | Public domain (US Government work, 17 USC §105) | 2026-05-04 | `b4dd89ba4721a61fbf766a97b879e84f13a141658860463dc50aaa153d63625b` |
| `page_staten.png` | Single-page render of *Staten v. United States*, a federal court opinion. Sister PDF at `kaos-pdf/tests/fixtures/staten_v_united_states.pdf`. | Public domain (US Government work) | 2026-05-04 | `a282ebe5bd71bb3625ba028d2f4ba46089599a50ed6aae87c8b05965731ff594` |
| `page_plaster.png` | Single-page render of a USPTO design patent (ornamental plaster design). | Public domain (US Government work) | 2026-05-04 | `2efecee4398c403e133be202eca869bce9a8dd23e8fd67444162719143b66cb6` |

## Why these specific pages

The perceptual-hash test exercises three properties:
1. **Identity** — re-encoding the same page (PNG → JPEG round-trip) must
   hash within the perceptual-hash sensitivity threshold.
2. **Cross-document distinctness** — pages from different documents
   (GPO report vs. court opinion vs. design patent) must hash to
   visibly different values.
3. **Cluster boundary** — small adversarial perturbations (mild blur,
   grayscale-to-RGB conversion) stay inside the cluster, while a
   different page (e.g., page_staten vs page_plaster) crosses out.

Each fixture is < 300 KB, grayscale, mid-resolution. Hand-picked from
the kaos-pdf fixtures because they have visibly different layouts (a
two-column GPO report, a single-column court opinion with a caption
banner, and a patent diagram).

## Re-rendering

```bash
# from the kaos-modules root
uv run --directory kaos-pdf python - <<'PY'
import kaos_pdf
for pdf, page_idx, out_name in [
    ("tests/fixtures/gpo_report.pdf", 0, "page_gpo.png"),
    ("tests/fixtures/staten_v_united_states.pdf", 0, "page_staten.png"),
    ("tests/fixtures/kl3m_court_burns.pdf", 0, "page_plaster.png"),
]:
    img = kaos_pdf.render_page(pdf, page_idx).convert("L")
    img.save(f"../kaos-content/tests/fixtures/perceptual/{out_name}")
PY
```

After re-rendering, update SHA-256 entries in this README.
