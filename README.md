# RHT Comparative Analysis

### Fluorescence intensity profiling of retinohypothalamic tract axon terminals in the SCN (*Opn4* WT vs KO)

Anterograde tracing study comparing retinohypothalamic tract (RHT) innervation of the
suprachiasmatic nucleus (SCN) between wild-type and melanopsin (*Opn4*) knockout mice.
Each animal received a dual-eye injection of spectrally distinct tracers, so the red and green
channels report the projections of the two eyes independently within the same section.

**Design:** WT (N = 10) vs KO (N = 10), one coronal SCN section per animal, two fluorescence
channels per section.

**Result:** *Opn4* knockout reduces the **ipsilateral** RHT projection to the SCN, in **both**
channels (red: -19.2%, p = 0.0046, d = 1.49, power = 0.88; green: -11.7%, p = 0.0113, d = 1.42,
power = 0.85). The **contralateral** RHT projection is normal in both channels. A descriptive
by-sex breakdown of the same data is provided in Section 13 of the notebook (no statistics; see
below).

## Start here

Open `RHT_Comparative_Analysis.ipynb`. Sections 1-12 (the pooled analysis) are committed with all
outputs executed, so GitHub renders every figure, table and statistic in the browser with no setup
required. Section 13 (the by-sex breakdown) renders once its inputs are present in
`data/07_By_Sex/`; until then it prints a short notice and skips its figures without error. See
"The pipeline" below for how to generate those inputs.

To run it yourself:

```bash
pip install -r requirements.txt
jupyter notebook RHT_Comparative_Analysis.ipynb
```

The notebook reads only from `data/` and performs no image processing, so it runs in seconds and
needs no MATLAB license.

## Coordinate convention

After alignment, **x = 0 is the SCN midline**; negative x is the left half of the image, positive
x the right half.

Left and right are *image coordinates*. Ipsilateral and contralateral are defined relative to the
*injected eye*. Because the two eyes were injected with different tracers, each eye's ipsilateral
SCN lobe lies on its own side, so the mapping from image half to laterality is channel-specific:

| Channel | Ipsilateral half | Contralateral half |
|---------|------------------|--------------------|
| Red     | x < 0            | x > 0              |
| Green   | x > 0            | x < 0              |

This mapping is declared once in each half of the project - `config.ipsiHalf` in the MATLAB
pipeline, and `IPSI_HALF` in the notebook - and every figure, table and methods statement derives
its laterality labels from it.

## Contents

```
├── RHT_Comparative_Analysis.ipynb   Results notebook: full analysis, QC, statistics, interpretation
├── data/                            Outputs of the MATLAB pipeline (read by the notebook)
│   ├── 01_Linear_Stretch/           Normalized image montages
│   ├── 02_Histograms/               Per-animal 1D intensity profiles (unaligned)
│   ├── 03_Alignment_with_Stats/
│   │   ├── QC_Plots/                Valley detection and alignment QC
│   │   ├── Before_After_Comparison/ Profiles before vs after alignment
│   │   ├── Outliers/                Largest deviations from each group mean
│   │   ├── Aligned_Profiles/        WT vs KO aligned group profiles
│   │   └── Statistical_Analysis/    Profile CSVs, Mann-Whitney results, per-animal stats plots
│   ├── 05_Heatmaps/                 WT - KO spatial difference maps
│   ├── 06_Average_Images/           Aligned group-average images (global intensity scale)
│   └── 07_By_Sex/                   By-sex descriptive outputs (profiles, averages, difference maps)
├── matlab/                          The analysis pipeline that produced data/
├── notebook_source/                 Script that generates the notebook from data/
├── requirements.txt                 To run the notebook
└── requirements-dev.txt             To regenerate the notebook
```

## The pipeline

`matlab/RHT_Comparative_Analysis_SpeerLab.m` runs in four steps, each independently toggleable
via `config.runStep1/2/3/4`:

1. **Normalization** - linear histogram stretch (1st-99th percentile) per genotype and channel.
2. **1D profiling** - each image collapsed to an intensity profile across the medial-lateral
   axis, at an 80 px bin for alignment and a 20 px bin for analysis.
3. **Alignment and statistics** - the midline valley is detected and every section shifted to a
   common origin; each animal contributes one mean-intensity value per half; WT and KO are
   compared by Mann-Whitney U with Cohen's d and post-hoc power.
4. **By-sex descriptive analysis** - the aligned data from Step 3 is regrouped into WT-F, WT-M,
   KO-F and KO-M (nothing is re-aligned) to produce four-group profile overlays and per-cell
   average images on the pooled global intensity scale. No statistics are computed; the groups
   are small by design. Output goes to `07_By_Sex/`. This step reads `normalized_data.mat`, so it
   must run in the same pass as Steps 1-3 rather than against a previously saved output folder.

**Filenames.** Sex is encoded as a single letter (`M` or `F`) immediately after the animal number,
e.g. `WT_animal3M_red.tif`. Filenames without the letter still load and still appear in the pooled
analysis; they are simply omitted from the by-sex breakdown. This means Steps 1-3 and their outputs
are identical whether or not sex is encoded - the pooled analysis is unaffected by this addition.

To reproduce `data/`, set `config.imgDir` to the raw `*_red.tif` / `*_green.tif` section images
and `config.outRoot` to the desired output folder, then run. The pipeline regenerates every
figure, statistic and methods document.

The following outputs are not committed here, to keep the repository within GitHub's size limits.
All are regenerated by re-running the pipeline:

- `normalized_data.mat`, `histogram_data.mat`, `aligned_data_with_stats.mat`,
  `average_images_data.mat`, `by_sex_average_data.mat` (intermediate state)
- `04_Shifted_Images/` (per-animal shifted sections)
- `01_Linear_Stretch/{WT,KO}_{Red,Green}/` (per-animal normalized sections; montages are kept)
- `Documentation/` (methods text emitted by the pipeline; the notebook's Methods section is the
  reference version)
- `.fig` and `.eps` versions of each figure, and the uncompressed difference-map TIFs
  (the `.png` version of every figure is kept)

## Citation

See `CITATION.cff`. Code is MIT licensed; the contents of `data/` are CC BY 4.0. See `LICENSE.md`.
