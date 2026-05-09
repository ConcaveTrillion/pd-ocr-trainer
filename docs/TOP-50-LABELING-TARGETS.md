# Top 50 Pages to Label/Train Next

A prioritized labeling shortlist drawn from `source-pgdp-data/output/`,
chosen to maximise OCR/layout-model improvement per page given what's
already in `pd-ocr-trainer/matched-ocr/` (53 pages so far).

## How this list was built

1. Hashed every PNG in `source-pgdp-data/output/` against
   `pd-ocr-trainer/matched-ocr/` so already-labeled pages are excluded.
2. Mined `pages.json` for high-value PGDP markers — chapter heads,
   accented/foreign text, transcriber notes (`[**...]`), super/subscripts,
   leader-dot rows, anchored footnote markers, blockquoted poetry,
   running-header all-caps lines.
3. Visually verified a sample from every project to confirm what the
   text mining could not see (decorative headpieces, sidenotes, drop
   caps — none of which are tagged in the PGDP source text for these
   books).
4. Cross-referenced the failure cases listed in
   `SPEC-layout-training.md` §1.1 (cursive drop caps S/O/A, page numbers,
   chapter headers, decorative rules, sidenotes, footnote markers vs
   text, marginalia).

## Already-labeled, excluded from this list

| Project | Pages already in `matched-ocr/` |
|---|---|
| `projectID629292e7559a8` Wilson, *History of the American People III* | `038, 039` |
| `projectID63ac6757567bd` Jones, *Credulities Past and Present* | `p0220, p0230, p0240` |
| `projectID63ac684a641d4` Russell, *A Visit to Chile* | `p0110–p0370` (28 consecutive body pages) |
| `projectID6737b15d33ff3` Singer, *From Magic to Science* | `f000–f014, p009–p011, p025` |
| `projectID67658de495d0c` *Book of Filial Duty* | `017, 018` |
| `projectID66c62fca99a93` Dilke, *French Furniture* | none — book is **completely unlabeled** |

The Dilke gap is significant: that book has the densest set of
high-value training features in the corpus (decorative headpiece +
drop cap + sidenote + footnotes + French + small-caps title on every
chapter opening). Six of those single pages buy a lot of coverage.

## Distribution across training needs

| Bucket | Pages | Notes |
|---|---|---|
| Drop-cap chapter openings | 14 | mix of plain block, cursive italic, italic French, small-caps continuation |
| Multi-feature chapter pages (drop cap + sidenote + headpiece + footnotes) | covered above (Dilke) | — |
| Plate/figure caption pages | 6 | Singer mostly — figure-region + caption + label detection |
| Foreign script + accented text bodies | 6 | French (Dilke), Spanish (Russell), Greek/Latin (Singer) |
| Footnote-anchor / footnote-text body | 5 | Singer + Dilke body pages |
| Frontmatter / title / TOC / dedication | 6 | leader dots, ornamental titles, all-caps display type |
| Running-header + page-number variants | 5 | mixed convention across the 6 books |
| Poetry / blockquote / inset typography | 4 | smaller font, indented runs |
| Transcriber-note clusters (`[**...]`) | 4 | hard-glyph cases the labeler should fix |

A single page often hits 3+ buckets; counts above are the **primary**
reason it's on the list.

## The 50 — ranked by training value

Convention: **proj** = project ID suffix; **page** = filename in
`source-pgdp-data/output/<projectID...>/`. Rationale calls out the
*specific* OCR/layout weakness the page exercises, with reference to
`SPEC-layout-training.md` §1.1 failure modes where applicable.

### Tier 1 — Dilke chapter openings (multi-feature gold pages)

Each one combines a **decorative headpiece**, a **4–5-line drop cap**,
a **left-margin sidenote** (untagged in PGDP source — must be drawn by
labeler), **CHAPTER + small-caps title**, **footnote anchors with
footnote-text region at page foot**, **st-ligature characters
(`architecture` etc.)** and **French text with diacritics**. These six
alone hit every category called out in `SPEC-layout-training.md` §1.1.

| # | proj | page | drop-cap | rationale |
|---|------|------|---------|-----------|
| 1 | 66c62fca99a93 | `029.png` | **D** (4-line) | Chapter I opener; sidenote `The "Golden Gallery"…`; catchword `B`; Roman numeral page `I` |
| 2 | 66c62fca99a93 | `060.png` | **I** (5-line) | Chapter II opener; 7 numbered footnotes with mixed-language text |
| 3 | 66c62fca99a93 | `136.png` | **T** (4-line) | Chapter V opener; multi-line sidenote, French quoted block |
| 4 | 66c62fca99a93 | `225.png` | **M** (italic, "MA FEMME") | Chapter IX — *italic French* drop-cap variant; rare style |
| 5 | 66c62fca99a93 | `258.png` | **N** (4-line) | Chapter X — long French footnote with multiple italic titles |
| 6 | 66c62fca99a93 | `333.png` | **T** (4-line) | Chapter XII — heavy use of `"ébénistes"` quotes, sidenote, footnotes |

### Tier 2 — drop-cap regressions called out in TODO

These directly target the cursive-cap stitcher failures
(`chapter-head-credulities` S, `chapter-head-filial-duty` O,
`footnotes-stacked-with-anchor` A) listed in `SPEC-layout-training.md` §1.1.

| # | proj | page | drop-cap | rationale |
|---|------|------|---------|-----------|
| 7 | 63ac6757567bd | `p0010.png` | **S** (cursive italic) | Jones Chapter I — cursive S the geometric stitcher fails on; cursive italic title `Credulities Past & Present` above it for double benefit |
| 8 | 67658de495d0c | `016.png` | **O** (plain block, hollow) | Filial Duty Chapter I — the classic O-shaped drop cap that confuses contour fallback |
| 9 | 63ac6757567bd | `p1200.png` | **F** (block) | Jones Chapter II — different glyph, plain block style; useful as the "easy positive" baseline |
| 10 | 63ac6757567bd | `p1520.png` | (block + transcriber note) | Jones Chapter III — chapter title contains a `[**comma, or mark on page?]` transcriber note inline (rare) |
| 11 | 63ac6757567bd | `p1950.png` | (block) | Jones Chapter IV |
| 12 | 63ac6757567bd | `p2200.png` | (block) | Jones Chapter V |
| 13 | 63ac6757567bd | `p2900.png` | (block) | Jones Chapter VII |
| 14 | 67658de495d0c | `023.png` | (block) | Filial Duty Chapter IX — different chapter glyph than 016 |

### Tier 3 — Plate / figure pages (caption + label detection)

Singer's plate pages are pure figure + multi-line caption. They give
the layout model crisp positive examples for `figure` and
`figure_caption` regions, and the caption text itself uses small-caps
display type the recognition model rarely sees.

| # | proj | page | rationale |
|---|------|------|-----------|
| 15 | 6737b15d33ff3 | `p064a.png` | Tenth-century zodiacal scheme — Latin caption with italic phrases, two-line title |
| 16 | 6737b15d33ff3 | `p102a.png` | Frontispiece-style figure with two-line italic caption |
| 17 | 6737b15d33ff3 | `p112a.png` | **Multiple captions on one page** (Figs 47/48/49) — exercises caption-region segmentation |
| 18 | 6737b15d33ff3 | `p184a.png` | Figure caption containing an inline `[**-?]` transcriber note |
| 19 | 6737b15d33ff3 | `p140a.png` | Plate IV — Anglo-Saxon herbal page; mixed scripts in caption |
| 20 | 63ac684a641d4 | `p1741.png` | Russell plate caption — `Oficina. Liverpool Nitrate Company. [To face page 174.` — small-caps + bracket attribution |

### Tier 4 — Frontmatter (title, TOC, dedication, list-of-figures)

These are layout-distinct from body pages and currently
under-represented. The model needs to know that a TOC is **not** a
body region.

| # | proj | page | rationale |
|---|------|------|-----------|
| 21 | 66c62fca99a93 | `005.png` | Dilke title page — `Châtéau de Fontainebleau` caption, mixed display sizes, accented text |
| 22 | 63ac6757567bd | `a0030.png` | Jones half-title with chapter list — narrow column, all-caps |
| 23 | 63ac684a641d4 | `a0070.png` | Russell **Contents** — leader dots of varying length, page numbers in right column (per TODO: model misses page-number columns) |
| 24 | 6737b15d33ff3 | `f015.png` | Singer Contents — Roman-numeral chapter column + leader dots + page numbers |
| 25 | 6737b15d33ff3 | `f017.png` | Singer **List of Figures** — long fig-name + page leader dots; tests two-column figure index layout |
| 26 | 67658de495d0c | `005.png` | Filial Duty Contents — small page count, simple leader-dot layout (clean positive baseline) |

### Tier 5 — Foreign script and accented body text

The recognition model is weakest on Greek inline text, French
diacritics in continuous prose, and Spanish proper nouns. Pick one
strong example per script.

| # | proj | page | rationale |
|---|------|------|-----------|
| 27 | 6737b15d33ff3 | `p050.png` | Singer body — **inline Greek** (τοῦ γὰρ καὶ γένος ἐσμέν), italic Latin titles, § section marker, B.C./A.D. small caps |
| 28 | 6737b15d33ff3 | `p129.png` | Singer body — **Anglo-Saxon characters** (`næsdhyrel`, ᵹ etc.), Greek glosses, mixed scholarly notation |
| 29 | 6737b15d33ff3 | `p127.png` | Singer body — Greek + footnote marker on same line |
| 30 | 66c62fca99a93 | `211.png` | Dilke body — long French quoted passage with `é à î ç` diacritics; transcriber-note `[** typo: made by]` mid-line |
| 31 | 63ac684a641d4 | `p0670.png` | Russell body — Spanish place-names with accents (Tarapacá, Iquique), em-dashed compound words |
| 32 | 66c62fca99a93 | `247.png` | Dilke poetry/quoted French — multiple stanzas in smaller font |

### Tier 6 — Footnotes (anchors in body + footnote-text region)

`SPEC-layout-training.md` §1.1 flags footnote-marker vs footnote-text
confusion. Need pages where the model has to draw the **horizontal
rule + smaller-font region** as a separate `footnote` block, not body.

| # | proj | page | rationale |
|---|------|------|-----------|
| 33 | 6737b15d33ff3 | `p119.png` | Singer body — footnote anchor `^1`, footnote text region with multi-line italic-title content |
| 34 | 6737b15d33ff3 | `p123.png` | Singer body — multiple footnote anchors clustered |
| 35 | 66c62fca99a93 | `184.png` | Dilke Chapter VII opener — Tier-1-style page also; chosen here because the footnote block is *unusually long* (5+ entries) |
| 36 | 66c62fca99a93 | `077.png` | Dilke body — heaviest footnote density in book; multi-paragraph footnote text with internal italic |
| 37 | 66c62fca99a93 | `048.png` | Dilke body — footnotes including superscript date markers (`^{1}`, `_{2}` style) |

### Tier 7 — Running-header + page-number variants

Each book uses a different convention; the model needs all of them so
header detection generalises rather than overfits Russell's two-column
synopsis style (which is over-represented in the current 53-page set).

| # | proj | page | rationale |
|---|------|------|-----------|
| 38 | 67658de495d0c | `039.png` | Filial Duty body — `CLAD IN A SINGLE GARMENT  39` header; **italic centered subtitle** mid-page (`No. V / He carried Rice for his Parents`) |
| 39 | 629292e7559a8 | `099.png` | Wilson body — small-caps running header, Roman-numeral folio at foot, hyphenated word at column break |
| 40 | 629292e7559a8 | `270.png` | Wilson Chapter IV opener — `CHAPTER IV / CRITICAL CHANGES / A CRITICAL ...` — small-caps continuation **without** drop cap; useful negative-example for drop-cap detector |
| 41 | 63ac6757567bd | `p4410.png` | Jones body — italic running header `BIRDS.` + page number; **Carpathian-song poetry block** in smaller indented font |
| 42 | 6737b15d33ff3 | `p053.png` | Singer body — section-marker §, italic phrase headers in body, no header (chapter-end variant) |

### Tier 8 — Transcriber-note hot spots (`[**...]`)

Pages with multiple `[**...]` clusters are where the source scan is
hardest to read; labelling them seeds the recognition model with
ground-truth for the exact glyphs the labeler hand-corrected.

| # | proj | page | rationale |
|---|------|------|-----------|
| 43 | 66c62fca99a93 | `015.png` | Dilke contents-detail — `[** governour\|P3 governor p. 138 referenced]` and similar multi-option transcriber notes |
| 44 | 66c62fca99a93 | `031.png` | Dilke contents-detail — heaviest `[**...]` density in the book |
| 45 | 67658de495d0c | `004.png` | Filial Duty colophon — `[** or .]` ambiguity on a small all-caps printer mark |
| 46 | 629292e7559a8 | `043.png` | Wilson body — clustered `[**P3, ?]` markers |

### Tier 9 — Poetry / blockquote / inset typography

Inset blocks at smaller size are systematically missed (the geometric
reorg merges them into surrounding paragraphs).

| # | proj | page | rationale |
|---|------|------|-----------|
| 47 | 67658de495d0c | `006.png` | Filial Duty front-matter blockquote — short centred lines, italic attribution |
| 48 | 63ac6757567bd | `p0080.png` | Jones body — multi-quote blockquote with em-dashed sources |
| 49 | 6737b15d33ff3 | `p033.png` | Singer body — verse passage centred in narrower column |
| 50 | 66c62fca99a93 | `365.png` | Dilke index page — **multi-column index entries** with page-number runs (rare; tests column detection at end-matter) |

## Suggested labeling order

1. **Pages 1–6 (Dilke chapter openings)** first — every other improvement
   compounds once the model has clean positives for headpiece + drop
   cap + sidenote + footnotes together. Six pages, ~20 min each.
2. **Pages 7–14 (drop-cap regressions)** next — directly retire the
   `chapter-head-*` test failures.
3. **Pages 15–26 (plates + frontmatter)** — quick wins; mostly small
   text regions, low per-page time.
4. Cycle through Tiers 5–9 in any order; reserve Tier 8
   (transcriber-note pages) for a focused session because they need
   careful per-glyph attention.

Target: 50 pages over ~3 weeks at 15–25 min/page (per
`LABELING_STRATEGY.md`'s 30/60/90-min cadence). After page 25, retrain
and inspect the layout-regression fixtures to confirm the cursive
drop-cap and sidenote cases improve before committing to the rest.

## Pages worth reconsidering once the model improves

- **Russell `a0140.png`** — Map of Tarapacá; pure cartography. Useful
  later for a `figure`-type detector that distinguishes maps from
  illustrations, but skip until the basics are stable.
- **Singer `p064a.png` / `p202a.png` / `p214a.png`** — Hildegard
  visions; highly stylised manuscript reproductions with text in the
  image itself. Easy to mislabel; defer until the recogniser has a
  clear "image text vs page text" signal.
- **Dilke index (`365–367`)** — heavy multi-column run; do *one*
  index page (#50 above) first, then decide if more help.
