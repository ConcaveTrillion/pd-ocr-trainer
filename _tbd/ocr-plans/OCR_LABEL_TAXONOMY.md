# OCR Label Taxonomy and Normalization

This file defines canonical label spaces and alias policy for word, line, and block classification.

## 1) Word text style labels

### Canonical labels

- all caps
- small caps
- italics
- bold
- blackletter
- has starting footnote marker
- has ending footnote marker
- underline
- strikethrough
- superscript
- subscript
- has drop cap
- monospace
- handwritten

### Notes

- Multi-select is allowed (example: `bold + italics`).
- Canonical values should be stored in normalized lowercase form.

### Alias examples to normalize

- `italic`, `ital` -> `italics`
- `drop cap` -> `has drop cap`
- `starting footnote marker`, `start footnote marker` -> `has starting footnote marker`
- `ending footnote marker`, `end footnote marker` -> `has ending footnote marker`
- `typewriter`, `monospaced` -> `monospace`

### Input normalization rules

1. Lowercase and trim whitespace.
2. Convert underscore/hyphen separators to spaces.
3. Collapse repeated spaces.
4. Resolve aliases.
5. Support compact no-space matching (`allcaps`, `smallcaps`, etc.).
6. Reject unknown labels with explicit errors.

## 2) Block-level labels

### Block role labels (semantic type)

- paragraph
- sidenote
- page header
- page footer
- page number
- printers mark
- blockquote
- poetry

### Block position labels (where on page)

- top
- bottom
- left
- right
- center
- margin left
- margin right

## 3) Line-level labels

### Line role labels

- body line
- heading line
- verse line
- blockquote line
- header line
- footer line
- footnote line
- caption line
- page number line

### Line position labels

- top
- bottom
- left
- right
- center
- column left
- column right

## 4) Additional labels worth considering

### Block candidates

- epigraph
- section heading
- chapter title
- footnote block
- figure caption
- table caption
- list
- table region
- figure region
- ornamental divider
- marginal note

### Line candidates

- continuation line
- list item line
- signature line

## 5) Role vs position principle

Keep role and position as separate label spaces.

Do:
- role = `page number`
- position = `top`, `right`

Avoid:
- combined classes like `top-right-page-number`.

This keeps annotation simpler, reduces class explosion, and improves model generalization.

## 6) Provenance recommendation

For each predicted label, track provenance:

- manual
- rule
- model
- fused

Optionally include confidence and model version for reproducibility.
