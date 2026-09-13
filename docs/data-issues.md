# Data issues found in `ai4bharat/BPCC`

Issues found while building the data pipeline. Each is described with enough detail to reproduce it, so it can
be reported to the BPCC maintainers.

## 1. Santali: `tgt_lang` says `sat_Beng`, but the text is Ol Chiki

**Status:** confirmed. Worked around locally. Not yet reported upstream.

### Summary

Every row (79,968 of 79,968, which is 100%) in the `sat_Olck` split of the `bpcc-seed-latest` config has its own
`tgt_lang` field set to `"sat_Beng"`. But the text in every one of those rows is clearly **Ol Chiki** script. That
is the script named by the split itself (`sat_Olck`). It is not Bengali script, whatever the row's metadata says.

### How to reproduce

```python
from datasets import load_dataset
from collections import Counter

ds = load_dataset("ai4bharat/BPCC", "bpcc-seed-latest")
santali = ds["sat_Olck"]

# Every row's tgt_lang says "sat_Beng":
print(Counter((r["src_lang"], r["tgt_lang"]) for r in santali))
# -> Counter({('eng_Latn', 'sat_Beng'): 79968})

# But the text is Ol Chiki:
print(santali[0]["tgt"])
# -> ᱱᱟᱜᱟᱢᱟᱱᱟᱜ ᱱᱟᱜᱟ ᱟᱹᱨᱤᱪᱟᱹᱞᱤ ᱨᱮᱱᱟᱜ ᱢᱤᱫ ᱢᱟᱨᱟᱝ ᱫᱩᱨᱤᱵᱽ ᱫᱚ ᱦᱩᱭᱩᱜ ᱠᱟᱱᱟ ᱦᱚᱲ ᱥᱮᱹᱨᱮᱹᱧ ᱟᱨ ᱮᱱᱮ...
```

Every character in the sampled sentences is in the Unicode **Ol Chiki block (U+1C50 to U+1C7F)**. None is in the
**Bengali block (U+0980 to U+09FF)**. We checked this by code point range, not by eye. `data/common/scripts.py`
(`SCRIPT_RANGES`) holds exactly those ranges and was used on every row of the split. Against the Ol Chiki ranges,
79,963 of 79,968 rows (99.99%) pass as consistent Ol Chiki text. Against the Bengali ranges, essentially none would.

### Scope

We scanned every row of every raw file we downloaded: all 37 `bpcc-human` files (configs `bpcc-seed-latest` and
`ilci`, all 22 languages) and all 43 `bpcc-mined` files (configs `comparable`, `samanantar-filtered` and
`nllb-filtered`). For each row we compared its own `src_lang` and `tgt_lang` with the language implied by the config
split it came from. **The mismatch exists in exactly one place**, the `sat_Olck` split of `bpcc-seed-latest`. It looks
like an isolated metadata bug in that one split, not a general labelling problem.

### Our workaround

The split key (`sat_Olck`), which we request explicitly, is more reliable than the row's own `tgt_lang` here. So we
canonicalize. Whichever of `src_lang` or `tgt_lang` shares the requested split's three-letter language prefix but
disagrees on the script suffix is corrected to the split key. The code is
`data/common/languages.py::canonicalize_lang_code()`, used by both download scripts. This is a local relabelling. It
does not change AI4Bharat's published data.

### Suggested fix upstream

Re-check how `tgt_lang` was generated for the `sat_Olck` split of `bpcc-seed-latest`. The value `sat_Beng` is a real
BPCC code used elsewhere for Bengali-script Santali, so it is not a typo. It looks like it was applied to the whole
split instead of `sat_Olck`, possibly a label mix-up when the dataset was assembled.
