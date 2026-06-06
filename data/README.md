# Held-out data

`holdout.txt` is the held-out long text used for offline perplexity and as filler for
the needle-in-a-haystack test. It is **downloaded, not committed** (gitignored).

Fetch it with:

```bash
python scripts/get_data.py
```

This pulls a public-domain Project Gutenberg book (Moby-Dick) and strips the Gutenberg
header/footer. Any sufficiently long UTF-8 text file at `data/holdout.txt` works as a
drop-in replacement.

Caveat for the interview: this is public-domain text that may appear in the model's
pretraining corpus, so absolute perplexity is a relative baseline for measuring the
*delta* introduced by compression, not a claim of unseen-data generalization.
