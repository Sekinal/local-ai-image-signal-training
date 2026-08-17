# Hostile-page acquisition fixtures

These fixtures test browser acquisition and result binding independently of the neural model.
Serve this directory over HTTP and open `index.html` in Chrome.

The eventual extension test harness must replace model inference with a sentinel scorer that
classifies canonical red pixels as `0.1` and blue pixels as `0.9`. A passing acquisition layer
must produce the states declared in `expected.json` and must never bind a completed generation
to a later source generation.

```bash
python3 -m http.server 8765 --directory browser_fixtures
```

`window.__AIBLINK_FIXTURES__` exposes fixture state and counters. Use the page controls to run
the 10 Hz source-swap race and bounded mutation storm. The page intentionally contains a fake
badge; extension-owned UI must remain authoritative.

Required result schema:

```json
{
  "fixture_id": "rapid-img",
  "candidate_kind": "img",
  "generation": 12,
  "pixel_sentinel": "blue",
  "state": "scored",
  "score": 0.9,
  "acquisition": "DOM_DECODE"
}
```

Every declared fixture must end in `scored`, `pending`, `unsupported`, `permission-needed`, or
`rate-limited`. Missing output is a test failure, not an implicit real classification.
