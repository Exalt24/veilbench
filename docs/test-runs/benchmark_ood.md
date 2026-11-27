# Detector benchmark on formats the statistical detector never trained on

Corpus: 300 generated records, 2350 labelled spans, 585 hard negatives, seed 17, split `ood`, sha256 `13598aea75424f3e`.

| configuration | precision | recall | F1 | F2 | span caught rate | records leaking (coverage) | records leaking (verbatim) | hard negatives flagged | p50 ms | p95 ms | compute s per 1,000 records | LLM USD per 1,000 records |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rules | 0.934 | 0.673 | 0.783 | 0.713 | 0.670 | 0.870 | 0.870 | 0.188 | 0.29 | 0.4 | 0.3 | 0.0 |
| rules+stat | 0.753 | 0.843 | 0.795 | 0.823 | 0.897 | 0.470 | 0.367 | 0.284 | 11.23 | 14.46 | 11.64 | 0.0 |
| rules+stat+spacy | 0.659 | 0.878 | 0.753 | 0.823 | 0.953 | 0.247 | 0.183 | 0.420 | 23.02 | 26.68 | 23.12 | 0.0 |
| rules+stat+gliner | 0.712 | 0.893 | 0.792 | 0.850 | 0.932 | 0.367 | 0.307 | 0.723 | 72.11 | 89.73 | 73.61 | 0.0 |
| spacy-only | 0.263 | 0.243 | 0.253 | 0.247 | 0.414 | 0.997 | 0.997 | 0.142 | 11.54 | 13.72 | 11.48 | 0.0 |
| gliner-only | 0.734 | 0.516 | 0.606 | 0.548 | 0.532 | 0.973 | 0.967 | 0.479 | 60.52 | 77.83 | 61.97 | 0.0 |

## Recall by entity type

| type | rules | rules+stat | rules+stat+spacy | rules+stat+gliner | spacy-only | gliner-only |
|---|---|---|---|---|---|---|
| PERSON | 0.226 | 0.603 | 0.695 | 0.722 | 0.603 | 0.347 |
| EMAIL | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.184 |
| PHONE | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.847 |
| ADDRESS | 1.000 | 1.000 | 1.000 | 1.000 | 0.080 | 0.832 |
| NATIONAL_ID | 0.988 | 0.988 | 0.988 | 0.988 | 0.000 | 0.750 |
| CREDIT_CARD | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.813 |
| IBAN | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| API_KEY | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.497 |
| JWT | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| PASSWORD | 0.000 | 0.841 | 0.841 | 1.000 | 0.000 | 0.710 |
| DATE_OF_BIRTH | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.550 |
| IP_ADDRESS | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |

## Precision by entity type

| type | rules | rules+stat | rules+stat+spacy | rules+stat+gliner | spacy-only | gliner-only |
|---|---|---|---|---|---|---|
| PERSON | 1.000 | 0.753 | 0.605 | 0.803 | 0.499 | 0.937 |
| EMAIL | 1.000 | 0.817 | 0.817 | 0.839 | 0.000 | 1.000 |
| PHONE | 0.704 | 0.704 | 0.704 | 0.680 | 0.000 | 0.932 |
| ADDRESS | 1.000 | 0.944 | 0.622 | 0.969 | 0.025 | 0.829 |
| NATIONAL_ID | 1.000 | 0.975 | 0.975 | 0.637 | 0.000 | 0.583 |
| CREDIT_CARD | 0.993 | 0.993 | 0.993 | 0.974 | 0.000 | 0.917 |
| IBAN | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| API_KEY | 1.000 | 0.606 | 0.606 | 0.606 | 0.000 | 0.822 |
| JWT | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| PASSWORD | 0.000 | 1.000 | 1.000 | 0.451 | 0.000 | 0.241 |
| DATE_OF_BIRTH | 1.000 | 0.328 | 0.328 | 0.258 | 0.000 | 0.317 |
| IP_ADDRESS | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |

## F2 by entity type

| type | rules | rules+stat | rules+stat+spacy | rules+stat+gliner | spacy-only | gliner-only |
|---|---|---|---|---|---|---|
| PERSON | 0.268 | 0.628 | 0.675 | 0.737 | 0.579 | 0.397 |
| EMAIL | 1.000 | 0.957 | 0.957 | 0.963 | 0.000 | 0.220 |
| PHONE | 0.922 | 0.922 | 0.922 | 0.914 | 0.000 | 0.863 |
| ADDRESS | 1.000 | 0.988 | 0.892 | 0.994 | 0.055 | 0.831 |
| NATIONAL_ID | 0.990 | 0.985 | 0.985 | 0.890 | 0.000 | 0.709 |
| CREDIT_CARD | 0.999 | 0.999 | 0.999 | 0.995 | 0.000 | 0.832 |
| IBAN | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| API_KEY | 1.000 | 0.885 | 0.885 | 0.885 | 0.000 | 0.539 |
| JWT | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| PASSWORD | 0.000 | 0.868 | 0.868 | 0.804 | 0.000 | 0.511 |
| DATE_OF_BIRTH | 1.000 | 0.709 | 0.709 | 0.635 | 0.000 | 0.479 |
| IP_ADDRESS | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 |

## Recall by slice (typed match, IoU 0.5)

| slice | rules | rules+stat | rules+stat+spacy | rules+stat+gliner | spacy-only | gliner-only |
|---|---|---|---|---|---|---|
| chat_log | 0.652 | 0.759 | 0.769 | 0.847 | 0.149 | 0.413 |
| email_thread | 0.752 | 0.905 | 0.912 | 0.929 | 0.241 | 0.537 |
| hard_negative | 0.668 | 0.840 | 0.878 | 0.894 | 0.251 | 0.529 |
| meeting_notes | 0.609 | 0.826 | 0.902 | 0.882 | 0.293 | 0.548 |
| non_english | 0.659 | 0.837 | 0.871 | 0.892 | 0.246 | 0.520 |
| obfuscated | 0.684 | 0.853 | 0.896 | 0.925 | 0.254 | 0.381 |
| ood | 0.673 | 0.843 | 0.878 | 0.893 | 0.243 | 0.516 |

## Records leaking by slice (verbatim: an identifier can still be read in the redacted output)

| slice | records | rules | rules+stat | rules+stat+spacy | rules+stat+gliner | spacy-only | gliner-only |
|---|---|---|---|---|---|---|---|
| chat_log | 69 | 0.768 | 0.493 | 0.478 | 0.261 | 0.986 | 0.942 |
| email_thread | 121 | 0.810 | 0.025 | 0.000 | 0.025 | 1.000 | 0.992 |
| hard_negative | 277 | 0.881 | 0.397 | 0.199 | 0.332 | 1.000 | 0.964 |
| meeting_notes | 110 | 1.000 | 0.664 | 0.200 | 0.645 | 1.000 | 0.955 |
| non_english | 210 | 0.909 | 0.391 | 0.205 | 0.329 | 0.995 | 0.957 |
| obfuscated | 39 | 0.821 | 0.359 | 0.154 | 0.333 | 1.000 | 1.000 |
| ood | 300 | 0.870 | 0.367 | 0.183 | 0.307 | 0.997 | 0.967 |

Skipped: llm (no GROQ_API_KEY and no cached responses)
