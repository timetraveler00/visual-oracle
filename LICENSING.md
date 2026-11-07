# Licensing scope and third-party materials

This document describes the intended licensing scope of the Visual Oracle
code-and-derived-results repository. It does **not** establish ownership of
external materials or substitute for a rights review.

## MIT-licensed material

The root [`LICENSE`](LICENSE) (MIT) applies to **original software source code
and its accompanying software documentation** for which Savaş Öztürk holds
the necessary rights. This includes the copyright holder's original Python
implementations and original executable notebook code, to the extent those
components are the copyright holder's own work. Third-party code, if present,
retains its original terms; the MIT notice cannot override another copyright
holder's rights.

Both named rights holders should approve the release, and any applicable
university, employment, or project intellectual-property requirements should
be checked before the repository becomes public.

## Material not licensed under MIT by this repository

- **Research data and derived results:** CSV tables, recorded measurement
  values, experiment scenarios and metadata, integrity records, and archived
  result ZIPs have **no separate open-data license assigned here**. Their
  inclusion in the repository is not a blanket grant to redistribute or
  republish them. Any later open-data license should be explicitly approved
  and documented separately.
- **NITS-IQA:** the benchmark's source images, underlying data rights, and
  original materials are governed by their respective providers' terms. The
  repository includes an authors' tabular analysis and score records, but not
  the original source BMP images. Benchmark rights and any restrictions on
  derived data must be checked before granting a separate data license.
- **Xiph DERF test media:** source videos such as Shields, FourPeople, and
  Parkrun are identified in `exp04_video_manifest.json`. Raw source videos
  and fault-injected video files are **not distributed** in this repository.
  References to those files do not grant redistribution rights. Verify the
  relevant media terms before sharing the videos separately.
- **Third-party software, libraries, and model weights:** Jitsi, Selenium,
  FFmpeg, pretrained neural models, and other external dependencies remain
  subject to their own terms. Their binaries, weights, and licenses are not
  replaced by this project's MIT license.

## Before changing visibility to Public

1. Obtain confirmation from all relevant source-code rights holders and check
   any institutional or grant-related intellectual-property obligations.
2. Review the Python scripts and notebooks for any copied/adapted third-party
   code requiring attribution or a separate license notice. A keyword scan is
   not a substitute for source-provenance review.
3. Establish whether the published derived tables, JSON records, and result
   ZIPs may be openly redistributed, and decide on an explicit data license
   only after checking the benchmark and participant/media conditions.
4. Document media provenance, external model/checkpoint identifiers, and the
   software environment independently of the MIT grant.

## Citation

For academic attribution, use [`CITATION.cff`](CITATION.cff). A citation request
is distinct from copyright licensing and does not imply journal acceptance.
