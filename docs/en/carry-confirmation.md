# Recurrent-state reset confirmation study

[中文](../carry-confirmation.md) · [Research design](research.md) · [Two-trajectory pilot](carry-study.md) · [Research results](research-results.md)

## Research question

Under the fixed DRQN, Foragax environment and training configuration, does the positive reward effect of a single live recurrent-state reset recur across new independent training trajectories?

The pilot reset-minus-retention mean reward differences were `+0.1903333333` and `+0.0617200000` for indices 23 and 24. Confirmation uses indices and seeds 25–32, with eight independent training trajectories as the analysis units.

## Fixed design

The study uses `ForagaxSquareWaveTwoBiome-v11`, 9×9 color observations and native DRQN. The upstream source commit is `9710f60fa30da5badc451ad7ce3ff296d5070830`, dependency-lock object is `18b5340e7ce408d03ee518ac40419f4269c0a1c0`, and Foragax version is `0.55.0`. Execution used frozen project commit `991482ba8734a49187133d122cb10c577ef74a0d` (tree `70e4a9289d40f2c95afc3a7ff71912547a8cc685`) on Linux x86_64, Python 3.12 and the JAX CPU backend (`jax_enable_x64 = false`, 8 logical CPUs), with run identity `4e64b8a3fa4cec7e41510a4c60af15b2e5024fe5f9cd009a96c4e4fd3415f853`. Model, optimizer, exploration and update settings follow the pilot configuration. [Study configuration](../../configs/carry-confirmation.json)

Each seed first completes 150,000 common interactions. Two branches then start from the same complete state. The retention branch keeps the live recurrent state. The reset branch sets `agent_state.carry` to `None` before the next policy computation. Each branch continues for 150,000 interactions.

Parameters, optimizer, replay, random state and environment state share the branch starting point. Each seed contributes 450,000 effective interactions, totaling 3,600,000 across eight seeds. Completed indices 23 and 24 are reported separately as an exploratory group and are excluded from the confirmatory inference sample.

## Primary endpoint and criterion

For each seed, use float64 reductions over environment interactions `[150000, 300000)` to compute

```text
R_i = mean reward(single reset) - mean reward(retained state)
```

Confirmation requires both a positive arithmetic mean of the eight `R_i` values and `p ≤ 0.05` in a one-sided exact binomial test of the positive-effect fraction. The null hypothesis is a positive-effect probability at most `0.5`. A positive effect means `R_i > 0`, and zero counts as nonpositive.

With eight trajectories, the one-sided probabilities for seven, eight and six positive effects are `9/256`, `1/256` and `37/256`, respectively. Confirmation therefore requires at least seven positive effects and a positive paired mean.

Across all eight completed confirmation trajectories, the primary endpoint results are:

- **Effective scientific interactions:** `3,600,000`
- **Arithmetic mean paired difference:** `+0.0219850000` (`+0.021985`)
- **Positive and zero counts:** `4 / 8` positive (indices 27–30), `4 / 8` negative (indices 25, 26, 31, 32), `0` zero values
- **One-sided exact binomial test:** `p = 163/256 = 0.63671875`
- **Two-sided 95% Clopper–Pearson interval for positive fraction:** `[0.157013, 0.842987]` (`[0.1570127705, 0.8429872295]`)
- **Prespecified confirmation criterion:** not met (`criterion_met = false`)

## Per-trajectory results

### Confirmation group (indices 25–32)

| Index (seed) | Retained `[150k, 300k)` | Single reset `[150k, 300k)` | Primary `R_i` | Early `[150k, 151024)` | Later `[151024, 300k)` | Retained events (elig/rec/cens) | Reset events (elig/rec/cens) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 25 | `1.0970666667` | `1.0955733333` | `-0.0014933333` | `-0.1152343750` | `-0.0007115240` | `762 / 754 / 8` | `725 / 717 / 8` |
| 26 | `0.9166133333` | `0.6622133333` | `-0.2544000000` | `+0.0390625000` | `-0.2564171410` | `807 / 795 / 12` | `846 / 834 / 12` |
| 27 | `0.8318533333` | `0.9907333333` | `+0.1588800000` | `+2.2363281250` | `+0.1446004726` | `838 / 822 / 16` | `809 / 801 / 8` |
| 28 | `0.8502800000` | `1.0355066667` | `+0.1852266667` | `+0.0703125000` | `+0.1860165396` | `913 / 898 / 15` | `965 / 956 / 9` |
| 29 | `0.8567066667` | `1.1368400000` | `+0.2801333333` | `+0.0253906250` | `+0.2818843304` | `809 / 798 / 11` | `776 / 768 / 8` |
| 30 | `0.8676133333` | `1.0234666667` | `+0.1558533333` | `+0.0351562500` | `+0.1566829556` | `808 / 798 / 10` | `751 / 742 / 9` |
| 31 | `0.9450000000` | `0.7088933333` | `-0.2361066667` | `-0.1855468750` | `-0.2364541940` | `806 / 800 / 6` | `794 / 776 / 18` |
| 32 | `1.1091866667` | `0.9969733333` | `-0.1122133333` | `+0.1738281250` | `-0.1141794651` | `594 / 592 / 2` | `636 / 626 / 10` |

### Exploratory pilot group (indices 23–24, reported separately)

| Index (seed) | Retained `[150k, 300k)` | Single reset `[150k, 300k)` | Primary `R_i` | Early `[150k, 151024)` | Later `[151024, 300k)` | Retained events (elig/rec/cens) | Reset events (elig/rec/cens) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 23 | `0.8749733333` | `1.0653066667` | `+0.1903333333` | `+0.3437500000` | `+0.1892788100` | `659 / 645 / 14` | `600 / 591 / 9` |
| 24 | `0.5910800000` | `0.6528000000` | `+0.0617200000` | `+0.0136718750` | `+0.0620502631` | `897 / 885 / 12` | `915 / 893 / 22` |

The exploratory pilot mean paired difference is `+0.1260266667` (`2/2` positive; `900,000` effective interactions, plus `100,000` interactions replayed during pilot interruption recovery).

## Secondary measurements and trajectory dynamics

- **Early and later fixed response windows:** Over the first 1,024 post-reset interactions `[150000, 151024)`, the eight-seed mean reward difference was `+0.2849121094` (`6/8` positive, range `-0.1855468750` to `+2.2363281250`, led by index 27 rising from `-0.6503906250` retained to `+1.5859375000` reset). Over the remaining interval `[151024, 300000)`, the mean difference was `+0.0201777468` (`4/8` positive, range `-0.2564171410` to `+0.2818843304`), matching the sign pattern of the primary endpoint.
- **25,000-interaction reward windows and 50,000-interaction update diagnostics:** Pre-boundary 25,000-interaction windows in `[0, 150000)` were byte-identical across paired arms (eight-seed means: `0.315530`, `0.902760`, `1.183940`, `1.185120`, `1.338020`, `1.380030`). Across the six post-boundary 25,000-interaction windows, the eight-seed mean differences (reset minus retained) were `+0.236880` (`7/8` positive), `-0.027440` (`3/8`), `-0.111660` (`3/8`), `-0.131450` (`4/8`), `+0.016140` (`4/8`), and `+0.149440` (`4/8`). Post-boundary 50,000-interaction update diagnostics averaged across the eight seeds were:

| Interval | `loss` (retained / reset) | `abs_td_error` (retained / reset) | `weight_change` (retained / reset) |
| --- | ---: | ---: | ---: |
| `[150000, 200000)` | `0.531813 / 0.529383` | `0.883816 / 0.885769` | `28.404413 / 28.919702` |
| `[200000, 250000)` | `0.534867 / 0.502083` | `0.893301 / 0.850384` | `29.135586 / 28.554013` |
| `[250000, 300000)` | `0.228127 / 0.291875` | `0.435377 / 0.517282` | `19.162294 / 20.382263` |

- **Biome-switch recovery and trajectory divergence:** Using the pilot's 64-step complete windows, 90% recovery threshold, and 2,048-step horizon, the median recovery among recovered events was `64.0` interactions in both arms for all eight seeds. Retained branches had `6,337` eligible events (`6,257` recovered, `80` unrecovered including `5` truncated by endpoint `300,000`); reset branches had `6,302` eligible events (`6,220` recovered, `82` unrecovered including `10` truncated by endpoint `300,000`). Jaccard overlap of post-boundary biome-switch timestamps (64-step windows) between paired arms was `0.047256`–`0.060458` (`154`–`389` spaced shared events), showing that a single state perturbation rapidly altered subsequent spatial visits and switch sequences.
- **Live-state bytes and scheduling:** All eight confirmation seeds applied a non-empty `agent_state.carry` intervention (changed path `["agent_state", "carry"]` only). Live carry size was `577,536` bytes (`144,384` `float32` elements across 10 tensor leaves) before reset, `0` bytes immediately after reset, and `577,536` bytes after the first subsequent policy computation (`initialized_carry = true`). First-step `agent_steps` (`150,001 -> 150,002`), `updates` (`37,488`), and first post-boundary optimizer update (`environment_step = 150,004`, `agent_steps = 150,005`, `updates = 37,489`) matched across paired arms.
- **Effective versus repeated interactions and resource profile:** Effective scientific interactions totaled `3,600,000` (`3,599,984` timed scan interactions over `11,789.82 s` plus `16` branch-start transitions). Across `81` worker launches, `80` workers completed (`72` checkpoint exits `10` and `8` final exits `0`); one interruption occurred on seed `28` during `common` after saving step `50,000` with logs reaching step `85,000`, accounting for `35,000` logged repeated interactions (`< 50,000` unsaved upper bound) upon resuming from `50,000` without adding independent samples. Completed workers totaled `21,047.85 s` wall time (`22,908.42 s` campaign wall span including the interruption), `30,542.71 s` user CPU, `3,784.37 s` system CPU, and `6,542,648 KiB` peak worker RSS (`3,999,488`–`6,542,648 KiB`).

## Interpretation and next step

The fixed eight-trajectory confirmation study did not reproduce a consistent positive long-horizon reward advantage from a single live-carry reset (`4/8` positive, `p = 0.63671875`). Secondary windows indicate a short-lived transient effect: reset branches outperformed retained branches in `6/8` seeds over the first `1,024` steps (`+0.284912`) and in `7/8` seeds over the first `25,000` steps (`+0.236880`), especially when the retained state was trapped in negative reward at the boundary (index `27`). Because both branches continued online replay insertion and optimizer updates every four steps over `150,000` interactions, this transient control perturbation altered subsequent experience streams (`< 6.1%` biome-switch overlap) and drove bidirectional parameter divergence (`-0.254400` to `+0.280133`).

Crossing `retain/reset carry` with `continue/freeze updates` in a `2×2` factorial design from the `150,000`-step common checkpoint directly separates immediate recurrent-state control from induced learning-trajectory divergence.

## Data and provenance

Analysis recomputes the primary and secondary endpoints from final NPZ files, checks common prehistory, native-state provenance, actual intervention paths, and resource records across all `80` completed workers, and counts recovery-repeated interactions separately from effective interactions. See the [contribution guide](../../CONTRIBUTING.md#recurrent-state-confirmation-study) for execution, continuation and resource settings.
