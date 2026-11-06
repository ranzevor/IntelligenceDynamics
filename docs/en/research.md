# Research questions and experimental contract

[中文](../research.md) · [Research API](api-research.md)

## 1. Scientific question

IntelligenceDynamics asks a falsifiable question: when systems must adapt in partially observed, changing environments while paying computation and memory costs, do engineering, evolutionary search, and natural evolution produce similar dynamical organization and causal effects? The comparison concerns how information persists, computation is allocated, and learning changes behavior, rather than numerical weights or visual resemblance.

The strong hypothesis is that principles sufficient for broad continual adaptation admit a relatively compact description. The weaker hypothesis is that there are recurring organizational families, even without a unique architecture: persistent state, conditional computation, circuit reuse, multiple timescales, and local plasticity.

Competing explanations include equally efficient but unrelated organizations; apparent convergence induced by shared tasks, optimizers, or metrics; and similarities so abstract, such as feedback or optimization, that they offer little predictive value. Experiments should distinguish these possibilities.

### Analytical concepts

- **State:** internal information affecting current action that cannot be recovered fully from the current observation.
- **Memory:** storage and updating that make past information available later.
- **Computation:** information processing and state transitions, measured in explicitly defined units.
- **Learning:** experience-dependent changes to behavior or update rules.
- **Control:** selection of actions, computational resources, update timing, and stopping conditions.

One dynamical system may implement these functions together. `S[t+1] = F(S[t], O[t], A[t], M[t])` is a descriptive framework, not a requirement for separate prediction, routing, or world-model modules. Matched experiments determine whether conditional computation reduces recomputation and whether learning reduces future adaptation cost.

## 2. Three research tracks

| Track | Object | Main operations | Main comparison |
|---|---|---|---|
| A: engineering and compression | Systems already capable of continual adaptation | Resource restriction, localized updates, altered state availability, sparsification, removal of task cues | Joint capability, adaptation, plasticity, and resource frontier |
| B: evolution and discovery | Learning rules, gating, modulation, memory, and organization-generating rules | Search over inherited parameters, lifetime learning, independent environment evaluation | Learning-mechanism search versus fixed-policy search with matched budgets |
| C: biology and mechanism extraction | Connectivity, activity, and intervention data | Extract implementation-neutral descriptors and compare perturbation responses | Local control, long-range coordination, reuse, persistence, and resource constraints |

Track A progressively compares task-ID routing, observation-inferred routing, persistent-state routing, and routing without explicit boundaries. Track B separately tests dependence on reward density, task labels, global updates, and fixed topology. Changing several of these at once does not isolate one mechanism. Track C distinguishes structural evidence, simulated dynamics, recorded activity, and causal intervention; graph-based influence estimates and direct neural interventions answer different questions.

Discovery processes must be independent. A structure imposed on every candidate cannot be counted as independently emergent. Biological systems provide naturally formed comparators, with historical and physical constraints that also require consideration.

## 3. Hypothesis matrix

These are predictions to test. H identifiers denote scientific questions, not completion order or evidence grades.

| Hypothesis | Observable prediction and main test | Alternative or weakening condition | Tracks |
|---|---|---|---|
| H1 Resource pressure promotes conditional computation | At matched capability, tighter budgets reduce active computation and increase state dependence | Architecture or regularization imposes sparsity; dense systems retain an equal or better frontier | A, B, C |
| H2 Persistent state improves efficiency under partial observability | After controlling capacity and computation, state retention reduces replay and improves adaptation; intervene on state | Extra capacity explains the benefit; equally budgeted history windows match it | A, C |
| H3 Continual adaptation promotes circuit reuse | Related contexts share functional subsets; targeted ablation changes the corresponding behavior | Reuse is imposed by search rewards or modules; non-reusing systems are equally efficient | A, B, C |
| H4 Multiple timescales have adaptive value | Activity, updating, and structural change separate measurably; timescale interventions affect recovery | Separation simply reflects fixed hyperparameters; single-timescale systems match performance | A, B, C |
| H5 Searching learning organization outperforms searching only fixed policies | With matched parameter, candidate-evaluation, and interaction budgets, learning mechanisms adapt faster to new instances | A smaller search space explains the advantage; direct policy search matches transfer | B |
| H6 External scaffolding can be removed progressively | Inferred state, local signals, or autonomous routing replace labels, dense rewards, or global updates while preserving capability | Hidden supervision remains; removal consistently fails across an adequate search range | A, B |
| H7 Error-related signals participate in resource allocation | Intervening on error, surprise, or uncertainty changes computation or updating and adaptation | Signals are incidental; equally effective systems do not require them | A, B, C |
| H8 Long-lifetime adaptation requires maintained plasticity | Late-life systems still learn standardized new perturbations; freezing updates causes reproducible adaptation changes | Static competence with bounded memory suffices; update magnitude is disconnected from learnability | A, B |
| H9 Convergence concerns dynamics and causal roles | Different implementations exhibit similar matched intervention responses rather than identical weights | Shared tasks induce representational similarity while causal responses differ | A, B, C |
| H10 Biological systems are one family of Pareto solutions | Some biological motifs recur artificially, while other details can be removed without harming the frontier | Relevant capability requires fine-grained biological reconstruction | A, C |
| H11 Sufficient principles may form a limited set of organizational families | Independent implementations form stable organizational clusters with corresponding causal signatures | Clusters depend on metric choice; there are no stable clusters or only one effective architecture | A, B, C |
| H12 Learning reduces future adaptation cost | Related recurring changes require fewer recovery interactions or operations while capability is maintained | Knowledge increases without lower recovery cost; changes themselves become easier | A, B |

Initial measurement priorities are persistent state, maintained plasticity, conditional computation, reuse, and timescales. Predictive coding, spiking networks, world models, and developmental topology are candidate implementation families rather than prerequisites.

## 4. Tasks and information boundaries

### Synthetic hidden-mapping task

The built-in observation contains only a discrete cue, and the action is binary. The current cue-to-correct-action mapping is hidden and may change at specified lifetime times. Feedback for action `t` arrives before action `t + delay + 1` and identifies the original action credit. Evaluation records include hidden state and immediate correctness; policies do not receive those fields.

Scenario banks contain stationary and switching conditions, with mirrored pairs sharing cue sequences and having opposite target mappings. Every deterministic fixed cue policy has mean accuracy 0.5 across a mirrored pair. Mirroring controls initial-mapping preference; it does not replace broad task distributions or external environment evaluation.

Inherited objects contain initial preferences and update-rule parameters. Value tables, outstanding credits, gate windows, and accumulators reset at each lifetime. Search uses training scenarios only. Random-control probabilities are calibrated on training data. Final evaluation scenarios are not used to select candidates or tune thresholds.

### Mechanism comparisons

Fixed, always-updating, instantaneous conditional, leaky-accumulator, persistence, buffered-persistence, and random-gating policies share the feedback interface. Persistence requires enough high-error events within a window. Buffered persistence additionally retains eligible earlier credits and commits them upon confirmation. Updates are attributed to the originating feedback, not to the corruption label at commitment time.

Isolated and burst corruption test individual erroneous feedback and temporally clustered errors. The rate parameter is an attempted-start probability; the realized fraction of flipped feedback is reported separately. When only isolated corruption enters training fitness, burst corruption can serve as a stress condition excluded from fitness.

## 5. Causal interventions and comparison units

### Matched branching

Generate the common history once, before a predetermined action boundary. Copy the environment, policy, random states, pending feedback, and counters, then verify copy identity. After intervention and before the next action, structural differences must be confined to declared fields.

- **Freeze updates:** retain learned values, exploration, and the action interface; change only whether later updates are allowed.
- **Reset a designated state once:** preserve parameters, optimizer, replay, randomness, and environment while changing that state.
- **Reset gate memory:** clear accumulated error, recent flags, and credits awaiting confirmation, while preserving values and feedback credits not yet delivered.

The built-in `reset_memory` performs the third operation, not a neural recurrent-state reset. A one-time reset is also different from disabling memory for the entire continuation. An intervention that changes no state is recorded as a no-op.

For late-life plasticity, compare continued and frozen learning over fixed continuation windows at predetermined ages: `P(t) = mean_reward_continued - mean_reward_frozen`. This measures the behavioral value of the learning path. Direct learnability tests additionally require comparable new perturbations. Gradients, losses, or weight changes alone do not fully define plasticity.

### Event support and fixed-time support

An event response is the post-window mean minus the equal-width pre-window mean. If policies affect event occurrence, report event counts and timestamp overlap in each arm. Common-support analysis intersects eligible timestamps first, then applies one deterministic spacing rule to the shared set. With no shared events, that contrast is unidentified and returns `None`.

Shared events may still be selected by treatment-induced behavior. Their contrast has a different estimand from a fixed-time effect. Fixed-time blocks do not condition on whether a post-intervention event occurred. State the direction of every contrast, retain both analyses, and do not replace a primary endpoint through post-outcome metric selection.

## 6. Shared measurements

| Layer | Quantities to report | Interpretation |
|---|---|---|
| Capability | Reward, success, or accuracy distributions; weakest conditions; lifetime trajectories | Preserve task-specific units |
| Adaptation | Recovery interactions, recovery-curve area, computation and updating during recovery | Specify reference level, window, and observation horizon in advance |
| Plasticity | Learning speed, achievable performance, and frozen controls for standardized perturbations at different ages | Distinguish forgetting from state construction |
| Computation | Active operations, full-activation opportunities, updates, routing, and recurrence overhead | Activity ratios do not automatically measure energy efficiency |
| Memory | Parameters, persistent state, replay, external history, and traffic | Separate storage capacity from access cost |
| Organization | Reuse, local and long-range relationships, routing conditionality, timescales, and perturbation responses | Identical labels need not imply identical causal roles |

Cross-system comparisons use Pareto frontiers over capability, adaptation, plasticity, computation, memory, and complexity. Search may use a prespecified scalar fitness, but evaluation must retain its components. That scalar is not a universal intelligence score.

## 7. Statistics and confirmatory contract

1. Before final data generation, specify hypotheses, primary endpoints, effect directions, thresholds or equivalence margins, joint decisions, seed counts, observation duration, exclusions, and missing-data handling.
2. Independent training or evolutionary seeds are inferential units. Timesteps, events, and branches from one run are dependent. Resampling and permutation procedures must respect this hierarchy.
3. Report every planned seed, paired contrasts, distributions, and suitable uncertainty intervals. Descriptive sign counts are not automatically significance tests.
4. Separate exploratory and confirmatory data. Confirmation data cannot select mechanisms, random-control probabilities, resource weights, or decision thresholds. Additional analyses require explicit hypotheses and data roles.
5. Nonfinite input is invalid numeric data. No observation, unrecovered behavior, zero denominators, and unidentified common support have different meanings. Preserve reasons and counts rather than replacing them with zero.
6. `summarize` returns `None` for the mean whenever any observation is missing. A recovery latency of `None` means no qualifying recovery was observed or the available window was too short; interpret it with its window, not as an arbitrary large number.
7. `assess` returns `True`, `False`, or `None` per threshold. For `all`, a definite failure determines failure; for `any`, a definite pass determines passage. Otherwise a combination containing missing values is undetermined. Passing thresholds supplies neither confidence intervals nor causal identification.
8. Separate execution validity from scientific conclusions. Broken inputs, incorrect branching, and information leakage affect validity. Complete data that do not support a hypothesis remain scientific outcomes.

Multiple-hypothesis analyses should prespecify priorities and multiplicity handling. Environment, optimizer, architecture, budget, and reasonable metric variations test whether common tasks or measurements induce apparent convergence.

## 8. Evidence grades

| Grade | Definition |
|---|---|
| E0 | A single scientific run or qualitative observation |
| E1 | Reproducible observations across seeds within one implementation and environment |
| E2 | Robust results across environment, resource, or implementation variants within one track |
| E3 | The same phenomenon independently observed in at least two tracks under a shared operational definition |
| E4 | Cross-track observations accompanied by matched causal intervention responses |
| E5 | Engineered, evolutionary, and biological sources causally aligned at the same abstraction level |

Grades describe a particular body of research material; they are not automatically assigned to algorithms, software versions, or citations. Unit tests validate software. Cross-track convergence additionally requires independent emergence, predictive value, resource trade-offs, and tests of alternatives. Structural similarity alone cannot raise a grade.

## 9. Current algorithms and subsequent study design

Built-in algorithms cover finite synthetic tasks, rule-parameter search, feedback gating, matched branching, configured criteria, and sequence comparisons. Neural networks, connectome dynamics, circuit-reuse estimators, multi-scale resource frontiers, and cross-track intervention alignment require corresponding models and data.

Studies follow their dependencies: validate inputs and measurement, establish continual-adaptation baselines, isolate state or update mechanisms, then compare resource frontiers and cross-track causal signatures. External native models first need explicit observation, state, action, feedback, and randomness contracts before auditable interventions. Long-lived ecology and open-ended evolution additionally require estimands for novelty, complexity, continual adaptation, and curriculum change.

## 10. Academic sources

These sources motivate questions or candidate mechanisms. Their findings retain their original task and model scope.

- **Forager, Tang et al., 2026.** A partially observable continual-RL environment with constant environment memory, including comparisons of plasticity and state construction. [Paper](https://arxiv.org/abs/2605.01131)
- **Nevo-CRL, Li et al., 2026.** A fixed-capacity network uses task-specific masks, neuron reuse, and recycling. Explicit task structure remains a condition to test separately. [ICML paper](https://proceedings.mlr.press/v306/li26gb.html)
- **Rank and Gradient, Wu et al., 2026.** Studies plasticity loss through NTK rank and gradient attenuation in nonstationary reinforcement learning. [ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/2886558a03d11f95d4020a460fe4a390-Abstract-Conference.html)
- **Barriers for Learning, Joudaki et al., 2026.** Analyzes gradient dynamics trapped in invariant submanifolds, highlighting potential tension between static compression and continued plasticity. [ICLR paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/511c7fd69db9f1ce7492a57285975849-Abstract-Conference.html)
- **DiscoRL, Oh et al., 2025.** Meta-learns reinforcement-learning update rules from multi-environment experience and evaluates benchmarks excluded from discovery. [Nature paper](https://www.nature.com/articles/s41586-025-09761-x)
- **DAL, Arnold, Suzuki, Arita, and Yamazaki, 2024.** Evolves locally modulated updates integrating non-reward information in a two-dimensional navigation domain. [ALIFE paper](https://arxiv.org/abs/2404.12631)
- **HBL, Arnold, Suzuki, and Arita, 2026.** Studies transitions from reward-driven learning toward reduced reward dependence in navigation and quadruped locomotion. [Author abstract](https://pubmed.ncbi.nlm.nih.gov/42518295/)
- **Plasticity-rule meta-learning, Confavreux et al., 2025.** Evolution strategies search local rules in recurrent spiking networks; rule complexity and objective constraints affect search effectiveness. [PLOS paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012910)
- **BANC, Bates et al., 2026.** A fly brain-and-cord reconstruction reveals local sensory–effector loops and long-range coordination; some visual structures are outside its reconstructed extent. [Nature paper](https://www.nature.com/articles/s41586-026-10735-w)
- **Spiking predictive coding survey, N’dri et al., 2026.** Organizes models by explicit error populations, membrane potentials, and implicit error encoding. [Journal review](https://www.sciencedirect.com/science/article/pii/S0893608025012523)
- **Hierarchical timescale hypothesis, Lee, Nemenman, and Levchenko, 2026.** A perspective proposing a cross-system account of layered organization and multiple timescales. [Cell Systems article](https://www.sciencedirect.com/science/article/pii/S2405471225003400)
- **Brain–AI convergence, Ohmae and Ohmae, 2026.** A comparative perspective on shared principles involving world models, prediction, and attention. [Patterns article](https://www.sciencedirect.com/science/article/pii/S2666389926001029)
- **Algorithmic-space convergence, Krauss et al., 2026.** A preprint comparing permutation-aligned weight structures of small static classification networks. [Preprint](https://arxiv.org/abs/2608.05985)
- **Flow-Lenia, Plantec et al., 2025.** Mass-conserving continuous cellular automata support localized complex patterns and evolutionary-activity measurements, providing a candidate ecological substrate. [Artificial Life paper](https://direct.mit.edu/artl/article/31/2/228/130572/Flow-Lenia-Emergent-Evolutionary-Dynamics-in-Mass)
