## Overview

Vision-Language Navigation (VLN) agents often struggle when deployed in **unseen environments** or given **unfamiliar instruction styles**. We propose **Slow4Fast-VLN**, an interactive fast–slow reasoning framework for General Scene Adaptation VLN (GSA-VLN).

Instead of treating fast and slow reasoning as separate systems, Slow4Fast-VLN allows a navigation policy to **learn from past experiences**: the fast policy acts and records trajectories, while an LLM-based slow-reasoning module reflects on these records, distills reusable knowledge, and feeds it back into future fast decisions.

## Motivation

<p align="center">
  <img src="https://github.com/user-attachments/assets/053a8a93-ca9e-4270-bead-64d3bd0723f3" alt="Slow4Fast-VLN motivation: generalization from seen to unseen environments" width="95%">
</p>

Conventional VLN methods are commonly developed under relatively closed-set scene and instruction distributions. In GSA-VLN, agents must generalize from familiar residential scenes to novel environments such as shopping malls, offices, and cinemas, while handling basic, scene-specific, and personalized instructions.

**Our key idea:** slow reasoning should not merely solve difficult cases once; its insights should become **reusable experience that improves subsequent fast navigation**.

## Framework

<p align="center">
  <img src="https://github.com/user-attachments/assets/6395cc5a-38b9-4e4f-9540-18ad2a7241f8" alt="Slow4Fast-VLN framework: fast reasoning, memory, LLM reflection, and experience feedback" width="100%">
</p>

Slow4Fast-VLN has three closely connected components:

- **Fast reasoning:** a DUET-based navigation policy processes current observations and instructions, predicts actions, and records navigation histories.
- **Slow reasoning and experience memory:** an LLM reflects on historical trajectories and extracts structured knowledge about scene types, spatial context, spatial rules, and navigation strategies.
- **Fast–slow interaction:** relevant experiences are retrieved from the experience library and fused with visual features through attention, strengthening future policy decisions.

An additional **instruction style conversion** component uses LLM prompting to normalize scene-specific and user-style instructions into a basic style while preserving navigation intent.

## Getting Started

Clone the repository:

```bash
git clone https://github.com/yl6017339/Slow4Fast-VLN.git
cd Slow4Fast-VLN
```

The main navigation code is in `r2r/`, including `main_nav.py`, the experience-enhanced navigation model, the reflection memory bank, and the instruction-style conversion modules.

### Installation

Set up a PyTorch environment and install the dependencies required by the navigation backbone and LLM components. Prepare the GSA-R2R/R2R annotations, connectivity files, image features, and model resources before running experiments.

### Training & Evaluation

The navigation entry point is `r2r/main_nav.py`. Configure dataset paths and experiment arguments in accordance with your environment:

```bash
python -m r2r.main_nav --help
```

Training, validation, and checkpoint loading are controlled through its command-line options (including `--root_dir`, `--output_dir`, `--resume_file`, and `--test`).


}
```
