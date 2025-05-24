# MSHTrans

## Introduction
- This is the PyTorch implementation of the MSHTrans, which has been accepted by KDD 2025.

## Paper
Z. Chen, Z. Wu, W. K. Cheung, H-N Dai, B. Choi and J. Liu. ​**MSHTrans: Multi-Scale Hypergraph Transformer with Time-Series Decomposition for Temporal Anomaly Detection**. SIGKDD 2025.

## Code Structure
```text
├── common/               # Codes for performance evaluation and data loading
│   ├── evaluation/       # Codes for performance evaluation
├── networks/             # Codes for networks
├── scripts/              # Running demos
├── main.py               # main function
├── requirements.txt      # Requirements
```

## Requirements
- See [`requirements.txt`](./requirements.txt).

## Data Download
- Download data from Google Drive: [Download Link](https://drive.google.com/file/d/1bnFMU0jhJYRnhFwuWkZgRAq_DxecZZBQ/view?usp=sharing)
- Unzip and move data to data folder (defined in parameter `--data-root`)

## Quick Start
- You can run the codes with scripts in `./scripts/scripts.sh`:
```bash
# Example command
bash ./scripts/scripts.sh
```