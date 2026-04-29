# PoreTracker-OF

Pore-scale tracking and analysis tools built on OpenFOAM.

## Overview

PoreTracker-OF provides utilities for simulating and analysing pore-scale transport
phenomena using OpenFOAM solvers.

## Structure

```
PoreTracker-OF/
├── cases/          # OpenFOAM case templates
├── src/            # Custom solvers and utilities
├── scripts/        # Pre/post-processing scripts
├── docs/           # Documentation
└── results/        # Simulation results (gitignored)
```

## Requirements

- OpenFOAM (v9+ or ESI OpenFOAM v2206+)
- Python 3.8+ (post-processing)

## Getting Started

```bash
# Source OpenFOAM
source /opt/openfoam9/etc/bashrc

# Run a case
cd cases/<case_name>
./Allrun
```

## Author

Toni Ivas
