# WWTP Optimization Package

This package provides Python implementations of optimization algorithms for Wastewater Treatment Plants (WWTPs). It includes both single-objective (HGPSAL) and multi-objective (MEGA) optimization algorithms to minimize costs and maximize effluent quality.

## Abstract

This package implements open-source Python optimization algorithms for wastewater treatment plants. Two main optimization algorithms were implemented: Hybrid Genetic Pattern Search Augmented Lagrangian (HGPSAL) for single-objective optimization and Multi-objective Elitist Genetic Algorithm (MEGA) for multi-objective optimization.

Both algorithms, originally implemented in MATLAB, were converted to Python to increase accessibility and interaction. The optimization results focused on two major objectives: minimization of Total Cost and Quality Index (a measure of effluent quality). The algorithms were validated on benchmark problems and demonstrated significant improvements compared to previous published work.

The package was applied to a comprehensive WWTP model based on BSM1, including 115 decision variables and multiple constraints, providing important information on trade-offs between cost minimization and effluent quality maximization.

## Installation

### Prerequisites

- Python 3.7 or higher

### Setting up the environment

1. Clone the repository
```bash
git clone https://github.com/vaniamiguel13/WWTP_opt_pack.git
cd WWTP_opt_pack
```

2. Create a virtual environment
```bash
# Using venv
python -m venv env

# Activate the virtual environment
# On Windows
env\Scripts\activate
# On Unix or MacOS
source env/bin/activate
```

3. Install required packages
```bash
pip install -r requirements.txt
```

## Main Features

- Single-objective optimization using HGPSAL algorithm
- Multi-objective optimization using MEGA algorithm
- Support for both cost and quality index optimization
- Comprehensive WWTP model based on BSM1
- Visualization tools for results analysis

## Usage

Example notebooks demonstrating the usage of both HGPSAL and MEGA algorithms are provided in the `examples` directory.

## License

This project is licensed under Creative Commons CC BY-SA 4.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## Authors

- Vânia Miguel Bento

## Acknowledgments

This work was supervised by Dr. Isabel Espírito Santo at the University of Minho, School of Engineering.

## Citation

If you use this package in your research, please cite:

```
Miguel, V. (2024). Development of a python package to optimize wastewater treatment plants. 
Master's Dissertation, University of Minho, School of Engineering.
```
