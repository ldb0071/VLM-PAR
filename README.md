# VLM-PAR

## Overview
VLM-PAR is a project designed for pedestrian attribute recognition. This repository contains datasets, models, and utilities to support the development and evaluation of machine learning models for identifying and classifying attributes of pedestrians in images. The project aims to advance research in computer vision and attribute recognition tasks.

## Features
- **Datasets**: Includes support for multiple datasets such as Market-1501, PETA, and others.
- **Models**: Implements various models for pedestrian attribute recognition.
- **Utilities**: Provides tools for data preprocessing, training, and evaluation.

## Repository Structure
```
registery.py
├── datasets/
│   ├── __init__.py
│   ├── market_1501.py
│   ├── mivia_par_kd_2025.py
│   ├── pa_100k.py
│   ├── parse27k.py
│   ├── peta.py
├── models/
│   ├── __init__.py
│   ├── t.py
│   ├── vlmpar.py
│   ├── vlmpar2.py
│   ├── vlmparbest.py
│   ├── vlmparcontrastive
│   ├── vlmparcrossdecouple.py
│   ├── vlmparnegative
├── utils/
│   ├── __init__.py
│   ├── configs.py
│   ├── data.py
│   ├── logger.py
│   ├── misc.py
│   ├── trainer.py
```

## Getting Started
### Prerequisites
- Python 3.8+
- Required libraries: [list libraries, e.g., TensorFlow, PyTorch, NumPy]

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ldb0071/VLM-PAR.git
   ```
2. Navigate to the project directory:
   ```bash
   cd VLM-PAR
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Prepare the datasets:
   - [Instructions for dataset preparation]
2. Train the model:
   ```bash
   python utils/trainer.py
   ```
3. Evaluate the model:
   ```bash
   python utils/evaluator.py
   ```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the [MIT License](LICENSE).