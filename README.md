# ODIR Multimodel LDA BiLSTM

A deep learning project combining Latent Dirichlet Allocation (LDA) and Bidirectional LSTM (BiLSTM) for analyzing ODIR (Ocular Disease Intelligent Recognition) dataset.

## Project Overview

This project implements a multimodal deep learning approach that integrates:
- **LDA (Latent Dirichlet Allocation)**: For topic modeling and feature extraction
- **BiLSTM (Bidirectional LSTM)**: For sequence learning and classification

The model is designed to work with the ODIR dataset for ocular disease recognition and classification.

## Features

- Multimodal architecture combining LDA and BiLSTM
- Topic modeling for feature extraction
- Bidirectional sequence processing for improved context understanding
- Jupyter notebook-based implementation for easy experimentation and visualization

## Project Structure

```
odir-multimodel-lda-bilstm/
├── README.md                 # Project documentation
├── project.ipynb            # Main Jupyter notebook with implementation
├── .gitignore               # Git ignore rules
└── LICENSE                  # Project license
```

## Installation

### Requirements
- Python 3.7+
- Jupyter Notebook
- TensorFlow/Keras
- PyTorch (optional, depending on implementation)
- NumPy, Pandas, Scikit-learn
- NLTK (for NLP tasks)
- Gensim (for LDA)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/iam-mukeshvenkatesh/odir-multimodel-lda-bilstm.git
cd odir-multimodel-lda-bilstm
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch Jupyter Notebook:
```bash
jupyter notebook project.ipynb
```

## Usage

1. Open `project.ipynb` in Jupyter Notebook
2. Follow the cells sequentially to:
   - Load and preprocess the ODIR dataset
   - Apply LDA for topic modeling
   - Build and train the BiLSTM model
   - Evaluate model performance
   - Visualize results

## Model Architecture

The model combines:
1. **LDA Component**: Extracts topic features from text/documents
2. **BiLSTM Component**: Processes sequential data bidirectionally for classification
3. **Integration Layer**: Combines features from both models for final prediction

## Results

[Add your results, accuracy metrics, and performance comparisons here]

## Dataset

The ODIR (Ocular Disease Intelligent Recognition) dataset is used for training and evaluation. The dataset contains fundus images and diagnostic information for various ocular diseases.

## References

- LDA: Blei, D. M., Ng, A. Y., & Jordan, M. I. (2003). "Latent Dirichlet Allocation"
- BiLSTM: Graves, A., & Schmidhuber, J. (2005). "Framewise phoneme classification with bidirectional LSTM networks"
- ODIR Dataset: https://odir2019.grand-challenge.org/

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**Mukesh Venkatesh**
- GitHub: [@iam-mukeshvenkatesh](https://github.com/iam-mukeshvenkatesh)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Thanks to the ODIR challenge organizers for the dataset
- Built with TensorFlow/Keras and PyTorch

---

**Last Updated**: June 2026
