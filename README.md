# Lyra: An Efficient and Expressive Subquadratic Architecture for Modeling Biological Sequences

[![arXiv](https://img.shields.io/badge/arXiv-2503.16351-b31b1b.svg)](https://arxiv.org/abs/2503.16351)

## Introduction

Lyra is a novel deep learning architecture designed for accessible and powerful modeling of biological sequences (DNA, RNA, and proteins). It addresses the challenge of understanding sequence-to-function relationships, particularly the complex phenomenon of epistasis, where the effect of mutations depends on their context within the sequence.

Lyra offers a subquadratic ($O(N \log N)$) approach that achieves state-of-the-art (SOTA) performance on a wide range of biological tasks while being significantly more efficient.

## Key Features

* **State-of-the-Art Performance:** Achieves SOTA results across numerous biological tasks, including protein fitness prediction, intrinsically disordered region analysis, RNA function prediction (splice sites, ribosome loading), CRISPR guide efficacy, and more.
* **Epistasis Modeling:** Explicitly designed to model epistatic interactions by leveraging the mathematical connection between State Space Models (SSMs) and polynomial approximation. Lyra effectively captures higher-order dependencies that are crucial for understanding biological function.
* **Computational Efficiency:**
    * **Speed:** Subquadratic ($O(N \log N)$) scaling significantly accelerates training and inference compared to quadratic ($O(N^2)$) Transformers. It has been tested on and efficiently processes sequences up to ~65k length, far beyond the typical limits of Transformer-based foundation models in biology.
    * **Size:** Requires dramatically fewer parameters than recent bio-foundation models. Most tasks in the paper were solved with models around 55k parameters
    * **Memory:** Exhibits significantly lower memory usage (e.g. 100-2000x less) compared to recent bio-foundation models, enabling training on consumer-grade hardware.

## Architecture Overview

Lyra combines two main components[cite: 42, 50, 58]:

1.  **Projected Gated Convolutions (PGC):** These layers efficiently capture local sequence features and interactions using depthwise convolutions combined with gating.
2.  **Diagonalized State Space Models (S4D):** An efficient SSM variant used to model long-range dependencies via convolution. S4D's mathematical structure aligns well with modeling polynomial interactions characteristic of epistasis.

This hybrid approach effectively integrates local and global information while maintaining subquadratic scaling.

## Usage

*Lyra can be configured to utilize optimized kernels like FlashFFTConv for efficiency. Ensure prerequisites are met.*
* examples/run_relso.ipynb runs code as included in this repo, using FlashFFTConv.
* examples/Lyra_example_streamlined.ipynb is a standalone notebook, designed for maximum simplicity of use. 


## Usage (Example Workflow)

`Lyra_example_streamlined.ipynb` is a standalone notebook designed for user-friendly operation of Lyra, without requiring coding. More sophisticated pipelines may use run_relso.ipynb as an example. 

For `Lyra_example_streamlined.ipynb`:
1.  **Prepare Data:**
    * Input data should be in CSV format.
    * `Lyra_example_streamlined.ipynb` requires a `seq` column containing the biological sequences (DNA, RNA, or Protein).
    * `Lyra_example_streamlined.ipynb` requires a target column (e.g., `label`, `fitness`) containing the values to predict (for regression) or class labels (for classification).
    * Typically use separate `train.csv`, `val.csv`, and `test.csv` files.
2.  **Configure and Run:**

    * Specify:
        * Sequence Type (RNA/DNA/Protein) - determines input encoding dimension.
        * Task Type (Regression/Classification).
        * Paths to your data files.
        * Name of the target label column.
    * The example notebook includes Python classes (`RNADataset`, `DNADataset`, `ProteinDataset`) and encoding functions (`one_hot_encode_sequences_RNA`, `one_hot_encode_dna`, `one_hot_encode_protein`).
3.  **Training:**
    * The script initializes the Lyra model, which is done automatically upon pressing "Initialize Task."
    * Next, the user can execute the subsequent cell to begin training. The model is trained for a specified number of epochs (default 50), evaluating on the validation set to save the best-performing model state.
4.  **Prediction:**
    * Load the saved best model checkpoint.
    * Provide a new CSV file containing sequences in a `seq` column.
    * The script will encode the sequences, run inference using the trained model, and output predictions (e.g., to `predictions.csv`).

## Citation

This work was led by Krithik Ramesh and Sameed M. Siddiqui, with contributions from Albert Gu, Michael D. Mitzenmacher, and Pardis C. Sabeti. 

If you use Lyra in your research, please cite the original paper:

```bibtex
@article{ramesh2025lyra,
  title={Lyra: An Efficient and Expressive Subquadratic Architecture for Modeling Biological Sequences},
  author={Ramesh, Krithik and Siddiqui, Sameed M. and Gu, Albert and Mitzenmacher, Michael D. and Sabeti, Pardis C.},
  journal={arXiv preprint arXiv:2503.16351},
  year={2025}
}
```

Or:

Ramesh, K., Siddiqui, S.M., Gu, A., Mitzenmacher, M.D., Sabeti, P.C. (2025). Lyra: An Efficient and Expressive Subquadratic Architecture for Modeling Biological Sequences. *arXiv preprint arXiv:2503.16351*. [https://arxiv.org/abs/2503.16351](https://arxiv.org/abs/2503.16351)


## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International - see the `LICENSE` file for details.