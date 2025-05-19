# Improved Transformer Model for scRNA-seq Analysis

## Project Overview

This project aims to build on the approach introduced in ScAtt (Liu et al.) by developing an enhanced transformer-based architecture for analyzing single-cell RNA sequencing (scRNA-seq) data. Key innovations focus on multi-cell-type analysis and integration of prior knowledge about gene regulatory networks.

## Limitations of Current Approach (ScAtt)

- Trains separate models for each cell type, missing potential cross-cell-type interactions
- Does not leverage existing gene regulatory network (GRN) knowledge
- Limited interpretability for distinguishing disease-specific vs cell-type-specific effects
- Computational inefficiency for handling all cell types simultaneously

## Key Innovations

### 1. Unified Multi-Cell-Type Model

**Approach:**
- Train a single model on all cell types simultaneously rather than separate models
- Encode cell type as a conditioning input (e.g., cell type embedding or token)
- Enforce permutation invariance to cell ordering within batches

**Benefits:**
- Capture cross-cell-type interactions in disease mechanisms
- Enable knowledge transfer between common and rare cell types
- Provide a more holistic view of disease pathology
- Distinguish between shared and cell-type specific disease mechanisms

### 2. LoRA-Inspired GRN Integration

**Approach:**
- Decompose attention maps: $A_{\textrm{total}} = A_{\textrm{GRN}} + A_{\textrm{learned}}$
- $A_{\textrm{GRN}}$ represents population-level gene regulatory networks (prior knowledge)
- $A_{\textrm{learned}}$ captures control and disease deviations from population patterns
- Parameterize the relationship with various options (TBD):
  - Simple scaling: $A_{\textrm{total}} = A_{\textrm{GRN}} + \lambda \cdot A_{\textrm{learned}}$ (hyperparameter $\lambda$)
  - Cell-type specific: $A_{\textrm{total}} = \alpha_c \cdot A_{\textrm{GRN}} + \beta_c \cdot A_{\textrm{learned}}$ (learnable parameters $\alpha_c, \beta_c$ for each cell type)
  - Gating mechanism: $A_{\textrm{total}} = G \odot A_{\textrm{GRN}} + (1-G) \odot A_{\textrm{learned}}$ (learnable gating function $G$)

**Benefits:**
- Directly measure how disease states differ from healthy regulation
- Improve interpretability by isolating disease-specific patterns
- More parameter-efficient training
- Generate testable hypotheses about disease mechanisms

### 3. Uncertainty Quantification for Attention Weights

**Approach:**
- Implement Monte Carlo dropout during both training and inference
- Perform multiple forward passes with different dropout masks
- Calculate statistics (mean $\mu$ and variance $\sigma^2$) over resulting attention distributions
- Quantify uncertainty as $\sigma^2 / \mu$ for each gene-gene relationship
- Identify high and low confidence regulatory relationships

**Benefits:**
- Distinguish between confident and uncertain regulatory relationships
- Improve biological interpretability
- Guide experimental validation toward high-confidence findings
- Capture model uncertainty in a computationally efficient way

## Implementation Plan

1. **Data Processing**
   - Process scRNA-seq data for all cell types 
   - Implement cell type encoding strategy
   - Prepare population-level GRN data for initialization

2. **Model Architecture**
   - Implement permutation-invariant transformer architecture
   - Design LoRA-inspired attention decomposition
   - Integrate Monte Carlo dropout for uncertainty quantification
   - Implement cell type conditioning mechanism

3. **Training Strategy**
   - Develop effective batching strategy for multi-cell-type training
   - Implement hyperparameter tuning for GRN integration
   - Design loss function that balances classification performance and GRN fidelity

4. **Evaluation**
   - Compare multi-cell-type model performance against separate models
   - Evaluate disease-specific attention patterns ($A_{\textrm{learned}}$)
   - Assess uncertainty in regulatory relationships
   - Validate findings against known disease mechanisms

5. **Biological Interpretation**
   - Identify shared and cell-type specific disease mechanisms
   - Characterize novel regulatory relationships with high confidence
   - Map findings to known biological pathways
   - Generate testable hypotheses for experimental validation

## Expected Outcomes

1. A more powerful and interpretable model for analyzing scRNA-seq data in disease contexts
2. Novel insights into cross-cell-type interactions in disease states
3. Quantitative characterization of disease-specific changes to gene regulatory networks
4. Identification of potential therapeutic targets with associated confidence levels
5. Methodological framework applicable to diverse disease contexts

# Model Mathematics

Imagine a simple example like:
$$
x = (\textrm{genes}, \textrm{gene counts}, \textrm{cell type}) = ([1,17,10769], [1.00,0.20, 0.50], [5])
$$

We have a cell-state encoder and a transformer.

The cell state encoder has:

1. An embedding layer for the genes:
$$
E_{\textrm{gene}_i} = M_{\textrm{gene}_i} \cdot \hat{e}_{\textrm{gene}_i}
$$
where
  - $M_{\textrm{gene}_i}$ is an $d_{\textrm{gene embedding}} \times n_{\textrm{gene}}$ matrix
  - $\hat{e}_{\textrm{gene}_i}$ is a unit vector of size $n_{\textrm{gene}}$ corresponding to the $i$-th gene
  - So the gene-embedding is a simple lookup in the $M_{\textrm{gene}_i}$ matrix.
2. A linear layer for the gene counts:
$$
C_{\textrm{gene}_i} = M_{\textrm{counts}_i} \cdot \vec{c}
$$
where
  - $M_{\textrm{counts}_i}$ is an $d_{\textrm{gene count}} \times 1$ vector
  - $\hat{c}_{\textrm{gene}_i}$ is a vector of size $n_{\textrm{gene}}$ corresponding to the $i$-th gene count
  - So the gene-count representation is a linear transformation of the gene embeddings.