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

   ## Walkthrough of the cell-state encoder

   Imagine a simple example like:
   $$
   x = (\textrm{genes}, \textrm{gene counts}, \textrm{cell type}) = ([1,17,10769], [1.00,0.20, 0.50], [5])
   $$

   There is a cell-state encoder and a transformer.

   The cell state encoder has:

   1. A gene embedding, $E_{\textrm{gene}}$, an embedding matrix of shape $(n_{\textrm{total genes}} \times d)$
   2. A gene count layer, $W_{\textrm{count}}$, a linear layer with weight matrix of shape $(1 \times d)$
   3. A cell type embedding, $E_{\textrm{cell type}}$, an embedding matrix of shape $(n_{\textrm{cell types}} \times d)$

   where
   - $d$ is an internal embedding dimension
   - $n_{\textrm{total genes}}$ is the total vocabulary of genes (~15,000 in this dataset)
   - $n_{\textrm{genes}}$ is the number of expressed genes in each cell (e.g., three in the example above)
   - $n_{\textrm{cell types}}$ is the number of cell types

   The cell state encoder takes the input $x$ and processes it as follows:

   1. For each gene index $\gamma$ in the input, look up its embedding:
      $$e_{\textrm{gene}}(\gamma) = E_{\textrm{gene}}[\gamma] \in \mathbb{R}^{d}.$$
      Doing so $n_{\rm genes}$ times results in a tensor of shape $(n_{\textrm{genes}} \times d)$

   2. For each gene count $c$ in the input, project it through a linear layer:
      $$e_{\textrm{count}}(c) = c \cdot W_{\textrm{count}} \, ({\textrm{shape: }} (1) (1 \times d) = d )\,.$$
      Doing so $n_{\rm genes}$ times results in a tensor of shape $(n_{\textrm{genes}} \times d)$

   3. These are combined to get gene+count embeddings:
      $$e_{\textrm{gene+count}} = e_{\textrm{gene}} + e_{\textrm{count}}$$
      with shape $(n_{\textrm{genes}} \times d)$

   4. For the cell type index $\tau$, look up its embedding:
      $$e_{\textrm{cell type}}(\tau) = E_{\textrm{cell type}}[\tau]$$
      with shape $(1 \times d)$

   5. If using FiLM conditioning:
      - Project the cell type embedding to get $\gamma$ (scale) and $\beta$ (shift) parameters:
      $$\gamma = f_{\gamma}(e_{\textrm{cell type}})$$
      $$\beta = f_{\beta}(e_{\textrm{cell type}})$$
      both with shape $(1 \times d)$ where
      $$
      f_{\gamma} \textrm{and} f_{\beta} : \mathbb{R}^{d} \rightarrow \mathbb{R}^{d}$$
      are small learnable MLPs.

      - Apply these to condition the gene embeddings:
      $$e_{\textrm{final}} = e_{\textrm{gene+count}} \odot \gamma + \beta$$
      where $\odot$ represents element-wise multiplication with broadcasting.

   6. If not using FiLM, add the cell type embedding to each gene embedding:
      $$e_{\textrm{final}} = e_{\textrm{gene+count}} + e_{\textrm{cell type}}$$

   In both cases, $e_{\textrm{final}}$ has shape $(n_{\textrm{genes}} \times d)$. FiLM preserves gene permutation equivariance through the cell state encoder, while the additive approach does not.

   ## Simple example - less explicit

   As a simple example, take $n_{\textrm{genes}} = 3$ and $d = 4$.
Represent $e_{\textrm{gene + count}}$ as a matrix of shape $(n_{\textrm{genes}} \times d) = (3 \times 4)$ by horizontally collecting four column vectors:

$$e_{\textrm{gene + count}} = (\vec{\epsilon_1}, \vec{\epsilon_2}, \vec{\epsilon_3}, \vec{\epsilon_4})$$

where each $\vec{\epsilon_i}$ is a three-dimensional column vector representing the same embedding dimension across all genes.

The FiLM conditioning parameters are:
$$ \gamma = (\gamma_1, \gamma_2, \gamma_3, \gamma_4)$$
$$ \beta = (\beta_1, \beta_2, \beta_3, \beta_4)$$
where each $\gamma_i$ and $\beta_i$ is a scalar.

When applying FiLM conditioning, each column vector $\vec{\epsilon_i}$ is scaled by the corresponding $\gamma_i$ and then shifted by $\beta_i$:

$$e_{\textrm{final}} = (\gamma_1\vec{\epsilon_1} + \beta_1\vec{1}, \gamma_2\vec{\epsilon_2} + \beta_2\vec{1}, \gamma_3\vec{\epsilon_3} + \beta_3\vec{1}, \gamma_4\vec{\epsilon_4} + \beta_4\vec{1})$$

where $\vec{1}$ is a three-dimensional vector of ones, and the operations are:
- $\gamma_i\vec{\epsilon_i}$: scalar-vector multiplication
- $\beta_i\vec{1}$: broadcasting the scalar $\beta_i$ across all genes

This maintains gene permutation equivariance because permuting the elements within each column vector $\vec{\epsilon_i}$ would result in the same permutation in the final output, preserving the relative relationships between genes.

## Simple example - more explicit

To be even more explicit, write out the $\vec{\epsilon}$ components explicitly into matrix form:

$$e_{\textrm{gene + count}} = 
\begin{pmatrix}
\epsilon_{1,1} & \epsilon_{1,2} & \epsilon_{1,3} & \epsilon_{1,4} \\
\epsilon_{2,1} & \epsilon_{2,2} & \epsilon_{2,3} & \epsilon_{2,4} \\
\epsilon_{3,1} & \epsilon_{3,2} & \epsilon_{3,3} & \epsilon_{3,4}
\end{pmatrix}$$

where each row represents a gene and each column represents one dimension of the embedding.

The FiLM conditioning parameters $\gamma$ and $\beta$ are represented as:

$$\gamma = (\gamma_1, \gamma_2, \gamma_3, \gamma_4)$$
$$\beta = (\beta_1, \beta_2, \beta_3, \beta_4)$$

The FiLM conditioning operation is then:

$$e_{\textrm{final}} = 
\begin{pmatrix}
\epsilon_{1,1} \cdot \gamma_1 + \beta_1 & \epsilon_{1,2} \cdot \gamma_2 + \beta_2 & \epsilon_{1,3} \cdot \gamma_3 + \beta_3 & \epsilon_{1,4} \cdot \gamma_4 + \beta_4 \\
\epsilon_{2,1} \cdot \gamma_1 + \beta_1 & \epsilon_{2,2} \cdot \gamma_2 + \beta_2 & \epsilon_{2,3} \cdot \gamma_3 + \beta_3 & \epsilon_{2,4} \cdot \gamma_4 + \beta_4 \\
\epsilon_{3,1} \cdot \gamma_1 + \beta_1 & \epsilon_{3,2} \cdot \gamma_2 + \beta_2 & \epsilon_{3,3} \cdot \gamma_3 + \beta_3 & \epsilon_{3,4} \cdot \gamma_4 + \beta_4
\end{pmatrix}$$

This can also be written using Hadamard (element-wise) product with broadcasting:

$$e_{\textrm{final}} = e_{\textrm{gene + count}} \odot 
\begin{pmatrix}
\gamma_1 & \gamma_2 & \gamma_3 & \gamma_4 \\
\gamma_1 & \gamma_2 & \gamma_3 & \gamma_4 \\
\gamma_1 & \gamma_2 & \gamma_3 & \gamma_4
\end{pmatrix} + 
\begin{pmatrix}
\beta_1 & \beta_2 & \beta_3 & \beta_4 \\
\beta_1 & \beta_2 & \beta_3 & \beta_4 \\
\beta_1 & \beta_2 & \beta_3 & \beta_4
\end{pmatrix}$$

## Equivariance of FiLM conditioning

This matrix formulation clearly shows that each column (dimension) is scaled by its corresponding $\gamma_i$ and shifted by $\beta_i$ uniformly across all genes. If the rows (genes) were permuted, the output would be permuted in exactly the same way, preserving equivariance.

To understand why FiLM conditioning preserves gene permutation equivariance, let's examine what happens when we permute the genes (rows) in our representation.

Consider a permutation operator $P$ that reorders the rows (i.e. genes) of our matrix. For example, if $P$ swaps the first and third rows, then:

$$P(e_{\textrm{gene + count}}) = 
\begin{pmatrix}
\epsilon_{3,1} & \epsilon_{3,2} & \epsilon_{3,3} & \epsilon_{3,4} \\
\epsilon_{2,1} & \epsilon_{2,2} & \epsilon_{2,3} & \epsilon_{2,4} \\
\epsilon_{1,1} & \epsilon_{1,2} & \epsilon_{1,3} & \epsilon_{1,4}
\end{pmatrix}$$

When we apply FiLM conditioning to this permuted matrix, we get:

$$e_{\textrm{final}}' = P(e_{\textrm{gene + count}}) \odot 
\begin{pmatrix}
\gamma_1 & \gamma_2 & \gamma_3 & \gamma_4 \\
\gamma_1 & \gamma_2 & \gamma_3 & \gamma_4 \\
\gamma_1 & \gamma_2 & \gamma_3 & \gamma_4
\end{pmatrix} + 
\begin{pmatrix}
\beta_1 & \beta_2 & \beta_3 & \beta_4 \\
\beta_1 & \beta_2 & \beta_3 & \beta_4 \\
\beta_1 & \beta_2 & \beta_3 & \beta_4
\end{pmatrix}$$

$$= 
\begin{pmatrix}
\epsilon_{3,1} \cdot \gamma_1 + \beta_1 & \epsilon_{3,2} \cdot \gamma_2 + \beta_2 & \epsilon_{3,3} \cdot \gamma_3 + \beta_3 & \epsilon_{3,4} \cdot \gamma_4 + \beta_4 \\
\epsilon_{2,1} \cdot \gamma_1 + \beta_1 & \epsilon_{2,2} \cdot \gamma_2 + \beta_2 & \epsilon_{2,3} \cdot \gamma_3 + \beta_3 & \epsilon_{2,4} \cdot \gamma_4 + \beta_4 \\
\epsilon_{1,1} \cdot \gamma_1 + \beta_1 & \epsilon_{1,2} \cdot \gamma_2 + \beta_2 & \epsilon_{1,3} \cdot \gamma_3 + \beta_3 & \epsilon_{1,4} \cdot \gamma_4 + \beta_4
\end{pmatrix}$$

Now, if we had first applied FiLM conditioning and then permuted the result, we would get:

$$P(e_{\textrm{final}}) = P(e_{\textrm{gene + count}} \odot \gamma + \beta)$$

$$= P\left(
\begin{pmatrix}
\epsilon_{1,1} \cdot \gamma_1 + \beta_1 & \epsilon_{1,2} \cdot \gamma_2 + \beta_2 & \epsilon_{1,3} \cdot \gamma_3 + \beta_3 & \epsilon_{1,4} \cdot \gamma_4 + \beta_4 \\
\epsilon_{2,1} \cdot \gamma_1 + \beta_1 & \epsilon_{2,2} \cdot \gamma_2 + \beta_2 & \epsilon_{2,3} \cdot \gamma_3 + \beta_3 & \epsilon_{2,4} \cdot \gamma_4 + \beta_4 \\
\epsilon_{3,1} \cdot \gamma_1 + \beta_1 & \epsilon_{3,2} \cdot \gamma_2 + \beta_2 & \epsilon_{3,3} \cdot \gamma_3 + \beta_3 & \epsilon_{3,4} \cdot \gamma_4 + \beta_4
\end{pmatrix}\right)$$

$$= 
\begin{pmatrix}
\epsilon_{3,1} \cdot \gamma_1 + \beta_1 & \epsilon_{3,2} \cdot \gamma_2 + \beta_2 & \epsilon_{3,3} \cdot \gamma_3 + \beta_3 & \epsilon_{3,4} \cdot \gamma_4 + \beta_4 \\
\epsilon_{2,1} \cdot \gamma_1 + \beta_1 & \epsilon_{2,2} \cdot \gamma_2 + \beta_2 & \epsilon_{2,3} \cdot \gamma_3 + \beta_3 & \epsilon_{2,4} \cdot \gamma_4 + \beta_4 \\
\epsilon_{1,1} \cdot \gamma_1 + \beta_1 & \epsilon_{1,2} \cdot \gamma_2 + \beta_2 & \epsilon_{1,3} \cdot \gamma_3 + \beta_3 & \epsilon_{1,4} \cdot \gamma_4 + \beta_4
\end{pmatrix}$$

Comparing these two results:
$$P(e_{\textrm{final}}) = e_{\textrm{final}}'$$

This means that permuting and then applying FiLM conditioning is equivalent to applying FiLM conditioning and then permuting. This property is the mathematical definition of equivariance:

$$P(f(x)) = f(P(x))$$

where f is the FiLM conditioning function.

Why does this work? Because:

1. The FiLM conditioning applies the same transformation parameters ($\gamma_i$ and $\beta_i$) to the same embedding dimension across all genes
2. Each gene is processed independently, with no mixing between genes
3. The transformation depends only on the embedding dimension (column), not on which gene (row) it belongs to

In contrast, if we had used a non-equivariant approach like a fully-connected layer that mixes information across genes, permuting the inputs would not result in the same permutation of the outputs, breaking equivariance.

This equivariance property is crucial for processing gene expression data, as it ensures that the model treats genes as an unordered set, where the identity of each gene is preserved regardless of the order in which they are presented to the model.

## General FiLM equivariance

Let $F_j$ be the transformation for column j: $F_j(x) = \gamma_j·x + \beta_j$
The full FiLM transformation is $F = [F_1, F_2, ..., F_d]$ applied independently to each column
Since each $F_j$ operates independently on each element without mixing information between rows,
any permutation of rows $P$ commutes with $F$: $P(F(X)) = F(P(X))$