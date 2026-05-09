> From: https://gemini.google.com/app/fe567f1bd9cc3c16

# gemini response

I've completed your research. Feel free to ask me follow-up questions or request changes.

Post-Hoc Calibration for Image Classification

# Post-Hoc Probability Calibration for Multiclass Image Classification: Algorithmic Analysis and Deployment Architecture

The deployment of deep convolutional neural networks in automated classification systems necessitates robust probability calibration to ensure that the output probabilities faithfully represent the empirical likelihood of correctness. In the context of the AutoLens AI project, an eight-class vehicle body type classification system utilizing an EfficientNet-B2 backbone, the model currently achieves a highly optimized macro F1 score and an accuracy of approximately 92.39%. However, like most modern high-capacity neural networks trained via cross-entropy loss, the uncalibrated model inherently suffers from severe overconfidence, pushing softmax probabilities toward absolute zero or one regardless of true generalization capability.[1, 2] 

The application of post-hoc Temperature Scaling to the AutoLens AI baseline has already demonstrated substantial improvements, reducing the Negative Log-Likelihood (NLL) from 0.29796 to 0.24160 and the Expected Calibration Error (ECE) from 0.03823 to 0.01043, all while perfectly preserving the 92.39% accuracy. This indicates a successful attenuation of the network's margin scaling issues. Nevertheless, deploying this system into a production environment via ONNX Runtime with a strict 95 MB artifact size limit and a Gradio user interface demands a rigorous evaluation of alternative calibration methodologies. The selection of an optimal calibration algorithm must perfectly balance mathematical expressivity against deployment overhead, the risk of overfitting on a restricted validation split of 3,157 samples, and the strict preservation of class ranking metrics such as the macro F1 score.

## Theoretical Foundations and Evaluation Metrics

To rigorously assess the alignment between predicted probabilities and empirical correctness in a multiclass computer vision environment, a comprehensive mathematical framework for evaluation must be established. The assessment of probability calibration relies on strictly proper scoring rules, bin-based expectation metrics, and visual diagnostic tools. These metrics evaluate whether a probability prediction of 0.9 for the "SUV" class empirically corresponds to a 90% accuracy rate across similarly scored inferences.[3, 4]

### Strictly Proper Scoring Rules

A scoring rule is defined as strictly proper if its expected value is uniquely minimized by the true underlying probability distribution.[4, 5] These rules assess both the reliability (calibration) and discriminative power (resolution) of a predictive model simultaneously.

The Negative Log-Likelihood (NLL), often equivalent to the cross-entropy loss in classification, is defined mathematically as $L = -\frac{1}{N} \sum_{i=1}^N \log(\hat{p}_{i, y_i})$, where $y_i$ represents the true class label and $\hat{p}_{i, y_i}$ denotes the predicted probability for that ground-truth class. NLL acts as a highly sensitive indicator of overconfident incorrect predictions. A severe logarithmic penalty is applied when the model predicts an incorrect class with near absolute certainty, making NLL an indispensable metric for assessing the safety and reliability of the probability vectors.[3, 6]

The Brier Score provides an alternative perspective, calculated as the mean squared error between the predicted probability distribution and the one-hot encoded true label vector: $BS = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (\hat{p}_{i,k} - y_{i,k})^2$. Unlike NLL, which only penalizes based on the predicted probability of the true class, the Brier Score natively incorporates the entire probability vector across all eight vehicle classes. The Brier score can be mathematically decomposed into three distinct components: Reliability (which isolates calibration error), Resolution (which measures the model's ability to discriminate between different underlying true probabilities), and Uncertainty (the inherent variance of the dataset). This decomposition allows engineers to isolate whether a poor Brier score originates from a lack of discriminatory power or a failure in calibration.[4, 7, 8]

### Expected Calibration Error and Variants

While strictly proper scoring rules are statistically robust, they combine resolution and calibration into a single continuous value, which can sometimes obscure whether the probability values themselves are misaligned. To isolate calibration specifically, bin-based metrics are utilized.

The Expected Calibration Error (ECE) is the most ubiquitous metric in modern machine learning literature. The continuous probability space between 0.0 and 1.0 is partitioned into $M$ equally spaced bins. For each bin, the absolute difference between the average empirical accuracy and the average predicted confidence is calculated, weighted by the proportion of samples falling into that bin: $\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$. While ECE is highly intuitive and easy to explain, its fundamental mathematical limitation in a multiclass setting is that it only evaluates the maximum predicted probability (the confidence of the argmax class) and completely ignores the calibration of the remaining $K-1$ class probabilities.[9, 10, 11]

To address the limitations of standard ECE in an eight-class system like AutoLens AI, the Classwise Expected Calibration Error (cwECE) is required. Rather than exclusively assessing the top-label prediction, cwECE computes the expected calibration error for each class individually across the entire probability spectrum and averages them: $\text{cwECE} = \frac{1}{K} \sum_{k=1}^K \text{ECE}_k$. This ensures that predictions for minority classes, or classes that frequently receive lower absolute prediction probabilities such as "OPEN WHEEL / F1", are still properly calibrated. This metric is essential for a user interface where the probabilities of the top three predicted classes might be displayed to the end-user.[10, 12]

Adaptive Expected Calibration Error (AdaECE) and Maximum Calibration Error (MCE) offer further specialized insights. Adaptive ECE utilizes equal-mass bins rather than equal-width bins, ensuring that each bin contains the same number of samples. This corrects for the heavy skew typical of deep learning predictions, which often cluster densely near 1.0. MCE measures the maximum discrepancy across any single bin, ignoring the weighted average entirely. This worst-case metric is critical for safety-critical environments where the maximum possible divergence between confidence and accuracy must be strictly bounded.[9, 12, 13]

### Reliability Diagrams

Reliability diagrams serve as the primary visual diagnostic tool for probability calibration, plotting empirical accuracy against predicted confidence per bin. A perfectly calibrated model exhibits a curve that perfectly tracks the $y=x$ diagonal identity line. Points falling below the diagonal indicate overconfidence, where the model's predicted probabilities exceed its actual accuracy. Points falling above the diagonal indicate underconfidence. In a multiclass system, relying solely on a global confidence reliability diagram is insufficient. Generating classwise reliability diagrams exposes specific semantic failure modes, allowing the engineer to diagnose if the EfficientNet-B2 model is universally overconfident on "SUV" predictions but systematically underconfident when predicting "MICRO" vehicles.[3, 4, 14]

## Algorithmic Analysis of Calibration Architectures

The selection of a post-hoc probability calibration algorithm is governed by a strict bias-variance tradeoff. Highly expressive models can correct complex, non-linear miscalibrations but risk catastrophic overfitting on small validation datasets. Conversely, simplistic models resist overfitting but may lack the mathematical capacity to fully rectify the probability simplex. The following sections provide an exhaustive analysis of nine distinct calibration methodologies, evaluating them against the specific constraints of the AutoLens AI project.

### 1. Temperature Scaling

Temperature Scaling is a single-parameter parametric approach and serves as a natural extension of Platt Scaling, specifically tailored for modern neural networks utilizing softmax activations. 

Mathematically, the unnormalized logit vector $\mathbf{z}$ produced by the final linear layer of the EfficientNet-B2 model is divided by a single learned global scalar $T > 0$ before the softmax function is applied: $\hat{q}_i = \frac{\exp(z_i / T)}{\sum_{j=1}^K \exp(z_j / T)}$. The parameter $T$ is optimized by minimizing the Negative Log-Likelihood on the hold-out validation set using algorithms such as L-BFGS.[2, 9, 15]

This method is exceptionally suitable for multiclass image classification. It natively handles multiclass outputs by operating directly on the logit vector prior to the softmax computation, thereby maintaining the relative exponential relationships between all classes.[4, 15] The risk of overfitting on a validation set of 3,157 samples across 8 classes is practically non-existent. Because the optimization landscape is constrained to a single scalar parameter across the entire dataset, the statistical capacity of the calibrator is incredibly low, entirely eliminating the risk of memorizing the validation split.[16, 17]

Crucially for the AutoLens AI project, Temperature Scaling is a strictly monotonic transformation. Because the same scalar $T$ is applied to all elements of the logit vector $\mathbf{z}$, the relative ordering of the logits remains identical. Therefore, Temperature Scaling cannot alter the predicted argmax class label. The Accuracy, Precision, Recall, and, most importantly, the primary model-selection metric of Macro F1 are guaranteed to remain entirely unchanged.[4, 18] 

Implementation in PyTorch is trivial, requiring only a custom module that wraps the base model, registers the temperature scalar as a buffer, and performs a simple tensor division. Exporting this architecture to ONNX Runtime is flawless; it manifests as a single scalar `Div` node inserted between the final linear layer and the softmax activation, utilizing standard highly optimized tensor operations. Consequently, it adds zero measurable latency during inference and its impact on the 95 MB artifact size limit is negligible, adding exactly one 32-bit floating-point value.

For a class project presentation, Temperature Scaling provides unparalleled explainability. The parameter $T$ acts directly as a dampening factor. A fitted temperature of $T=1.78$ clearly indicates to the audience that the raw model is overconfident, and the logits must be "softened" or shrunk by approximately 44% to align with reality. Conversely, a temperature $T < 1$ would indicate underconfidence, serving to sharpen the distribution.[2] The primary pros for AutoLens AI are its simplicity, zero risk of F1 degradation, and seamless ONNX compatibility. The only con is its limited mathematical expressiveness; it assumes miscalibration is uniform across all eight vehicle classes and cannot correct complex inter-class confusion or class-specific overconfidence.[1, 19]

### 2. Vector Scaling

Vector Scaling expands the representational capacity of Temperature Scaling by applying a distinct scaling factor and bias to each individual class in the output vector.

Mathematically, a weight vector $\mathbf{w}$ and a bias vector $\mathbf{b}$ are learned, and an element-wise affine transformation is applied to the logit vector: $\hat{q}_i = \frac{\exp(w_i z_i + b_i)}{\sum_{j=1}^K \exp(w_j z_j + b_j)}$. For an eight-class problem, this algorithm introduces $2K = 16$ learnable parameters.[20, 21]

This method is highly suitable for multiclass computer vision environments, allowing the network to adjust its confidence on a per-class basis. This is particularly valuable if the EfficientNet-B2 model exhibits systemic overconfidence on majority classes like "SEDAN" but is well-calibrated or underconfident on minority classes.[20] The risk of overfitting on 3,157 samples is low to moderate. While 16 parameters are easily constrained by thousands of samples, the optimization process typically requires a mild $L_2$ regularization penalty $\lambda \sum (w_i - 1)^2$ to prevent the scaling weights from drifting excessively far from unity, ensuring the calibrator focuses on minor corrections rather than drastic logit distortions.[22]

Unlike Temperature Scaling, Vector Scaling will change the predicted class labels. Because different classes are scaled and biased at different rates, a logit that was originally the second-highest could be scaled to become the argmax. This alters the decision boundaries of the network, which consequently impacts the Accuracy, Precision, Recall, and the primary Macro F1 metric.[23] While this can sometimes improve the F1 score by correcting inherent class biases, it risks degrading the highly optimized 92.39% baseline.

Implementation in PyTorch is straightforward, utilizing element-wise multiplication and addition. In the ONNX computational graph, this maps natively to highly accelerated `Mul` and `Add` nodes applied to the 1D logit tensor. It incurs sub-millisecond latency and the 16 additional parameters have zero practical impact on the 95 MB size limit. In terms of explainability, Vector Scaling can be presented as a mechanism that assigns a unique "confidence dampener" to each vehicle type, making it easy to illustrate which specific classes the raw model struggles with the most.

### 3. Matrix Scaling

Matrix Scaling represents the fully connected generalization of parametric logit transformation, applying a complete linear transformation to the output vector to capture complex cross-class correlations and structural confusion.

The mathematical formulation utilizes a full weight matrix and bias vector: $\hat{q} = \sigma_{\text{SM}}(\mathbf{W} \mathbf{z} + \mathbf{b})$, where $\mathbf{W} \in \mathbb{R}^{K \times K}$ and $\mathbf{b} \in \mathbb{R}^K$. For an eight-class system, this requires $K^2 + K = 72$ learnable parameters.[23] Theoretical studies demonstrating the optimality of calibration maps for models operating on Gaussian class-conditional distributions prove that the optimal recalibration function is fundamentally quadratic. Matrix Scaling provides a significantly closer linear approximation to this theoretical optimum than Temperature Scaling, providing a high degree of theoretical suitability for multiclass tasks.[19, 24]

However, the risk of overfitting on a restricted validation set of 3,157 samples is exceptionally high. Unregularized Matrix Scaling is notorious for aggressively memorizing small validation sets, causing catastrophic degradation in the test Negative Log-Likelihood and classwise Expected Calibration Error.[18, 25] With 72 parameters mapping the interactions between 8 classes, 3,157 samples (averaging less than 400 samples per class) is a dangerously sparse optimization landscape. To deploy Matrix Scaling safely, advanced regularization techniques such as Structured Matrix Scaling (SMS) or Off-Diagonal Intercept Regularization (ODIR) must be utilized to heavily penalize the off-diagonal elements, effectively tethering the calibration map to a simple identity matrix and only allowing cross-class scaling where statistically justified.[24, 26, 27]

Because it maps the logits through a dense linear layer, Matrix Scaling will aggressively alter class predictions. It shifts decision boundaries significantly, which poses a severe risk of degrading the primary Macro F1 metric. Implementation requires the insertion of a standard fully connected layer (`nn.Linear(8, 8)`) into the PyTorch pipeline. This exports seamlessly to ONNX Runtime via `MatMul` and `Add` operations, executing with negligible latency and maintaining full compliance with the 95 MB limit. Explainability is moderately complex but highly insightful; an audience can be shown the $8 \times 8$ weight matrix as a heatmap, where large off-diagonal values visually represent the calibrator automatically learning to correct the base model's systematic confusion between visually similar classes, such as "SUV" and "HATCHBACK".

### 4. Platt Scaling (One-vs-Rest)

Platt Scaling was originally architected to calibrate the outputs of Support Vector Machines by fitting a logistic regression model to the raw margin scores to produce pseudo-probabilities.

In a multiclass setting, Platt Scaling is traditionally implemented via a One-vs-Rest (OvR) paradigm. The system creates $K$ independent binary calibration problems, fitting a sigmoid function to each: $\hat{q}_i = \frac{1}{1 + \exp(- (a_i z_i + b_i))}$. Because these independently calibrated probabilities do not natively sum to 1.0, an artificial post-hoc $L_1$ normalization step is required to force the outputs back into a valid probability distribution: $\hat{q}_i = \frac{\hat{q}_i}{\sum \hat{q}_j}$.[28, 29]

The suitability of this method for deep multiclass image classification is fundamentally poor. Applying independent binary calibrations and artificially normalizing the outputs breaks the geometric properties of the probability simplex. It is mathematically inferior to methods that natively utilize the softmax function to handle joint probability distributions.[30, 31] While the overfitting risk on 3,157 samples is low to moderate (requiring 16 total parameters), the disjoint optimization guarantees that the predicted class labels will be altered, frequently corrupting ranking metrics and unpredictably degrading the F1 score.[15, 31]

Implementing Platt Scaling via a One-vs-Rest framework in PyTorch is cumbersome and exporting it cleanly to ONNX is problematic due to the complex split-and-normalize control flow required in the computational graph. Furthermore, its explainability is compromised by its theoretical flaws; it is difficult to justify artificially normalizing independent sigmoids when native softmax methods exist. The primary reason this method persists is its inclusion as the default multi-class calibrator in Scikit-Learn's `CalibratedClassifierCV`, which is why it often severely underperforms natively multiclass methods in deep learning contexts.[4, 15]

### 5. Isotonic Regression

Isotonic Regression is a non-parametric mathematical approach that seeks to minimize the mean squared error between predictions and true labels subject to a strict monotonicity constraint.

The algorithm minimizes $\sum (y_i - f_i)^2$ under the condition that $f_i \geq f_j$ whenever the uncalibrated input $z_i \geq z_j$. The resulting calibration map is a stepwise constant, monotonically increasing step function. Out of the box, Isotonic Regression only supports binary targets, meaning multiclass deployment relies entirely on the same flawed One-vs-Rest decomposition and subsequent $L_1$ normalization discussed in Platt Scaling.[4, 30] While novel mathematical variants such as Normalization-Aware Isotonic Regression (NA-FIR) exist to address this, they remain algorithmically complex.[30, 32]

The risk of overfitting Isotonic Regression on a validation set of 3,157 samples across 8 classes is extreme. Because the method is non-parametric, it relies entirely on the empirical data density to define its bin boundaries and step heights. A standard statistical rule of thumb dictates that Isotonic Regression requires a minimum of 1,000 samples per class to avoid learning the noise of the validation set; AutoLens AI possesses an average of only 394 samples per class. Applying it will result in severe overfitting, producing an highly irregular step function that memorizes the 3,157 validation points and generalizes poorly to the final instructor test data.[4, 17] 

Isotonic Regression introduces probability ties, generating wide flat regions on the step function where multiple different uncalibrated inputs map to the exact same calibrated output probability. This fundamentally destroys the continuous ranking dynamics of the model, altering the argmax predictions and violently disrupting metrics like F1, AUC, and ROC.[4]

Crucially, implementing and exporting Isotonic Regression to ONNX Runtime presents severe architectural issues. Non-parametric, stepwise functions do not map logically to static tensor multiplication. Implementing it requires complex control flow nodes, such as dynamic `If` statements, `Gather` operations, or large lookup tables. This dramatically inflates the ONNX artifact size, breaks hardware acceleration and kernel fusion within the ONNX execution provider, and significantly degrades inference latency.[33, 34] It is entirely unsuitable for a low-latency edge deployment.

### 6. Histogram Binning

Histogram binning is an older, simpler non-parametric technique that partitions the uncalibrated probability space into $M$ discrete, mutually exclusive bins. All predictions that fall within a specific bin are re-assigned a new probability equal to the average empirical accuracy of the samples within that bin.[9, 35]

Mathematically, the calibrated probability $\hat{q}_i = \theta_m$ if the uncalibrated probability $p_i \in B_m$, where $\theta_m$ represents the empirical accuracy of bin $m$. Like Isotonic Regression, its suitability for multiclass computer vision is highly suboptimal. Attempting to generate a true multidimensional histogram over an 8-class probability simplex is combinatorially explosive, meaning implementation again falls back to flawed One-vs-Rest binary reductions.

The overfitting risk on 3,157 samples is very high. Empty or sparse bins, which are common for minority classes or high-confidence edge cases, completely distort the calibration map. Histogram Binning destroys continuous probability rankings by grouping wide ranges of inputs into single, flat probability outputs, which modifies predictions and wreaks havoc on the Macro F1 score. 

Similar to Isotonic Regression, exporting a piecewise constant function to a static ONNX graph is highly inefficient, relying heavily on conditional logic tensors that create substantial deployment overhead, increase inference latency, and violate the core ethos of deploying smooth, continuous mathematical functions on tensor processors.

### 7. Dirichlet Calibration

Dirichlet Calibration represents the mathematical state-of-the-art for post-hoc multiclass probability calibration. Derived directly from Dirichlet distributions, it acts as the natively multiclass generalization of Beta calibration and is specifically designed to correct complex topological distortions within the deep learning probability simplex.

Mathematically, it operates not on the raw logits, but on the log-transformed probabilities (pseudo-logits). A linear transformation is applied to these log-probabilities, followed by a softmax operation to return a valid probability distribution: $\hat{q} = \sigma_{\text{SM}}(\mathbf{W} \ln(\mathbf{p}) + \mathbf{b})$. For an eight-class problem, this requires the same $K^2 + K = 72$ parameters as Matrix Scaling.[23, 27, 31] It is exceptionally suitable for multiclass computer vision tasks because it acts natively in the class probability space without requiring One-vs-Rest decomposition.[23, 31, 36]

The risk of overfitting on 3,157 samples is moderate to high without intervention. To prevent the 72 parameters from memorizing the validation split, the original authors introduced ODIR (Off-Diagonal and Intercept Regularization). ODIR is a specialized loss penalty that heavily penalizes the off-diagonal elements of the weight matrix $\mathbf{W}$ and the intercept vector $\mathbf{b}$. By applying ODIR, the calibration map is tightly regularized toward a simple scaling vector, allowing it to fit safely and robustly on a dataset as small as 3,157 samples while still retaining the capacity to model complex class confusion if strong statistical evidence exists.[27]

Dirichlet Calibration can and will alter class predictions, potentially modifying the Macro F1 score. Because the map corrects deep topological distortions in the simplex, it shifts the decision boundaries when class biases are heavily asymmetrical.[27] 

Implementation in PyTorch is straightforward but requires numerical care. It can be implemented as `torch.log` (with a small epsilon added to prevent NaN gradients on zero-probabilities), followed by an `nn.Linear` layer and an `F.softmax` activation. It exports flawlessly to ONNX Runtime, utilizing standard, highly accelerated tensor multiplication and exponential operations, adding sub-millisecond latency and less than 1 KB to the 95 MB model artifact. Explainability is exceptionally high for an academic presentation; the resulting $\mathbf{W}$ matrix can be displayed, where diagonal elements indicate intra-class confidence scaling, while the off-diagonal elements explicitly map the EfficientNet-B2 model's empirical confusion correlations between specific vehicle body types (e.g., statistically correcting the network's tendency to confuse a "STATION WAGON" for a "HATCHBACK").[27, 31]

### 8. Monte Carlo Dropout and Deep Ensembles

Monte Carlo Dropout (MC-Dropout) and Deep Ensembles are not traditional post-hoc probability recalibrators, but rather Bayesian approximation techniques designed to map and quantify epistemic uncertainty.

Deep Ensembles average the softmax outputs of $M$ independently trained network architectures. MC-Dropout mathematically approximates a Gaussian Process by performing $M$ distinct stochastic forward passes through a single network, leaving the dropout layers enabled during inference time: $\hat{q} = \frac{1}{M} \sum_{m=1}^M f_{\text{drop}_m}(x)$. The variance across these passes quantifies the model's uncertainty.[37, 38, 39]

While these methods are considered the gold standard for deep epistemic uncertainty quantification and perform exceptionally well on multiclass medical and autonomous driving tasks [40, 41], they violently violate the strict architectural and deployment constraints of the AutoLens AI project. 

Deep Ensembles require storing $M$ copies of the model. Given that a single FP32 EfficientNet-B2 occupies approximately 35 MB, an ensemble of just three models will breach the strict 95 MB artifact size constraint. MC-Dropout does not increase model size, but it requires dynamic graph execution and loop constructs at inference time to perform $M$ stochastic forward passes. This dramatically complicates ONNX exports, essentially requiring the ONNX graph to manage stochastic number generators and loops natively, while simultaneously multiplying the inference latency $M$-fold.[42, 43, 44] They are mathematically fascinating but structurally incompatible with low-latency edge criteria.

### 9. Label Smoothing and Focal Loss

Label Smoothing and Focal Loss are training-time interventions that inherently modify the loss landscape to produce more uniformly distributed probabilities natively. 

Label smoothing replaces one-hot targets with a smoothed distribution $y_{LS} = (1-\alpha)y + \alpha/K$, penalizing the network for becoming excessively confident during gradient descent. Focal Loss scales the standard cross-entropy loss by a modulating factor $(1-p_t)^\gamma$ to down-weight the loss contribution of easy, high-confidence examples, forcing the optimizer to focus on harder edge cases.[7, 45, 46]

While highly suitable for multiclass computer vision tasks, these are not post-hoc calibrators. Implementing them requires completely discarding the current EfficientNet-B2 checkpoint and initiating a full retraining cycle from scratch, violating the premise of the research request which asks how to calibrate the *current working candidate*. Furthermore, recent research has indicated that while label smoothing generates lower ECE scores in-distribution, it artificially flattens the logit landscape, which can severely degrade a model's ability to selectively reject out-of-distribution inputs or signal true uncertainty under dataset shift.[6, 45]

## Method Comparison Matrix

The table below synthesizes the analysis of the nine methodologies, mapping their mathematical traits directly against the strict deployment constraints of the AutoLens AI architecture.

| Calibration Method | Parameter Count | Preserves F1 (Argmax)? | ONNX Exportability & Latency | Overfitting Risk (N=3,157) | Suitability for AutoLens |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Temperature Scaling** | 1 | **Yes** | Flawless (`Div` node); Zero Latency | None | **Excellent** |
| **Vector Scaling** | 16 ($2K$) | No | Flawless (`Mul`, `Add`); Zero Latency | Low (requires $L_2$) | Good |
| **Matrix Scaling** | 72 ($K^2+K$)| No | Flawless (`MatMul`); Zero Latency | Extreme (requires SMS/ODIR)| Poor |
| **Platt Scaling (OvR)** | 16 ($2K$) | No | Suboptimal (Split + Norm Logic) | Low | Poor |
| **Isotonic Regression** | Non-parametric | No | **Failure (Requires Step Functions)** | **Catastrophic** | Unusable |
| **Histogram Binning** | Non-parametric | No | **Failure (Requires Step Functions)** | **Catastrophic** | Unusable |
| **Dirichlet Calibration**| 72 ($K^2+K$)| No | Flawless (`Log`, `MatMul`); Low Latency| Moderate (requires ODIR) | Good |
| **MC-Dropout** | N/A | No | **Failure (Requires stochastic loops)** | N/A | Unusable |
| **Deep Ensembles** | $M \times$ Base | No | **Failure (Breaches 95 MB Limit)** | N/A | Unusable |

## Validation Protocol Design

A critical constraint for AutoLens AI is ensuring that the internal test dataset and the instructor final test dataset are never exposed to the calibration algorithm during the fitting phase. Calibration algorithms, especially those with higher capacity like Vector Scaling or Dirichlet Calibration, require the optimization of hyperparameters, specifically the $L_2$ penalty $\lambda$ or the ODIR regularization term. If the entire 3,157-sample validation set is used simultaneously to fit the calibrator and calculate the final ECE score, the reported metrics will suffer from data leakage and exhibit overly optimistic calibration scores that will fail to generalize to the instructor's final test data.

To prevent data leakage while safely selecting and tuning the calibration method, a Nested Cross-Validation protocol must be executed exclusively on the validation split:

1. **Feature Extraction Freezing:** Do not run the full EfficientNet-B2 model during the calibration loop. Execute a single forward pass over the 3,157 validation images to extract the uncalibrated logit matrix $\mathbf{Z}_{val} \in \mathbb{R}^{3157 \times 8}$ and the true labels $\mathbf{y}_{val}$. Freeze these tensors to RAM.
2. **K-Fold Partitioning:** Split the 3,157 validation samples into 5 stratified folds (approximately 2,525 samples for fitting the calibrator, 632 samples for assessing the penalty).
3. **Hyperparameter Grid Search:** Iterate over a grid of regularization parameters ($\lambda$). For Dirichlet Calibration, this defines the strength of the ODIR penalty. For Vector Scaling, this defines the $L_2$ penalty. Fit the calibration matrix on the 4 active folds, and measure the strictly proper scoring rule (Negative Log-Likelihood) on the 1 out-of-fold set.
4. **Optimal Selection:** Average the out-of-fold NLL scores across all 5 folds for each $\lambda$. Identify the calibration method and the specific $\lambda^*$ that yields the lowest cross-validated NLL, ensuring that the model is maximizing true generalizable likelihood rather than memorizing the fold.
5. **Final Production Fit:** Once the optimal algorithm and hyperparameter $\lambda^*$ are identified, discard the folds. Fit the final calibrator instance on the *entirety* of the 3,157 validation samples using $\lambda^*$.
6. **ONNX Graph Compilation:** Append the learned final parameters to the PyTorch graph and export the model to ONNX. Because the internal test and instructor test sets were physically entirely separate from this process, the final evaluation on those sets will be mathematically pure and free of any data leakage.[4, 16, 17]

## Architectural Implementation Guidance

When modifying the computational graph for deployment, it is highly recommended to implement the post-hoc calibration layer explicitly as a standard `nn.Module` in PyTorch, wrapping the EfficientNet-B2 baseline, and exporting the combined sequence via `torch.onnx.export`. This prevents the need for manual ONNX Proto manipulation or complex ONNX GraphSurgeon interventions, which are highly error-prone and brittle.

### Implementing Temperature Scaling (Top Recommendation)

Because Temperature Scaling preserves the argmax, it is the safest implementation for preserving the primary F1 metric required by the project constraints. The temperature scalar must be registered as a PyTorch buffer so that the ONNX exporter inherently recognizes it as a persistent, non-trainable parameter of the static graph rather than an arbitrary Python variable.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperatureScaledEfficientNet(nn.Module):
    def __init__(self, base_model, temperature_val=1.78):
        super().__init__()
        self.base_model = base_model
        # Registered as a buffer so it exports with the model parameters natively
        # into the safetensors/ONNX graph without requiring an independent variable.
        self.register_buffer('temperature', torch.tensor([temperature_val]))

    def forward(self, x):
        # Extract uncalibrated logits from EfficientNet-B2
        logits = self.base_model(x)
        
        # Broadcasting scalar division across the 8-class dimension
        scaled_logits = logits / self.temperature 
        
        # Output calibrated probability simplex natively
        return F.softmax(scaled_logits, dim=1)

# Deployment Export Execution:
# base_enet = load_efficientnet_safetensors(...)
# scaled_model = TemperatureScaledEfficientNet(base_enet, temperature_val=1.78)
# scaled_model.eval()
# torch.onnx.export(scaled_model, dummy_input, "autolens_calibrated.onnx",
#                   input_names=['input'], output_names=['output'],
#                   dynamic_axes={'input': {0: 'batch_size'}})
```

### Implementing Dirichlet Calibration (Secondary Recommendation)

If empirical analysis via Classwise Expected Calibration Error reveals severe inter-class miscalibration that necessitates Dirichlet Calibration, the PyTorch implementation relies on `torch.log`. Note that raw probabilities must be rigorously stabilized with a small epsilon scalar before applying the logarithm to prevent catastrophic `NaN` errors during ONNX evaluation or gradient tracing.

```python
class DirichletCalibratedEfficientNet(nn.Module):
    def __init__(self, base_model, W_matrix, b_vector):
        super().__init__()
        self.base_model = base_model
        # W_matrix is 8x8, b_vector is 8x1
        self.calibration_layer = nn.Linear(8, 8)
        self.calibration_layer.weight.data = torch.tensor(W_matrix, dtype=torch.float32)
        self.calibration_layer.bias.data = torch.tensor(b_vector, dtype=torch.float32)

    def forward(self, x):
        logits = self.base_model(x)
        raw_probs = F.softmax(logits, dim=1)
        
        # Epsilon added for strict numerical stability in ONNX log operation
        # Prevents log(0) which yields -Inf and crashes inference pipelines
        log_probs = torch.log(raw_probs + 1e-9) 
        
        # Apply the learned W matrix and b vector
        calibrated_pseudo_logits = self.calibration_layer(log_probs)
        
        return F.softmax(calibrated_pseudo_logits, dim=1)
```

Both implementations are completely natively supported by ONNX Runtime, execute in microseconds, and effortlessly feed directly into a Gradio UI, mapping the output tensor dynamically to the eight target class string labels.

## Conclusion and Ranked Recommendations

Based on the intersection of mathematical robustness, deployment limitations (ONNX compatibility, F1 preservation, 95 MB size limit), and sample size limitations (3,157 samples), the post-hoc calibration architectures are ranked as follows:

1. **Temperature Scaling (Strongest Recommendation):** The current working baseline is mathematically optimal given the strict project constraints. It requires zero hyperparameter regularization, is mathematically immune to overfitting on a validation set of 3,157 samples, exports flawlessly to ONNX as a single division node, and—critically—is the only method logically guaranteed to preserve the optimized Macro F1 score of the EfficientNet-B2 baseline. The already observed reduction of NLL (0.297 to 0.241) and ECE (0.038 to 0.010) indicates that a global scalar is highly successful at mitigating the model's structural margin scaling issues.[1, 2]
2. **Dirichlet Calibration with ODIR Regularization:** If further diagnostic analysis via Classwise ECE (cwECE) or per-class reliability diagrams reveals that specific vehicle body types exhibit extreme calibration asymmetry (e.g., universal overconfidence on "VAN" but systemic underconfidence on "SUV"), Dirichlet calibration serves as the state-of-the-art alternative. It models inter-class confusion directly and natively on the probability simplex.[23, 27, 31] However, it absolutely requires careful nested cross-validation to tune the ODIR regularization to prevent overfitting, and it inherently risks shifting the decision boundaries, which could slightly degrade the Macro F1 metric.[27]
3. **Vector Scaling (with $L_2$ Regularization):** A moderate middle-ground offering per-class scaling factors without the massive parameter jump of Matrix or Dirichlet models. Like Dirichlet, it alters predictions and requires regularization to prevent logit drift. 

Methods relying on stepwise non-parametric functions or disjoint binary normalizations—specifically Isotonic Regression, Histogram Binning, and One-vs-Rest Platt Scaling—should be discarded entirely. They create massive friction within ONNX Runtime graphs, inflate artifact size with complex logic gates, and will drastically overfit on a data split of 3,157 samples across eight visual classes.[4, 30, 33, 34]

### Final Resolution

**"Should AutoLens AI keep Temperature Scaling or try another method?"**

**Keep Temperature Scaling.** For a deployment-constrained ONNX pipeline where Macro F1 remains the primary evaluation metric governing model selection, Temperature Scaling is the mathematically safest and most structurally sound choice. It guarantees that the argmax class predictions—and thus the F1, Precision, and Recall scores—remain mathematically untouched, while simultaneously driving the global ECE down to a highly reliable 1.04%. Alternative methods should only be investigated if a classwise reliability diagram exposes systemic, disproportionate miscalibration on minority visual classes that a global continuous scalar inherently cannot resolve.