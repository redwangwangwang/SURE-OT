# SURE-OT Method

## 1. Motivation

Balanced optimal transport fixes the row and column marginals of the transport plan. Consequently, a change score based only on row/column transported mass is largely determined by the prescribed marginals. This is unsuitable for disease evolution, because newly emerged and resolved findings are precisely the regions that may not have a credible temporal counterpart.

SURE-OT treats longitudinal alignment as **correspondence plus birth and death**.

## 2. Unbalanced transport

For current patch features $X^c$ and historical patch features $X^p$, SURE-OT minimizes an entropic UOT objective:

$$
\langle C,P\rangle + \epsilon \sum_{ij}P_{ij}(\log P_{ij}-1)
+\tau\,\mathrm{KL}(P\mathbf 1\|a)
+\tau\,\mathrm{KL}(P^\top\mathbf 1\|b).
$$

The semantic cost uses cosine distance, with an optional normalized patch-coordinate penalty.

The implementation uses log-domain generalized Sinkhorn updates with

$$
\rho=\frac{\tau}{\tau+\epsilon}.
$$

Setting `--sure_ot_balanced True` uses $\rho=1$, providing a balanced-OT ablation.

## 3. Residual evolution maps

For a current-to-history plan $P^{c\rightarrow p}$, the normalized birth residual is

$$
r_i^{new}=\left[\frac{a_i-\sum_j P^{c\rightarrow p}_{ij}}{a_i}\right]_{[0,1]}.
$$

For a history-to-current plan $P^{p\rightarrow c}$, the resolution residual is

$$
r_j^{resolved}=\left[\frac{b_j-\sum_i P^{p\rightarrow c}_{ji}}{b_j}\right]_{[0,1]}.
$$

Matched mass yields persistent evidence. Row-wise transport entropy and bidirectional mass disagreement yield uncertain evidence.

## 4. Continuous evolution tokens

Four learnable query banks pool visual features using the soft residual maps as attention priors:

- new;
- resolved;
- persistent;
- uncertain.

For query $q_k$, feature $x_i$, and prior $r_i$,

$$
\alpha_{ki}=\mathrm{softmax}_i\left(\frac{\langle q_k,x_i\rangle}{\sqrt d}+\eta\log(r_i+\delta)\right).
$$

The resulting tokens are inserted directly into the LLM input sequence. No patch index is converted into text, and gradients flow from report generation back through the token pooling and UOT module.

## 5. Temporal swap consistency

The pair is evaluated in both orders:

$$
(X^p,X^c),\qquad(X^c,X^p).
$$

SURE-OT applies:

- map consistency between forward new and reverse resolved residuals;
- token consistency between the corresponding evolution token groups;
- transport-plan consistency between a forward plan and the transpose of its reverse counterpart.

Temporal role adapters make current/history processing direction-aware while remaining parameter-efficient.

## 6. Training objective

The original report-generation and vision-language consistency terms are retained. The added objective is

$$
\mathcal L = \mathcal L_{RRG}+\mathcal L_{VL}+\lambda_{swap}\mathcal L_{swap}+\lambda_{uot}\mathcal L_{uot}+\lambda_{reg}\mathcal L_{reg}.
$$

$\mathcal L_{reg}$ combines residual sparsity and spatial total variation.

## 7. Dataset contract

SURE-OT consumes the same fields as BiOTPrompt:

- current image;
- historical image;
- current report;
- existing current disease labels.

No dataset split, annotation file, image, report, mask, box, or progression label is added or rewritten.
