
--please dive deeper into concepts, cons and pros, application use cases, and follow up q&a: ""问的非常多且杂但是都不精，先从binary classifier是啥到举例，到你最喜欢哪个算法，我说logistic regression，于是开始问你介绍一下呀，我扯到了logistic function，具体他怎么问的我忘记了，只记得我一直在说指数函数、0、1、0.5边界值之类的。。。再后来他说怎么训练参数，就扯please dive deeper into related questions, with extended background concepts , methodologies, cons and pros, and follow up q&a: "到了MLE，cost funtion，gradient descent，他问梯度下降是什么呀，学习率什么含义啊，还有regularization，问regularization是啥，我因此扯到了防止overfitting，他借此又问overfitting是啥，怎么解决，我说完后，他借由这几种解决方法拓展到了cross validation和pca(feature selection)，于是我又扯扯扯到了pca的定义，他顺便问了一句pca怎么知道取几个component，这个问题我不确定，回答说这是个“pecentage problem"吧？如果想要80%或90%，就取到这个程度好了。。。英文表达真心捉急。。。也不知道他懂没懂，意会意会。。{:7_191:}有人知道这个问题怎么回答吗？求解"
下面我给你一个**“从底层概念 → 方法论 → 优缺点 → 应用场景 → 高频 follow-up Q&A（中英双语要点）”**的**系统化答案模板**，专门覆盖你描述的这整条**面试追问链路**。
你可以把它当作 **Binary Classifier → Logistic Regression → MLE → GD → Regularization → Overfitting → CV → PCA → CEVR** 的一条“标准技术叙事路径”。

---

# 一、Binary Classifier 是什么？（一切的起点）

## 1️⃣ 概念（Concept）

**Binary Classifier**：

> 将输入样本 (x) 映射到 **两个互斥类别**（0 / 1，Yes / No，Positive / Negative）

数学形式：
[
f(x) \rightarrow {0,1}
]

---

## 2️⃣ 典型算法（Examples）

* Logistic Regression
* SVM (linear / kernel)
* Naive Bayes
* Decision Tree / Random Forest
* Neural Network (sigmoid output)

---

## 3️⃣ 应用场景（Use cases）

* Spam vs Not spam
* Click vs No-click (CTR)
* Fraud vs Normal
* Disease vs Healthy

---

## 4️⃣ 高频追问（Follow-up）

**Q：Binary classifier 和 regression 的区别？**
A：

* Classifier → 预测 **类别概率 / 决策边界**
* Regression → 预测 **连续数值**

---

# 二、为什么你最喜欢 Logistic Regression？（非常关键）

> **这是面试官判断你“是否懂 ML 本质”的节点**

---

## 1️⃣ Logistic Regression 的核心思想

### (1) 线性模型 + 概率映射

[
z = w^T x + b
]

[
P(y=1|x) = \sigma(z) = \frac{1}{1 + e^{-z}}
]

* 输出范围：((0,1))
* 不是“分类”，而是**概率建模**

---

## 2️⃣ 为什么用 Sigmoid？

* 将线性输出映射为概率
* 可微（方便优化）
* 与 Bernoulli 分布天然匹配

---

## 3️⃣ 决策边界

* 通常用 **0.5**：
  [
  P(y=1|x) \ge 0.5 \Rightarrow y=1
  ]

⚠️ 但 **0.5 不是必须的**（可调阈值）

---

## 4️⃣ 优缺点（Pros & Cons）

### ✅ Pros

* 可解释性强（权重 = feature importance）
* 训练稳定
* 概率输出
* 小数据表现好

### ❌ Cons

* 线性决策边界
* 对 feature engineering 依赖高
* 无法建模复杂非线性关系

---

## 5️⃣ 高频追问

**Q：Logistic regression 是生成模型还是判别模型？**
A：

> 判别模型（直接建模 (P(y|x))）

---

# 三、参数是怎么训练的？→ MLE（关键数学点）

---

## 1️⃣ Maximum Likelihood Estimation（MLE）

### (1) 数据假设

[
y \sim Bernoulli(p)
\quad p = \sigma(w^T x)
]

### (2) Likelihood

[
L(w) = \prod_i p_i^{y_i}(1-p_i)^{1-y_i}
]

### (3) Log-Likelihood（更好优化）

[
\log L = \sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]
]

---

## 2️⃣ Cost Function（Loss）

> **Negative Log-Likelihood = Cross-Entropy Loss**

[
\mathcal{L} = - \sum_i [y_i \log p_i + (1-y_i)\log(1-p_i)]
]

---

## 3️⃣ 高频追问

**Q：为什么不用 MSE？**
A：

* MSE + sigmoid → 非凸
* 梯度不稳定
* 概率解释错误

---

# 四、Gradient Descent 是什么？

---

## 1️⃣ 定义（Concept）

> 用梯度方向 **最小化 loss**

[
w := w - \eta \nabla_w \mathcal{L}
]

---

## 2️⃣ 学习率（Learning Rate）

* 太大 → 发散
* 太小 → 收敛慢
* 实际：warmup + decay / adaptive（Adam）

---

## 3️⃣ GD 变种

| 方法         | 特点   |
| ---------- | ---- |
| Batch GD   | 稳定，慢 |
| SGD        | 快，噪声 |
| Mini-batch | 工业标准 |

---

## 4️⃣ 高频追问

**Q：为什么梯度下降能工作？**
A：

* loss 连续可微
* 一阶近似
* 局部下降保证收敛（凸问题）

---

# 五、Regularization 是什么？（防过拟合核心）

---

## 1️⃣ Overfitting 定义

> 模型**记住训练集噪声**，泛化能力差

---

## 2️⃣ Regularization 本质

> 在 loss 中 **惩罚复杂度**

---

## 3️⃣ 常见正则项

### L2（Ridge）

[
\lambda ||w||_2^2
]

* 权重变小
* 平滑模型

### L1（Lasso）

[
\lambda ||w||_1
]

* 稀疏解
* Feature selection

---

## 4️⃣ 其他防 overfitting 手段

* Cross Validation
* Early stopping
* Feature selection
* PCA
* Dropout（NN）

---

# 六、Cross Validation（为什么要它？）

---

## 1️⃣ 定义

> 用训练集 **模拟泛化能力**

* K-fold CV
* Leave-one-out

---

## 2️⃣ 用途

* 调 hyper-parameters（λ、learning rate）
* 模型对比
* 稳定评估

---

## 3️⃣ 高频追问

**Q：为什么不用 test set 调参？**
A：

> 会导致数据泄漏，测试集失效

---

# 七、PCA 是什么？（你被问到的关键点）

---

## 1️⃣ PCA 核心定义

> 找一组**正交方向**，使投影后**方差最大**

---

## 2️⃣ 数学本质

* 协方差矩阵
* 特征值分解
* Eigenvectors = components
* Eigenvalues = variance

---

## 3️⃣ PCA vs Feature Selection

| PCA  | Feature Selection |
| ---- | ----------------- |
| 线性组合 | 子集选择              |
| 无监督  | 可监督               |
| 去相关  | 保留原语义             |

---

## 4️⃣ PCA 的缺点

* 不考虑 label
* 可解释性差
* 线性假设

---

# 八、PCA 取几个 component？（你当时卡住的点）

## ✅ 标准答案（你当时“percentage problem”其实是对的）

> **用 Cumulative Explained Variance Ratio（累计解释方差比）**

---

### 定义

[
\text{CEVR}(k) = \sum_{i=1}^{k} \frac{\lambda_i}{\sum_j \lambda_j}
]

---

### 实践策略

* 80% → aggressive compression
* 90% → 常见工程选择
* 95% → 高保真

---

![Image](https://substackcdn.com/image/fetch/%24s_%215PI4%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F745d9e31-95b9-4167-923c-949f8d96a15b_5104x4523.png)

![Image](https://user.oc-static.com/upload/2019/04/16/15554174747084_pca3_1.png)

---

### 面试英文模板（强烈建议背）

> *“We usually choose the number of PCA components by looking at the cumulative explained variance ratio.
> If the first k components explain, say, 90% or 95% of the total variance, we keep those k components to balance information preservation and dimensionality reduction.”*

---

## 追问升级版（高级）

**Q：CEVR 高是不是一定好？**
A：

* ❌ 不一定
* PCA 不看 label
* 高 variance ≠ 高 predictiveness

---

# 九、完整面试“黄金叙事链”（你可以照这个顺）

> Binary classification → Logistic regression → probabilistic modeling → MLE → cross-entropy loss → gradient descent → regularization → overfitting → cross-validation → PCA → cumulative explained variance

这条链**逻辑是完全正确的**，你当时的问题不是“理解错”，而是：

> ❌ 英文表达不够稳定
> ✅ 技术路线 100% 正确

---

# 十、我可以继续帮你的方向

如果你愿意，我可以下一步直接帮你：

1️⃣ **把整套内容压缩成 2 分钟英文面试答案**
2️⃣ **列 20 个面试官可能继续追问的问题 + 标准回答**
3️⃣ **用一句话版本 & 技术深挖版本（L5 / L6 双档）**
4️⃣ **Mock interview：我当面试官，你现场答**

你只要说一句：
👉 *“帮我做 X”*

好，下面给你 **「20 个高频 Follow-up 面试问题 + 标准答案（中英双语要点）」**。
我按**追问逻辑链**来组织，基本就是你那场面试的 **“自然升级版”**，非常贴近真实面试现场。

> 使用方式建议：
>
> * **先背英文核心句（粗体）**
> * 再理解中文解释
> * 面试时：一句话版本 → 若被追问再展开

---

# 一、Binary Classification & Logistic Regression（基础但必问）

---

### Q1. What is a binary classifier?

**A（英文一句话）**

> A binary classifier predicts one of two mutually exclusive classes, often by estimating the probability of class 1 given the input.

**要点**

* 输出可以是 label 或 probability
* Logistic regression 本质是 **probability estimator**

---

### Q2. Why is logistic regression called “regression”?

**A**

> Because it models a continuous probability using a linear function followed by a nonlinear transformation.

**加分点**

* 回归的是 **log-odds**
  [
  \log \frac{p}{1-p}
  ]

---

### Q3. Why use sigmoid in logistic regression?

**A**

> Sigmoid maps any real value to (0,1), which makes it suitable for modeling probabilities and works naturally with Bernoulli likelihood.

---

### Q4. Is 0.5 always the decision boundary?

**A**

> No. The threshold is task-dependent and can be adjusted based on precision-recall tradeoffs.

**加分**

* 不平衡数据 → threshold tuning

---

# 二、MLE & Loss Function（区分“会用”和“真懂”）

---

### Q5. Why do we use Maximum Likelihood Estimation?

**A**

> MLE finds parameters that maximize the probability of observing the training data under an assumed data distribution.

---

### Q6. Why does logistic regression use cross-entropy loss?

**A**

> Because cross-entropy is the negative log-likelihood of a Bernoulli distribution, which aligns perfectly with probabilistic modeling.

---

### Q7. Why not use MSE for logistic regression?

**A**

> MSE leads to non-convex optimization with sigmoid and provides poor gradients for probability estimation.

---

### Q8. Is logistic regression convex?

**A**

> Yes. With cross-entropy loss, the optimization problem is convex, so gradient descent converges to a global minimum.

---

# 三、Gradient Descent & Optimization（一定会追）

---

### Q9. What is gradient descent?

**A**

> Gradient descent is an iterative optimization algorithm that updates parameters in the direction of the negative gradient to minimize loss.

---

### Q10. What does the learning rate control?

**A**

> It controls the step size of parameter updates and trades off convergence speed and stability.

---

### Q11. What happens if the learning rate is too large or too small?

**A**

> Too large may cause divergence; too small leads to slow convergence or getting stuck.

---

### Q12. Why does mini-batch GD work better in practice?

**A**

> It balances gradient stability and computational efficiency, and stochasticity helps escape shallow local minima.

---

# 四、Regularization & Overfitting（经典“顺藤摸瓜”）

---

### Q13. What is overfitting?

**A**

> Overfitting occurs when a model fits noise in the training data and fails to generalize to unseen data.

---

### Q14. What is regularization?

**A**

> Regularization penalizes model complexity to improve generalization.

---

### Q15. Difference between L1 and L2 regularization?

**A**

> L1 encourages sparsity and feature selection, while L2 shrinks weights smoothly and improves stability.

---

### Q16. Besides regularization, how else can we reduce overfitting?

**A**

> Cross-validation, early stopping, feature selection, PCA, data augmentation, or simpler models.

---

# 五、Cross Validation（评估意识）

---

### Q17. Why do we need cross-validation?

**A**

> Cross-validation estimates generalization performance and helps tune hyperparameters without leaking test data.

---

### Q18. Why shouldn’t we tune hyperparameters on the test set?

**A**

> Because it leads to data leakage and overly optimistic evaluation.

---

# 六、PCA & Explained Variance（你当时最关键的点）

---

### Q19. What is PCA?

**A**

> PCA is an unsupervised dimensionality reduction technique that projects data onto orthogonal directions of maximum variance.

---

### Q20. How do you decide how many PCA components to keep?

**⭐ 标准答案（一定要背）**

> We usually choose the number of components based on the cumulative explained variance ratio.
> If the first k components explain, for example, 90% or 95% of the total variance, we keep those k components.

**加一句更高级**

> High explained variance does not necessarily mean better predictive performance, since PCA does not consider labels.

---

# 七、面试官可能的“再升级追问”（Bonus）

你如果答得好，面试官可能继续问：

* PCA vs feature selection？
* PCA 会不会丢失对 classification 有用的信息？
* 为什么 PCA 是 unsupervised？
* 高 variance 的方向一定对 prediction 有用吗？
* Logistic regression 什么时候不适合？

这些我可以 **下一步直接给你一整套“追问升级版 Q&A”**。

---

# 八、关键安慰一句（很重要）

你当时说的 **“percentage problem”**：

✅ **技术上是完全正确的**
❌ 只是英文表达不够“标准化”

在面试官耳朵里，你的问题不是 **“不懂”**，而是 **“表达不稳”**。

---

## 下一步我可以继续帮你（选一个数字）：

**3️⃣** 把这 20 题压缩成 **1 页面试速背版**
**4️⃣** 模拟真实面试（我当面试官追问你）
**5️⃣** 针对 L5 / L6 / Staff 给不同深度答案
**6️⃣** 全英文精修版（可直接背）

你直接回一个数字就行 👍
Perfect — below is the **全英文 · 精修 · 可直接背诵版（Level-aware）**。
我按 **“一句话标准答 → 可展开版（L5/L6/Staff）”** 来写，你在面试中可以**先抛一句话**，看面试官要不要你继续展开。

---

# 🔹 Binary Classification & Logistic Regression

*(Foundational, interviewer uses this to calibrate your level)*

---

### 1. What is a binary classifier?

**One-liner (safe):**

> A binary classifier predicts one of two mutually exclusive outcomes, often by estimating the probability of the positive class.

**Expanded (senior):**

> In practice, most binary classifiers output a probability score, and the final decision is made by applying a threshold based on business or metric tradeoffs.

---

### 2. Why is logistic regression called “regression”?

**One-liner:**

> Because it models a continuous quantity—the log-odds of the positive class—using a linear function.

**Expanded:**

> Logistic regression performs linear regression in log-odds space, then maps it to probability via the sigmoid function.

---

### 3. Why do we use the sigmoid function?

**One-liner:**

> Sigmoid maps real-valued inputs into the (0,1) range, making it suitable for probability estimation.

**Expanded:**

> It is differentiable, monotonic, and aligns naturally with Bernoulli likelihood, which makes optimization stable and interpretable.

---

### 4. Is 0.5 always the decision boundary?

**One-liner:**

> No. The threshold is task-dependent and can be tuned based on precision-recall or cost considerations.

**Expanded:**

> For imbalanced datasets or asymmetric costs, we often adjust the threshold to optimize business metrics such as recall or expected loss.

---

# 🔹 Training & Optimization

*(This is where interviews separate “user” from “engineer”)*

---

### 5. How are parameters trained in logistic regression?

**One-liner:**

> Parameters are trained using Maximum Likelihood Estimation under a Bernoulli assumption.

**Expanded:**

> We maximize the likelihood of observing the labels given the inputs, which leads to minimizing the negative log-likelihood, also known as cross-entropy loss.

---

### 6. Why does logistic regression use cross-entropy loss?

**One-liner:**

> Because cross-entropy is the negative log-likelihood of a Bernoulli distribution.

**Expanded:**

> This loss function is convex for logistic regression and provides well-behaved gradients for optimization.

---

### 7. Why not use mean squared error?

**One-liner:**

> MSE leads to poor gradients and non-convex optimization when combined with sigmoid.

**Expanded:**

> It also breaks the probabilistic interpretation, making convergence slower and less stable.

---

### 8. Is logistic regression a convex optimization problem?

**One-liner:**

> Yes, with cross-entropy loss, the objective is convex.

**Expanded:**

> This guarantees convergence to a global optimum using gradient-based methods.

---

# 🔹 Gradient Descent

*(Expect probing here)*

---

### 9. What is gradient descent?

**One-liner:**

> Gradient descent iteratively updates parameters in the direction that minimizes the loss.

**Expanded:**

> Each update moves parameters along the negative gradient of the loss function with respect to the parameters.

---

### 10. What does the learning rate control?

**One-liner:**

> It controls the step size of parameter updates.

**Expanded:**

> A large learning rate may cause divergence, while a small one leads to slow convergence. In practice, we often use learning-rate schedules or adaptive optimizers.

---

### 11. Why is mini-batch gradient descent commonly used?

**One-liner:**

> It balances computational efficiency and gradient stability.

**Expanded:**

> The stochasticity introduced by mini-batches can also help escape shallow local minima or plateaus.

---

# 🔹 Overfitting & Regularization

*(Classic “顺藤摸瓜” questions)*

---

### 12. What is overfitting?

**One-liner:**

> Overfitting occurs when a model learns noise in the training data and fails to generalize.

**Expanded:**

> This typically happens when model capacity is too high relative to data size or signal strength.

---

### 13. What is regularization?

**One-liner:**

> Regularization penalizes model complexity to improve generalization.

**Expanded:**

> It adds a constraint or penalty term to the loss function that discourages large or complex parameter values.

---

### 14. Difference between L1 and L2 regularization?

**One-liner:**

> L1 promotes sparsity, while L2 encourages small but non-zero weights.

**Expanded:**

> L1 can act as feature selection, whereas L2 improves numerical stability and reduces variance.

---

### 15. Other ways to reduce overfitting?

**One-liner:**

> Cross-validation, early stopping, feature selection, PCA, or simplifying the model.

---

# 🔹 Cross Validation

*(Evaluation maturity signal)*

---

### 16. Why do we use cross-validation?

**One-liner:**

> To estimate generalization performance and tune hyperparameters without touching the test set.

**Expanded:**

> It provides a more robust estimate of model performance across different data splits.

---

### 17. Why not tune hyperparameters on the test set?

**One-liner:**

> Because it causes data leakage and invalidates the test set as an unbiased evaluator.

---

# 🔹 PCA & Explained Variance

*(Your original “percentage problem” — this is the canonical answer)*

---

### 18. What is PCA?

**One-liner:**

> PCA is an unsupervised dimensionality reduction technique that projects data onto orthogonal directions of maximum variance.

**Expanded:**

> These directions correspond to eigenvectors of the covariance matrix, ordered by their eigenvalues.

---

### 19. How do you decide how many PCA components to keep?

**⭐ Gold-standard answer (memorize):**

> We usually choose the number of components based on the cumulative explained variance ratio.
> If the first *k* components explain, for example, 90% or 95% of the total variance, we keep those *k* components.

**Senior add-on:**

> However, high explained variance does not guarantee better predictive performance, since PCA does not consider labels.

---

### 20. PCA vs feature selection?

**One-liner:**

> PCA transforms features, while feature selection keeps a subset of original features.

**Expanded:**

> PCA reduces dimensionality by projection, whereas feature selection preserves semantic interpretability.

---

# 🔹 Closing Meta-Answer (if interviewer challenges you)

> *“In practice, I treat PCA as a variance-preserving compression tool rather than a performance guarantee. I always validate downstream metrics after dimensionality reduction.”*

---

## ✅ 最重要的一句话（你当时那题）

你说 **“percentage problem”**
👉 **Conceptually 100% correct**
👉 标准英文表达就是：
**“cumulative explained variance ratio”**

---

### If you want next:

* **Mock interview (live pressure mode)**
* **Staff-level deep dives (why PCA can hurt classification)**
* **One-page printable cheat sheet**

Just tell me.

Great question — this is exactly where interviewers probe whether someone **knows PCA’s limits**, not just its definition.

I’ll structure this as:

1. **Why PCA struggles with nonlinear data**
2. **What PCA *still* does in nonlinear/correlated settings**
3. **Nonlinear dimensionality reduction methods (taxonomy)**
4. **When to use which method (engineering intuition)**
5. **Interview-ready follow-up Q&A**

---

## 1️⃣ Does PCA work for nonlinear (covarianced) data?

### Short answer

> **PCA only captures linear correlations.**
> It does **not** model nonlinear structure.

### Important distinction (many people confuse this):

| Case                    | PCA works? | Why                                |
| ----------------------- | ---------- | ---------------------------------- |
| **Correlated features** | ✅ Yes      | PCA decorrelates linear covariance |
| **Nonlinear manifold**  | ❌ No       | PCA uses linear projections only   |

So:

* **“Covarianced” ≠ “nonlinear”**
* PCA *handles covariance*, but **fails on curved manifolds**

---

## 2️⃣ What PCA actually does (mathematically)

PCA:

* Computes the **covariance matrix**
* Finds **orthogonal directions** maximizing variance
* Projects data onto a **linear subspace**

Geometrically:

* PCA fits a **flat plane** (or hyperplane)
* If the data lies on a **curve or surface**, PCA flattens it poorly

---

## 3️⃣ Why PCA fails on nonlinear structure (intuition)

### Example: Swiss Roll

![Image](https://www.researchgate.net/publication/337531501/figure/fig5/AS%3A829327780352000%401574738559009/llustration-of-PCA-and-MN-PCA-on-Swiss-roll-data-From-left-to-right-the-3-D-scatter.ppm)

![Image](https://sites.gatech.edu/omscs7641/files/2024/03/image-6651756662f11cb6-965x1024.png)

* Data lives on a **2D nonlinear manifold**
* PCA projects it into overlapping regions
* Distances and neighborhood structure are destroyed

💡 PCA minimizes **reconstruction error in Euclidean space**, not **geodesic distance**.

---

## 4️⃣ What *can* PCA still do on nonlinear data?

Even on nonlinear data, PCA can:

✅ Remove **linear redundancy**
✅ Reduce noise
✅ Improve numerical stability
❌ Recover intrinsic nonlinear geometry

This is why PCA is often used as:

> **a preprocessing step**, not the final solution

---

## 5️⃣ Nonlinear dimensionality reduction methods (taxonomy)

Here’s the clean mental model interviewers love:

---

### 🔹 1. Kernel Methods (Implicit Nonlinearity)

#### **Kernel PCA**

![Image](https://ml-explained.com/articles/kernel-pca-explained/kernel_pca.png)

![Image](https://ml-lectures.org/_images/circles_pca_kpca.png)

**Idea**:

* Map data into high-dimensional feature space
* Apply PCA there using the **kernel trick**

**Pros**

* Captures nonlinear variance
* Theoretically grounded

**Cons**

* Kernel choice sensitive
* (O(n^2)) memory
* Hard to scale

**Use when**

* Medium-sized datasets
* You want “PCA but nonlinear”

---

### 🔹 2. Manifold Learning (Geometry-Preserving)

#### Isomap / LLE / Laplacian Eigenmaps

![Image](https://media.geeksforgeeks.org/wp-content/uploads/20231209145753/Screenshot-%281391%29.jpg)

![Image](https://scikit-learn.org/stable/_images/sphx_glr_plot_manifold_sphere_001.png)

**Core idea**

> Preserve **local neighborhood geometry** instead of global variance

| Method    | Preserves                   |
| --------- | --------------------------- |
| Isomap    | Global geodesic distances   |
| LLE       | Local linear reconstruction |
| Laplacian | Graph smoothness            |

**Pros**

* Excellent for visualization
* Captures intrinsic structure

**Cons**

* Not scalable
* No inverse transform
* Poor out-of-sample handling

**Use when**

* 2D/3D visualization
* Scientific exploration

---

### 🔹 3. Probabilistic / Information-Theoretic

#### t-SNE / UMAP

![Image](https://www.researchgate.net/publication/327064001/figure/fig3/AS%3A660423665790976%401534468680654/t-SNE-plot-visualizing-cluster-assignments-of-cells-The-clustering-parameters-can-be.png)

![Image](https://www.researchgate.net/publication/368255524/figure/fig1/AS%3A11431281117824378%401675547686454/UMAP-visualization-of-splice-site-embeddings-Visualization-of-canonical-splice-sites-and.ppm)

**Idea**

* Preserve **neighborhood similarity distributions**

**Pros**

* Beautiful cluster separation
* Robust to nonlinear structure

**Cons**

* Distorts global distances
* Not stable across runs
* Not suitable for downstream modeling

**Use when**

* Embedding visualization
* Debugging representations

---

### 🔹 4. Neural Network Based (Learned Nonlinearity)

#### Autoencoders (AE / VAE)

![Image](https://miro.medium.com/v2/resize%3Afit%3A984/1%2AF8WIr3wLFwlRGpiytCPdlQ.png)

![Image](https://tiao.io/post/tutorial-on-variational-autoencoders-with-a-concise-keras-implementation/featured.png)

**Idea**

* Learn nonlinear encoder/decoder
* Optimize reconstruction loss

**Pros**

* Scales well
* Learns task-specific representations
* Handles complex manifolds

**Cons**

* Requires tuning
* Less interpretable
* Risk of overfitting

**Use when**

* Large datasets
* Production systems
* Multimodal inputs

---

## 6️⃣ Comparison table (interview gold)

| Method       | Linear | Preserves           | Scales | Production |
| ------------ | ------ | ------------------- | ------ | ---------- |
| PCA          | ✅      | Variance            | ✅      | ✅          |
| Kernel PCA   | ❌      | Variance            | ❌      | ⚠️         |
| Isomap / LLE | ❌      | Geometry            | ❌      | ❌          |
| t-SNE        | ❌      | Local similarity    | ❌      | ❌          |
| UMAP         | ❌      | Local + some global | ⚠️     | ⚠️         |
| Autoencoder  | ❌      | Reconstruction      | ✅      | ✅          |

---

## 7️⃣ Interview-ready answers (memorize)

### Q: Does PCA work for nonlinear data?

**Answer**

> PCA only captures linear correlations. It cannot recover nonlinear manifold structure because it relies on linear projections.

---

### Q: When would PCA still be useful for nonlinear data?

**Answer**

> As a preprocessing step to remove noise and linear redundancy before applying nonlinear models.

---

### Q: What would you use instead of PCA for nonlinear structure?

**Answer**

> Kernel PCA for moderate data sizes, manifold learning methods for visualization, and autoencoders for large-scale or production systems.

---

### Q: Why not always use t-SNE or UMAP?

**Answer**

> They distort global structure, are unstable across runs, and are not suitable for downstream predictive modeling.

---

## 8️⃣ Staff-level closing statement (very strong)

> *“I view PCA as a second-order statistics method. When data lies on a nonlinear manifold, I prefer representation learning approaches that optimize task-relevant objectives rather than variance alone.”*

---

If you want next, I can:

* Compare **Kernel PCA vs Autoencoder mathematically**
* Explain **why PCA is optimal under Gaussian assumptions**
* Show **how PCA + nonlinear models still works in practice**
* Turn this into a **2-minute spoken interview answer**

Just tell me which one.


--问的非常多且杂但是都不精，先从binary classifier是啥到举例，到你最喜欢哪个算法，我说logistic regression，于是开始问你介绍一下呀，我扯到了logistic function，具体他怎么问的我忘记了，只记得我一直在说指数函数、0、1、0.5边界值之类的。。。再后来他说怎么训练参数，就扯please dive deeper into related questions, with extended background concepts , methodologies, cons and pros, and follow up q&a: "到了MLE，cost funtion，gradient descent，他问梯度下降是什么呀，学习率什么含义啊，还有regularization，问regularization是啥，我因此扯到了防止overfitting，他借此又问overfitting是啥，怎么解决，我说完后，他借由这几种解决方法拓展到了cross validation和pca(feature selection)，于是我又扯扯扯到了pca的定义，他顺便问了一句pca怎么知道取几个component，这个问题我不确定，回答说这是个“pecentage problem"吧？如果想要80%或90%，就取到这个程度好了。。。英文表达真心捉急。。。也不知道他懂没懂，意会意会。。{:7_191:}有人知道这个问题怎么回答吗？求解"

对于机器学习工程师（MLE）的面试，这类从“基础概念”一路杀到“底层原理”的连环追问（Drill-down）非常典型。你的面试官是在测试你的**知识深度**以及**技术直觉**。

针对你提到的 PCA 维度选择问题，以及面试中涉及的逻辑回归、梯度下降等核心概念，我为你整理了一份“Staff 级别”的深度解析和面试通关指南。

---

### 1. PCA 到底取几个 Component？（深度回答）

你的“Percentage Problem”回答其实触及了核心，但在 Staff 级别的面试中，面试官希望听到更系统、多维度的决策方案。

* **方法一：累计解释方差贡献率 (Cumulative Explained Variance Ratio)**
* **核心：** 正如你所说，画出累计贡献率曲线（Scree Plot），选取能解释 80%~95% 方差的  值。
* **进阶：** 提到“碎石图的拐点（Elbow Point）”，即方差贡献开始急剧衰减的那个点，它是信息增益与模型复杂度之间的平衡点。


* **方法二：凯撒准则 (Kaiser Criterion)**
* **核心：** 只保留特征值（Eigenvalue）大于 1 的主成分。这意味着该主成分解释的信息量至少要比一个原始特征多。


* **方法三：交叉验证 (Cross-Validation)**
* **核心：** 这是最硬核的工程回答。将  作为一个超参数，看在下游任务（如分类或回归）中，取多少个  能让验证集指标（F1, AUC, MSE）达到最优。


* **方法四：重构误差 (Reconstruction Error)**
* **核心：** 计算原始数据与降维后还原数据的差异。如果业务对数据保真度要求高，需控制重构误差在一定范围内。



---

### 2. 逻辑回归 (Logistic Regression) 深度连环炮

#### A. MLE (极大似然估计) 与 Cost Function

* **面试官问：** 为什么逻辑回归不用 MSE（均方误差）？
* **深度回答：** 1.  **非凸性 (Non-convexity)：** 如果将 Sigmoid 带入 MSE，代价函数会变成非凸的，有很多局部最小值，梯度下降很难找到全局最优。
2.  **梯度消失：** MSE 的导数包含 ，当预测值接近 0 或 1 时，梯度极小，导致模型收敛极慢。
3.  **统计学意义：** 逻辑回归假设  服从伯努利分布，使用 **Cross-Entropy（交叉熵）** 本质上就是最大化似然函数。

#### B. 梯度下降 (Gradient Descent) 与 学习率

* **扩展概念：** * **学习率的作用：** 它决定了步长。太大会导致震荡不收敛，太小会导致收敛太慢甚至陷入局部最优。
* **Staff 视角：** 提到“学习率调度 (Learning Rate Scheduling)”，如衰减、Warmup 或使用 Adam 等自适应优化器。



#### C. 正则化 (Regularization) 与 Overfitting

* **L1 (Lasso) vs L2 (Ridge)：**
* **L1 导致稀疏性：** L1 的等值线是“棱形”，极值点容易落在轴上，使得某些权重  变为 0，从而实现**自动特征选择**。
* **L2 倾向平滑：** L2 的等值线是“圆形”，它会让  整体变小但不会变 0，能有效应对多重共线性（Multicollinearity）。


* **Overfitting 的其他解法：** 除了正则化，还要提到：
* **数据层面：** 增加样本量、数据增强（Data Augmentation）。
* **模型层面：** 降低模型复杂度、Dropout（神经网络）、Early Stopping（早停）。



---

### 3. 给你的面试建议（中英夹杂没关系）

面试官并不期待你像教科书一样背诵，他更在意你的**工程化思考**。

> **当英语表达吃力时，可以尝试这个套路：**
> 1. **给出直觉 (Intuition)：** "To prevent the model from memorizing noise..." (为了防止模型死记硬背噪声...)
> 2. **给出结论 (Trade-off)：** "It's a trade-off between bias and variance." (这是偏差和方差的权衡。)
> 3. **给出工具 (Tools)：** "We use tools like L1/L2 penalty or Cross-Validation."
> 
> 

### 总结 Q&A 练习：

* **Q: Why use PCA?**
* **A:** To reduce dimension while preserving maximum variance, helping to speed up training and remove noise/redundancy.
* **Q: How to handle Imbalanced Data in Logistic Regression?**
* **A:** (Staff 必考题) Resampling (SMOTE), class weight adjustment in loss function, or changing evaluation metric from Accuracy to PR-AUC.

---

### 视频推荐

[PCA 核心原理与维度选择决策](https://www.youtube.com/watch?v=FgakZw6K1QQ)
该视频详细讲解了如何通过 Scree Plot 和累计方差比例来确定主成分的数量，能够直观地弥补面试中提到的“如何知道取几个 component”的逻辑盲区。

**下次面试你需要我陪你针对某个特定公司（比如 LinkedIn 或 Meta）进行一次模拟面试（Mock Interview）吗？**

Calculating the **Cumulative Explained Variance Ratio** is a standard Staff-level practice used to determine how many Principal Components are necessary to represent your data accurately without over-complicating the model.

In Principal Component Analysis (PCA), each component captures a specific percentage of the total variance. The cumulative ratio is simply the running total of these percentages.

### 1. The Mathematical Foundation

The variance explained by each principal component corresponds to its **eigenvalue** (). If you have  features, the explained variance ratio for the -th component is:

The **Cumulative Explained Variance Ratio** for  components is:

---

### 2. Implementation in Python (Scikit-Learn)

Using `sklearn`, you can compute this easily with `np.cumsum` on the `explained_variance_ratio_` attribute.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1. Standardize your data (Crucial for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Fit PCA without reducing dimensions first
pca = PCA()
pca.fit(X_scaled)

# 3. Calculate Cumulative Variance
exp_var_ratio = pca.explained_variance_ratio_
cum_exp_var = np.cumsum(exp_var_ratio)

# 4. Find the 'K' components for 95% variance
k = np.argmax(cum_exp_var >= 0.95) + 1
print(f"Number of components to explain 95% variance: {k}")

```

### 3. Visualizing with a Scree Plot

A "Scree Plot" helps you identify the **"Elbow Point,"** where adding more components provides diminishing returns.

```python
plt.figure(figsize=(8, 5))
plt.step(range(1, len(cum_exp_var) + 1), cum_exp_var, where='mid', label='Cumulative')
plt.bar(range(1, len(exp_var_ratio) + 1), exp_var_ratio, alpha=0.5, label='Individual')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% Threshold')
plt.xlabel('Principal Component Index')
plt.ylabel('Explained Variance Ratio')
plt.legend(loc='best')
plt.show()

```

---

### 4. Staff-Level Strategy: How many to keep?

In a technical interview, avoid giving a single number. Instead, discuss the **Trade-off Strategy**:

* **The 80/90 Rule:** A common heuristic is to retain enough components to explain **80% to 90%** of the variance.
* **The Elbow Method:** Look for the point in the plot where the curve flattens out. This point represents a balance between compression and information retention.
* **Downstream Performance:** If PCA is a preprocessing step for a classifier (e.g., Logistic Regression), treat  (number of components) as a **hyperparameter** and optimize it using cross-validation.
* **Noise Filtering:** Sometimes, the last 5-10% of variance is just random noise. Discarding these components can actually improve model generalization (reducing variance).

[Check out this breakdown of Cumulative Explained Variance](https://www.youtube.com/watch?v=NnZ43wnjsGE) to see how to visually identify the optimal number of components for high-dimensional datasets.

This video provides a practical look at how to use these measures to balance simplicity and accuracy in your feature extraction workflows.

 -- please explain :"Cumulative Explained Variance Ratio "
 **Cumulative Explained Variance Ratio (CEVR)** is a concept most commonly used in **dimensionality reduction**, especially **PCA (Principal Component Analysis)**, to quantify **how much of the total variance in the data is captured as you add more components**.

---

## 1. Intuition (Plain English)

* Every dataset has **total variance** (overall information / spread).
* PCA decomposes this variance into **orthogonal components** (PC1, PC2, PC3, …).
* Each component explains a **fraction** of the total variance.
* **Cumulative explained variance ratio** answers:

> *“If I keep the first **k** components, how much of the original information have I preserved?”*

---

## 2. Formal Definition

Let:

* ( \lambda_i ) = eigenvalue of the (i)-th principal component
* ( \sum_{j=1}^{d} \lambda_j ) = total variance

### Explained Variance Ratio (per component)

[
\text{EVR}*i = \frac{\lambda_i}{\sum*{j=1}^{d} \lambda_j}
]

### **Cumulative Explained Variance Ratio**

[
\text{CEVR}*k = \sum*{i=1}^{k} \text{EVR}_i
]

This value is **monotonic increasing** and bounded by **1.0 (100%)**.

---

## 3. Concrete Example

Assume PCA produces:

| Component | Explained Variance Ratio |
| --------- | ------------------------ |
| PC1       | 0.50                     |
| PC2       | 0.30                     |
| PC3       | 0.10                     |
| PC4       | 0.05                     |
| PC5       | 0.05                     |

### Cumulative Version

| # Components | Cumulative Explained Variance |
| ------------ | ----------------------------- |
| 1            | 0.50                          |
| 2            | 0.80                          |
| 3            | 0.90                          |
| 4            | 0.95                          |
| 5            | 1.00                          |

**Interpretation**:

* Keeping **2 components → 80%** of information
* Keeping **3 components → 90%**
* Keeping **4 components → 95%**

---

## 4. Why It Matters (Decision-Making Tool)

### 1️⃣ Dimensionality Reduction

Choose the smallest (k) such that:

* ( \text{CEVR}_k \ge 90% ) or (95% )

This balances:

* **Information preservation**
* **Model simplicity**
* **Lower overfitting risk**
* **Faster training/inference**

### 2️⃣ Noise Reduction

Later components often capture:

* Measurement noise
* Redundant correlations
  Dropping them improves robustness.

### 3️⃣ Visualization

* 2D or 3D PCA plots are meaningful **only if** CEVR is reasonably high.

---

## 5. In Practice (scikit-learn)

```python
from sklearn.decomposition import PCA
import numpy as np

pca = PCA()
pca.fit(X)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)
```

Typical usage:

```python
pca = PCA(n_components=0.95)  # keep 95% variance automatically
X_reduced = pca.fit_transform(X)
```

---

## 6. Scree Plot vs Cumulative Plot

![Image](https://substackcdn.com/image/fetch/%24s_%215PI4%21%2Cf_auto%2Cq_auto%3Agood%2Cfl_progressive%3Asteep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F745d9e31-95b9-4167-923c-949f8d96a15b_5104x4523.png)

![Image](https://statisticsglobe.com/wp-content/uploads/2022/12/screeplot_pca_mtcars.png)

* **Scree plot**: per-component variance
* **Cumulative plot**: running total (used to select (k))

---

## 7. Common Pitfalls

⚠️ **High CEVR ≠ Better Task Performance**

* PCA is **unsupervised**
* It preserves variance, **not label predictiveness**

⚠️ **Feature Scaling Matters**

* Always standardize before PCA:

  * Otherwise, high-magnitude features dominate variance

⚠️ **Correlated Features Inflate Early PCs**

* Expected behavior, not a bug

---

## 8. Mental Model (Interview-Ready)

> *“Cumulative explained variance ratio measures how much total data variance is preserved as we retain more principal components. It
> It’s used to select the minimum number of dimensions that capture a target percentage (e.g., 95%) of the original information, trading off compression and fidelity.”*

---

If you want, I can:

* Walk through **eigenvalues → variance** step-by-step
* Explain **why PCA maximizes variance mathematically**
* Connect this to **autoencoders / latent dimension selection**
* Show how it affects **downstream ML model performance**

Just tell me which angle you want.

-- formular for activators, sigmoid, softmax, tanh, relu, leakage relu, and other relus and other activators, which is good for what cases


In deep learning, activation functions introduce non-linearity, allowing neural networks to learn complex patterns. Choosing the right one depends on the layer type, the network depth, and the specific problem (e.g., vanishing gradients).

---

### 1. Classical Activations (Squashing Functions)

#### **Sigmoid**

* **Formula:** 
* **Range:** 
* **Best Use Case:** Output layer of **Binary Classification**.
* **Cons:** Saturated neurons kill gradients (Vanishing Gradient problem); output is not zero-centered.

#### **Tanh (Hyperbolic Tangent)**

* **Formula:** 
* **Range:** 
* **Best Use Case:** Often used in **RNNs** or hidden layers where zero-centered data is preferred to speed up convergence.
* **Cons:** Still suffers from vanishing gradients at extreme values.

---

### 2. ReLU Family (The Modern Standards)

#### **ReLU (Rectified Linear Unit)**

* **Formula:** 
* **Best Use Case:** The **default choice for hidden layers** in CNNs and MLP.
* **Pros:** Computationally efficient; reduces vanishing gradient.
* **Cons:** "Dying ReLU" problem (neurons can get stuck at 0 and never update).

#### **Leaky ReLU**

* **Formula:** , where  is a small constant (e.g., ).
* **Best Use Case:** When you encounter **Dying ReLU** problems.
* **Pros:** Ensures that gradients are never exactly zero for negative values.

#### **Parametric ReLU (PReLU)**

* **Formula:** , where  is a **learnable parameter**.
* **Best Use Case:** Large-scale vision tasks where the model can benefit from learning the negative slope.

#### **ELU (Exponential Linear Unit)**

* **Formula:**  if , else 
* **Best Use Case:** Deep networks where you want to speed up learning and avoid the bias shift of ReLU.
* **Pros:** Smoother transition at zero; zero-centered outputs.

---

### 3. Advanced & Gated Activations

#### **Softmax**

* **Formula:** 
* **Best Use Case:** **Output layer for Multi-class Classification**.
* **Pro:** Turns logits into a probability distribution that sums to 1.

#### **Swish (SiLU)**

* **Formula:** 
* **Best Use Case:** State-of-the-art CNNs (like **EfficientNet**).
* **Pro:** It is non-monotonic and smoother than ReLU, which helps in deeper networks.

#### **GELU (Gaussian Error Linear Unit)**

* **Formula:** 
* **Best Use Case:** **Transformers (BERT, GPT, etc.)**.
* **Pro:** It weighs inputs by their percentile, providing a stochastic feel to the activation.

---

### 4. Selection Strategy Summary

| Layer Type | Activation to Use | Why? |
| --- | --- | --- |
| **Hidden Layers (Default)** | **ReLU** | Fast, simple, reliable. |
| **Hidden Layers (Deep/SOTA)** | **GELU / Swish** | Better gradient flow in Transformers/EfficientNets. |
| **Output (Binary Class)** | **Sigmoid** | Maps output to  probability. |
| **Output (Multi-class)** | **Softmax** | Maps output to probability distribution. |
| **Regression (Any Value)** | **Linear** | Returns the raw logit . |
| **Recurrent Layers (RNN)** | **Tanh** | Keeps values bounded between -1 and 1. |

### Staff MLE Interview Question

**Q: Why does GELU perform better than ReLU in Transformers?**
**A:** "ReLU is a hard gate (0 or 1). GELU acts as a **stochastic regularizer**; it scales the input by the probability that the input is dropped based on a Gaussian distribution. This smoothness allows the model to capture more complex non-linear relationships in high-dimensional attention spaces."

**Would you like me to demonstrate how to implement a custom "Gated Linear Unit" (GLU) in PyTorch, which is common in modern LLM architectures?**


# https://www.1point3acres.com/interview/thread/1154740

1. AI leader host. 这一轮主要包含3部分：1) 介绍自己的工作经历。2）问一些behavioral的问题，比如你工作中遇到的最大的困难是什么，你怎么克服的？你最proud的project是什么，你有什么贡献？3）回答一个关于recommendation system high latency issue的问题，你怎么解决？从infra和ML model的角度有什么措施可以解决。这一轮总体还不错，给的feedback也是通过了。

2. ML coding. 这一轮的coding题是关于sparse vector and matrix的计算，跟刷题网的幺屋企灵和伞遥遥类似。这个实现起来比刷题网那两题要难一些，因为面试官要求你从0开始写class，设计constructor, function, attributes, variables等，更重要的是，要实现sparse的特性，time complexity不能是O(M*N）(M,N分别是矩阵的行数和列数)， 在memory上要最优化。我把sparse vector的mulitplication实现了，在实现sparse matrix product时，一开始用的是O(M*N）的方法，面试官觉得这样不行，得memory是最优化，后来没改完就到时间了。所以这一轮挂了。

3. AI System Design。这一轮不是常见的推荐系统design，面试官给定一个情形：根据linkedin网页上的user信息，找到每个user对应的job category和相应category里面的attributes。比如developer是其中一个job category, 这个category下面有很多种attributes (e.g., frontend engineer, backend engineer, infra engineer等等)。首先你得构建这么一个job category和对应attributes的table，然后根据每个user的相关信息找到对应的category and corresponding attributes。这一轮我其实觉得交流的还不错，最后把我挂了也是有点小意外。后来从recruiter哪里得到的feedback显示，给我red flag的问题，我之前觉得都不是大问题，有时是因为沟通没那么顺畅，结果就被red flag了。比如说，我说到这是一个multi-class的classification的问题，面试官问那可以用logistic regression (LR)吗？我说LR一般是用来做binary classification, 面试官又follow up问那有没有办法用LR来做multi-class，我说可以，一个class用一个LR，然后combine结果就可以，就是one vs rest的问题。最后的feedback中，这就是一个red flag，其他2个red flag也非常类似。只能说，没有一下子回答到面试官心坎里，需要hint才能回答对，这就没有达到他们要求的bar，这都是red flag。

4. Data coding. 2道刷题网原题，幺儿期，午留。除了coding，面试官问了些底层的逻辑，比如，第一题中为什么要用deque而不是用list来存储queue。总体进行的比较顺利，这一轮通过了。

5. AI modeling and fundamentals。我之前在地里看到的面经多数是说ML八股文，所以准备了一堆八股文，结果面试内容跟八股文完全不沾边。面试官一上来就问怎么根据一堆sample的数据点来推算distribution的parameters，我就说可以用maximum likelihood estimation (MLE)，然后他就让我在coderpad上写下MLE的数学公式，然后让我一步步推算怎么求normal distribution的mu and sigma (需要求likelihood function对mu和sigma的导数分别为0时候的值)，然后我就使劲回想多年前学习的高数知识，算是勉强把这一步做完了。接下来就问对于gaussian mixture model怎么求不同distribution的参数，为什么不能用MLE(因为复杂方程可能不能直接求解导数为0的情形)，我说得用EM (expectation maximization), 然后大概讲了下EM的原理，但是里面有些细节记不起来了。所以这一轮跪了！
6. 

This interview is a classic "First Principles" deep dive. At the Staff level, LinkedIn doesn't just want you to know the name of an algorithm; they want you to prove you understand the **probabilistic engine** under the hood.

To prepare for this in English, you need to master the mathematical narrative. Here is a breakdown of the concepts you missed, the solutions, and the English vocabulary needed to explain them.

---

### 1. The Core Task: Parameter Estimation

When an interviewer asks how to estimate parameters from samples, they are looking for **Frequentist** (MLE) vs. **Bayesian** (MAP) perspectives.

#### Maximum Likelihood Estimation (MLE)

**Concept:** Finding the parameters  that maximize the probability of observing the given data.
**The "English" Narrative:** "We assume the data points are i.i.d. (independent and identically distributed). We construct a Likelihood Function , take the log to simplify the math (log-likelihood), and find the stationary point by setting the partial derivatives to zero."

**Solution: MLE for Normal Distribution **

1. **Likelihood:** 
2. **Log-Likelihood:** 
3. **Derivative w.r.t :**  (Sample Mean).
4. **Derivative w.r.t :** .

---

### 2. The Gaussian Mixture Model (GMM) Trap

The interviewer asked why MLE fails here. You were right about "complexity," but the Staff-level answer involves **Latent Variables**.

**Why MLE fails for GMM:**

* **Latent Variables:** We don't know which Gaussian "generated" which data point.
* **Non-convexity:** The log-likelihood of a sum (mixture) is , which doesn't simplify nicely. You cannot decouple the parameters.
* **Singularities:** If a Gaussian centers perfectly on one data point and its variance goes to zero, the likelihood goes to infinity (overfitting).

---

### 3. Solution: The EM Algorithm (The "Details")

If you can't remember the exact math, explain the **Two-Step Logic** clearly in English.

* **Objective:** Maximize the likelihood when data is "incomplete" (missing cluster assignments).
* **E-Step (Expectation):** "Calculate the **responsibilities**." For each data point, use current parameters to estimate the probability that it belongs to each cluster .
* **M-Step (Maximization):** "Update the parameters." Use the responsibilities as weights to re-calculate the mean, variance, and mixing coefficients () of each cluster.

---

### 4. Advanced Follow-ups: Sampling & Distributions

Expect these "Staff" level statistics questions:

| Question | Key English Vocabulary |
| --- | --- |
| **How do you handle outliers in MLE?** | "MLE is sensitive to outliers. I would use a **t-distribution** or **Huber loss** because they have 'heavier tails' than a Gaussian." |
| **What is the Central Limit Theorem (CLT)?** | "As the sample size  increases, the **sampling distribution of the mean** approaches a normal distribution, regardless of the population's original distribution." |
| **Bias vs. Unbiased Estimators?** | "The MLE for  is **biased** (it divides by ). We use **Bessel’s Correction** (dividing by ) to make it an unbiased estimator." |

---

### 5. Mock Interview Script (Practice This)

**Interviewer:** "Explain the intuition behind EM for GMM."
**You:** "Since we have **latent cluster assignments**, we can't solve for parameters directly using a closed-form MLE. Instead, we use an iterative approach. In the **E-step**, we perform 'soft clustering' by assigning a weight to each point for each Gaussian. In the **M-step**, we treat these weights as fixed and update the Gaussians to better fit their assigned points. We repeat this until the **log-likelihood converges**."

**Next Step:**
Would you like to practice a **CoderPad session** where you write out the derivation for a **Bernoulli distribution** (relevant for Click-Through Rate) or a **Poisson distribution** (relevant for event counts)? This is the "high math" version of ML fundamentals.

This round is **not ML 八股文**.
It is a **probability + statistics + estimation theory + optimization** round, delivered **in English, on a whiteboard / coderpad**.

Below is a **targeted, interview-ready preparation guide**, optimized for exactly what you described.
I’ll give you:

1. **Mental model of what they are testing**
2. **Canonical English answer flow (step-by-step)**
3. **Must-memorize derivations (Normal MLE)**
4. **Sampling distribution & asymptotics (hidden follow-ups)**
5. **GMM + EM (what to say, what to derive, what NOT to derive)**
6. **High-probability follow-up Q&A (English scripts)**

You can literally **practice reading this out loud**.

---

# 0️⃣ What this round is REALLY testing

They are not testing:

* sklearn usage
* model zoo knowledge
* ML system design

They ARE testing:

* **Statistical estimation from data**
* **Likelihood thinking**
* **Derivation discipline**
* **Sampling distribution intuition**
* **Ability to reason from first principles in English**

> This is a **classical applied statistics interview**, disguised as “AI modeling”.

---

# 1️⃣ How to answer in English: the correct opening posture

When asked:

> “Given samples, how do you estimate distribution parameters?”

### ✅ Ideal opening answer (memorize)

> *“I would first assume a parametric form for the data distribution, then estimate its parameters using Maximum Likelihood Estimation.”*

This immediately signals:

* You understand **model assumptions**
* You’re not guessing formulas

---

# 2️⃣ Normal Distribution MLE — MUST MEMORIZE DERIVATION

You **must** be able to do this **without hesitation**.

---

## Step 1: State assumptions clearly (English matters)

> *“Assume the data points are i.i.d. samples from a normal distribution with mean μ and variance σ².”*

[
x_1, x_2, \dots, x_n \sim \mathcal{N}(\mu, \sigma^2)
]

---

## Step 2: Write the likelihood function

[
L(\mu, \sigma) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left(-\frac{(x_i - \mu)^2}{2\sigma^2}\right)
]

### Say out loud:

> *“The likelihood is the joint probability of observing the data given the parameters.”*

---

## Step 3: Take log-likelihood (critical move)

> *“For easier optimization, I take the log-likelihood.”*

[
\ell(\mu, \sigma) =
-\frac{n}{2}\log(2\pi)

* n\log\sigma
* \frac{1}{2\sigma^2}\sum_{i=1}^n (x_i - \mu)^2
  ]

---

## Step 4: Derivative w.r.t. μ

[
\frac{\partial \ell}{\partial \mu}
= \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu)
]

Set to zero:

[
\sum (x_i - \mu) = 0
\Rightarrow \mu = \frac{1}{n}\sum x_i
]

### English explanation:

> *“The MLE estimate of μ is simply the sample mean.”*

---

## Step 5: Derivative w.r.t. σ² (important detail)

[
\frac{\partial \ell}{\partial \sigma^2}
= -\frac{n}{2\sigma^2}

* \frac{1}{2\sigma^4}\sum (x_i - \mu)^2
  ]

Solve:

[
\sigma^2 = \frac{1}{n}\sum (x_i - \mu)^2
]

⚠️ **Notice**:

* This is **not** the unbiased estimator
* MLE divides by **n**, not **n−1**

### Say this explicitly:

> *“The MLE variance estimator is biased but consistent.”*

That sentence is **huge signal**.

---

# 3️⃣ Sampling Distribution (they often pivot here)

Once you derive estimators, the *next hidden layer* is:

> “What is the distribution of your estimator?”

---

## Sample Mean Distribution

If:
[
x_i \sim \mathcal{N}(\mu, \sigma^2)
]

Then:
[
\bar{x} \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)
]

### Interview English:

> *“The sampling distribution of the sample mean is normal, with variance shrinking at rate 1/n.”*

---

## If data is NOT normal?

> *“By the Central Limit Theorem, the sample mean converges in distribution to a normal distribution as n increases.”*

This is **mandatory CLT usage**.

---

## Variance estimator distribution (advanced but impressive)

[
\frac{(n-1)s^2}{\sigma^2} \sim \chi^2_{n-1}
]

Say only if confident.

---

# 4️⃣ Why MLE fails for GMM (this is where you slipped)

You said the **right idea**, but you need **crisper phrasing**.

---

## GMM likelihood

[
p(x) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k,\sigma_k^2)
]

### Why direct MLE fails

> *“The log-likelihood contains a log of a sum, which leads to coupled nonlinear equations without closed-form solutions.”*

🚫 Do **NOT** say “cannot take derivative”
✅ Say “no closed-form stationary point”

---

# 5️⃣ EM Algorithm — this is how to explain it cleanly

---

## Step 1: Latent variable introduction (key idea)

> *“We introduce a latent variable z indicating which component generated each sample.”*

---

## Step 2: E-Step (Expectation)

[
\gamma_{ik} = P(z_i = k | x_i, \theta^{old})
]

English:

> *“In the E-step, we compute soft assignments using Bayes’ rule.”*

---

## Step 3: M-Step (Maximization)

Update:

* (\mu_k)
* (\sigma_k^2)
* (\pi_k)

Using **weighted MLE**.

English:

> *“In the M-step, we maximize the expected complete-data log-likelihood.”*

---

## Step 4: Convergence property (IMPORTANT)

> *“EM guarantees non-decreasing likelihood, but may converge to local optima.”*

---

# 6️⃣ Sampling + EM follow-up questions (very common)

### Q: Is EM guaranteed to find global optimum?

> No, it is sensitive to initialization and may converge to local maxima.

---

### Q: Why is EM better than gradient descent here?

> EM exploits conditional independence and yields closed-form updates in each step.

---

### Q: What happens with small sample size?

> Parameter estimates have high variance and components may collapse.

---

# 7️⃣ What to practice BEFORE next interview (very concrete)

You should be able to **derive on whiteboard, in English**:

✅ Normal MLE (μ, σ²)
✅ Sampling distribution of mean
✅ CLT statement
✅ Why MLE variance is biased
✅ Why GMM MLE has no closed form
✅ EM steps without formulas panic

---

# 8️⃣ One-paragraph “redemption-level” answer (memorize)

> *“This round focuses more on statistical estimation than model selection. Given samples, I start from distributional assumptions, derive likelihoods, and obtain estimators using MLE. For simple distributions like Gaussian, closed-form solutions exist. For latent variable models like GMMs, direct maximization is intractable, so we use EM to iteratively estimate latent assignments and parameters. I also pay attention to the sampling distribution of estimators to understand variance and convergence behavior.”*

---

# 9️⃣ Want next?

I can:
1️⃣ Run a **mock version of this exact round (English only)**
2️⃣ Give you a **1-page derivation cheat sheet**
3️⃣ Drill **probability questions interviewers love to ambush with**
4️⃣ Upgrade this to **Staff / Principal bar**

Just tell me **which number**.

For a Staff-level MLE interview, these questions transition from basic definitions to system design and hardware-aware optimization. Here is the translated and deep-dive analysis of your technical prompts.

---

### 1. Integer Quantization (Int8) and Overflow Prevention

**English Question:** How do you prevent numerical overflow when using Int8 quantization?

**Deep Dive:**
Int8 quantization maps Float32 weights to the range . The risk of overflow occurs during the **Multiply-Accumulate (MAC)** operation: .

* **Accumulator Precision:** Standard hardware (like NVIDIA Tensor Cores or ARM NEON) uses **Int32 accumulators**. Since  results in a maximum value of  (), you can safely sum up to  products before risking an Int32 overflow.
* **Quantization Schemes:** * **Symmetric:** Scales values to . Simple but wastes one bit for asymmetric distributions (like ReLU outputs).
* **Asymmetric:** Uses a "Zero-point" to map the min/max of the float range to  and .


* **Saturation:** If the final result after scaling back to Int8 exceeds 127, we "clamp" or saturate the value at 127 rather than letting it wrap around to -128.

---

### 2. ALBERT: Improvements over BERT

**English Question:** What are the advantages and improvements of ALBERT compared to BERT?

**Deep Dive:**
ALBERT (A Lite BERT) addresses the memory bottleneck of scaling BERT by introducing two key parameter-reduction techniques:

1. **Factorized Embedding Parameterization:** Instead of a huge embedding matrix  (where  is the hidden size), ALBERT decomposes it into  and  (where ). This decouples vocabulary size from hidden size.
2. **Cross-layer Parameter Sharing:** All Transformer layers share the same weights. This significantly reduces the total parameter count while maintaining depth, acting as a form of regularization.
3. **SOP (Sentence Order Prediction):** ALBERT replaces BERT's NSP (Next Sentence Prediction) task, which was deemed too easy, with SOP (detecting if two segments were swapped), forcing the model to learn finer coherence.

---

### 3. BERT Acceleration & Distillation Models

**English Question:** Which models optimize the BERT structure to achieve acceleration? How is BERT distillation performed?

**Deep Dive:**
To deploy BERT at scale (e.g., for LinkedIn search), we use "Student" models.

* **DistilBERT:** Focuses on the output layer (Softmax) and intermediate hidden states using a triple loss: distillation loss, masked language modeling loss, and cosine distance loss.
* **TinyBERT:** Performs distillation at every level: **Embedding layer**, **Transformer layer** (Attention matrices and Hidden states), and **Prediction layer**.
* **MiniLM:** Only distills the **Self-Attention Relation** (the KL-divergence of attention distributions) of the last layer, making it very flexible regarding the student's architecture.

---

### 4. Model Ensemble: Bagging vs. Boosting

**English Question:** What are the methods for model ensemble (e.g., combining BERT with non-BERT models), and what are their pros/cons?

**Deep Dive:**
In a production "Duplicate Detection" (De-duplication) system, you might combine a fast BM25/TF-IDF model (Wide) with a BERT model (Deep).

* **Bagging (Bootstrap Aggregating):** Reduces **Variance**. Models are trained independently in parallel. (e.g., Random Forest). Best for high-variance models.
* **Boosting:** Reduces **Bias**. Models are trained sequentially, with each new model focusing on the errors of the previous one (e.g., XGBoost, LightGBM).
* **Stacking (Blending):** A meta-model is trained to combine the predictions of base models. This is common for combining a BERT score with "hand-crafted" features (like text length, overlap ratio, etc.).

---

### 5. Time Series & Sequence Models

**English Question:** What are the common time-series models for CTR prediction?

**Deep Dive:**

* **DIN (Deep Interest Network):** Uses an **Attention mechanism** over a user's historical behavior sequence to calculate the relevance of past behaviors to the current candidate ad.
* **DIEN (Deep Interest Evolution Network):** Uses **GRU with Attentional Update Gate (AUGRU)** to model the evolution of user interests over time.
* **DeepFM:** While not strictly a sequence model, it handles categorical time-series data via FM layers (low-order) and Deep layers (high-order).

---

### 6. System Design: Balancing Business Value & UX

**English Question:** In a Search Ads scenario, how do you balance commercial value (Revenue) and User Experience (UX)?

**Staff-level Strategy:**
You treat this as a **Multi-Objective Optimization (MOO)** problem.

1. **Objective Function:** .
2. **Constraints:** Set a "Relevance Floor." If the semantic similarity (via BERT) is below a threshold (e.g., 0.7), the ad is disqualified regardless of the bid.
3. **Negative Feedback:** Incorporate user signals (skips, "not interested") as negative rewards in the reinforcement learning loop.
4. **Ad Load Control:** Dynamically adjust the number of ads shown based on user session length and engagement to prevent "Ad Fatigue."

---

### 7. Optimizing Relevance in Ads

**English Question:** How do you optimize "Relevance" in an advertising search scenario?

**Deep Dive:**

* **Query Expansion:** Using LLMs to expand short queries into semantic synonyms to increase recall.
* **Two-Tower Models:** Training a Query-Tower and an Ad-Tower to map both into a unified embedding space for fast vector search (Milvus/HNSW).
* **Cross-Encoders:** For the final re-ranking stage, feeding (Query, Ad) pairs into a BERT-style model to capture fine-grained interaction.

---

### Staff-Level Mock Q&A

**Q: "Why does ALBERT's parameter sharing improve memory but not necessarily inference speed?"**
**A:** "Because parameter sharing only reduces the number of *unique* parameters stored in memory. During the forward pass, the model still has to perform the same number of FLOPs (Floating Point Operations) across all layers. To improve speed, we would need **Layer Pruning** or **Quantization**."

**Q: "If you are ensemble-ing a BERT model and a Logistic Regression model for production, how do you handle the difference in score distributions?"**
**A:** "I would use **Probability Calibration** (Platt Scaling or Isotonic Regression). Since LR is naturally well-calibrated via log-loss, but BERT can be overconfident, we must map both to the same 'Real-world Probability' space before weighted averaging."

**Would you like to practice the English explanation for the "Knowledge Distillation Loss" formula used in DistilBERT?**

### 2. When using Int8 quantization, how to prevent numerical overflow?

**Translated Reference Answer from AI Interview Guide:**  
1) When computing convolutions, use Int32 as the intermediate value. Because after multiplication and addition with Int8, the result won't exceed the Int32 range.  
2) Through linear mapping methods, i.e., in layers with bias, first convert Int8 to Float32 to compute the result, then convert back to Int8.

**Deeper Dive with Extended Knowledge:**  
Int8 quantization reduces model size and inference speed by representing weights/activations in 8-bit integers (range: -128 to 127 for signed), but risks overflow during operations like matrix multiplications where partial sums can exceed this range (e.g., 128 * 128 = 16384, far beyond Int8). This leads to accuracy loss or NaNs. Beyond the reference, advanced techniques include:  
- **Per-Tensor vs. Per-Channel Scaling**: In TensorRT or ONNX, use per-channel scales to normalize different feature maps, reducing overflow in heterogeneous distributions.  
- **Quantization-Aware Training (QAT)**: Retrain with fake-quant nodes (e.g., in TensorFlow/PyTorch) to simulate Int8 during forward/backward passes, adjusting for overflow via clip gradients or dynamic ranges.  
- **Overflow Detection Tools**: Use libraries like TensorFlow's Quantization Debugger or PyTorch's torch.quantization to profile and insert saturation ops (e.g., clamp to Int32 mid-computation).  
- **Hybrid Precision**: In modern accelerators (e.g., NVIDIA's Ampere with TF32), mix Int8 with higher precision for accumulators. Post-2023 advancements include INT4/INT8 hybrids in LLMs like GPTQ, where overflow is mitigated by outlier handling (storing outliers in FP16).  
Empirically, in CNNs like ResNet, Int8 can drop accuracy by <1% with proper handling, but in transformers, attention scores are prone to overflow due to softmax scaling.

**Follow-up Q&A:**  
**Q: What are the trade-offs between symmetric and asymmetric quantization in preventing overflow?**  
A: Symmetric (zero-centered, e.g., -127 to 127) simplifies hardware but risks overflow in positive-biased activations. Asymmetric (with zero-point offset) better fits real distributions (e.g., ReLU outputs 0+), reducing clipping but adds computational overhead for zero-point adjustments.  

**Q: How does post-training quantization (PTQ) differ from QAT in overflow prevention?**  
A: PTQ calibrates on a dataset post-training, using histograms to set ranges and prevent overflow via static clips, but it's less accurate (2-5% drop). QAT fine-tunes with quantization in the loop, dynamically learning to avoid overflow, often recovering full accuracy but requiring more compute.

### 3. What are the advantages and improvements of ALBERT compared to BERT?

**Translated Reference Answer from AI Interview Guide:**  
ALBERT is a lightweight BERT. ALBERT uses a factorization method to significantly reduce BERT's model parameters, solving the problem of too many parameters exceeding memory, which prevents deepening or widening the network.

**Deeper Dive with Extended Knowledge:**  
ALBERT (A Lite BERT, 2019) addresses BERT's parameter explosion (110M for base) by:  
- **Factorized Embedding Parameterization**: Decouples vocab embedding size (E) from hidden size (H), reducing params from O(V*H) to O(V*E + E*H) where E<<H. This cuts embedding params by 80-90%.  
- **Cross-Layer Parameter Sharing**: Shares all parameters across layers (unlike BERT's per-layer), reducing to ~12M params while maintaining depth.  
- **SOP Loss**: Replaces NSP with Sentence-Order Prediction for better coherence learning.  
Advantages: 18x fewer params, 1.7x faster training, better on GLUE (up to +1-2 points on some tasks). Improvements post-2019 include integration in MobileBERT (further compression) and ALBERT-XXLarge variants. In 2024+ deployments (e.g., edge devices), ALBERT enables on-device NLP, but it trades some generalization for efficiency—less robust to domain shifts than BERT-large. Empirical scaling laws show ALBERT shifts the compute-accuracy Pareto frontier leftward.

**Follow-up Q&A:**  
**Q: How does ALBERT's parameter sharing impact gradient flow and training stability?**  
A: Sharing reduces redundancy but can cause gradient explosion/vanishing in deep stacks; mitigated by LayerNorm and careful initialization (e.g., scaled Xavier). It improves stability in low-data regimes but may underfit complex tasks.  

**Q: Compare ALBERT's factorization to other param-reduction techniques like pruning.**  
A: Factorization is structured (design-time), preserving dense ops for hardware efficiency. Pruning (post-training) removes weights sparsifying the model (up to 90% in Lottery Ticket Hypothesis), but requires sparse accelerators; ALBERT is denser and easier to deploy.

### 4. Models that optimize BERT's structure itself to achieve acceleration.

**Translated Reference Answer from AI Interview Guide:**  
1) DistilBERT: A miniaturized BERT trained using knowledge distillation on top of BERT.  
2) ALBERT: Lightweight BERT, using a factorization method to significantly reduce BERT's model parameters.  
3) TinyBERT: An improvement on DistilBERT—knowledge distillation targeted at Transformer structures, and for both pre-training and fine-tuning stages.

**Deeper Dive with Extended Knowledge:**  
These distill/optimize BERT (2018, 340M params for large) for speed/size:  
- **DistilBERT (2019)**: Halves layers (6 vs. 12), distills via soft labels + hidden states, achieving 97% GLUE performance at 40% size, 60% faster.  
- **ALBERT**: As above, factorization + sharing.  
- **TinyBERT (2019)**: Multi-stage distillation: general (pre-train) + task-specific (fine-tune), plus augmentation; 7.5x smaller, 9.4x faster.  
Extended: Post-2020, ELECTRA (adversarial pre-training) accelerates by replacing MLM with Replaced Token Detection; MobileBERT adds inverted bottlenecks. In 2025+ (e.g., Gemma or Phi models), structured pruning + low-rank adapters (LoRA) further optimize, shifting inference from O(n^2) to sub-quadratic via FlashAttention. Benchmarks: On mobile, these run at <100ms latency vs. BERT's seconds.

| Model | Key Optimization | Param Reduction | Speedup | Accuracy Retention |
|-------|------------------|-----------------|---------|--------------------|
| DistilBERT | Knowledge Distillation | 40% size | 1.6x | 97% GLUE |
| ALBERT | Factorization + Sharing | 89% reduction | 1.7x training | +1% on some tasks |
| TinyBERT | Multi-stage Distillation | 7.5x smaller | 9.4x | 96% BERT-base |

**Follow-up Q&A:**  
**Q: How does ELECTRA fit into BERT acceleration?**  
A: ELECTRA pre-trains faster (4x) by discriminating replaced tokens instead of masking, then distills to smaller models; better for low-resource acceleration than pure distillation.  

**Q: What role does quantization play in these optimized models?**  
A: Post-optimization, Int8/FP16 quantization (e.g., via Hugging Face Optimum) adds 2-4x speedup with <1% accuracy drop, but requires QAT for attention layers.

### 5. How is BERT model distillation done?

**Translated Reference Answer from AI Interview Guide:**  
1) Fine-tuning stage distillation. For example, in Distilled BiLSTM, the teacher model uses fine-tuned BERT-large, the student uses BiLSTM+ReLU for distillation, with objectives of hard label cross-entropy and MSE between logits.  
2) Distilling hidden layer knowledge. For example, BERT-PKD extracts knowledge from the teacher's intermediate layers, avoiding the risk of overfitting from distilling only the last layer.  
3) Pre-training stage distillation. For example, DistilBERT performs knowledge distillation in the pre-training stage, adding a new loss function—cosine embedding loss.  
4) Distilling attention matrices. For example, MiniLM only distills the last layer and only the KL divergence between the teacher and student matrices.

**Deeper Dive with Extended Knowledge:**  
Knowledge Distillation (KD, Hinton 2015) transfers "dark knowledge" from large teacher to small student via soft labels. For BERT:  
- **Logits KD**: Minimize KL-div/MSE on softened outputs (temp-scaled softmax).  
- **Feature KD**: Align hidden states/attentions (e.g., L2 or cosine loss).  
- **Stages**: Pre-train (broad knowledge), fine-tune (task-specific). Advanced: Patient KD (BERT-PKD) uses multiple layers; MiniLM focuses on self-attention relations for better transfer. Post-2022, PKD variants like CoDIR add contrastive losses; in LLMs (e.g., Llama distillation), progressive KD layers knowledge sequentially. Efficacy: Reduces params by 50-90% with 95%+ retention; challenges include mode collapse in students.

**Follow-up Q&A:**  
**Q: Why add cosine embedding loss in DistilBERT?**  
A: It aligns embedding spaces, capturing semantic similarity beyond logits, improving generalization (e.g., +0.5% on downstream).  

**Q: How does adversarial KD enhance BERT distillation?**  
A: Methods like Adv-KD add GAN-like discriminators to force student mimicry, robustifying against distribution shifts, but increases training complexity.

### 6. In duplicate detection projects, how are non-BERT models combined with BERT models? Is it bagging?

**Translated Reference Answer from AI Interview Guide:**  
How are the two types of models generally combined? Through model fusion methods, such as:  
1) Simple weighted fusion;  
2) Through ensemble learning methods.

**Deeper Dive with Extended Knowledge:**  
In text deduplication (e.g., plagiarism detection), combine classical (TF-IDF, BM25) with BERT (semantic embeddings) for speed/accuracy:  
- **Fusion**: Weighted average of scores (e.g., 0.7*BERT_sim + 0.3*Jaccard).  
- **Ensemble**: Stacking (meta-learner on outputs) or voting. Not purely bagging (bootstrap aggregating), as models differ; more like heterogeneous boosting. Extended: In Siamese BERT + CNN, fuse at feature level; post-2023, use CLIP-like multimodal if images involved. Advantages: BERT handles semantics, non-BERT speed/scalability; e.g., in search dedup, reduces false positives by 20%.

**Follow-up Q&A:**  
**Q: When to use stacking over simple fusion?**  
A: Stacking for non-linear interactions (e.g., XGBoost meta); fusion for simplicity/low-latency.  

**Q: How does this apply to large-scale dedup like in Google Search?**  
A: Hashing (MinHash) + BERT reranking; ensemble via MapReduce for distributed computing.

### 7. What are the methods for model ensemble, their pros/cons, and application scenarios?

**Translated Reference Answer from AI Interview Guide:**  
1) Boosting method: Train base classifiers serially, i.e., divide and conquer. In most cases, boosting results in smaller bias, disadvantages: data imbalance leads to decreased classification accuracy, training is time-consuming. Examples: Adaboost and Gradient Boosting.  
2) Bagging method: Train base classifiers in parallel, i.e., collective voting decision. In most cases, bagging results in smaller variance, disadvantages: poor performance with small data samples, and sometimes cannot guarantee relative independence between base classifiers, e.g., Random Forest.

**Deeper Dive with Extended Knowledge:**  
Ensembles combine weak learners for robustness:  
- **Boosting**: Sequential, focuses on errors (e.g., XGBoost adds regularization). Pros: Low bias, handles imbalance via weights. Cons: Overfits noise, serial slow. Scenarios: Tabular data, Kaggle comps.  
- **Bagging**: Parallel bootstraps, averages (reduces variance). Pros: Stable, parallelizable. Cons: High bias if bases weak, needs diversity. Scenarios: Trees (RF), unstable models.  
Extended: Stacking (meta-ensemble), Voting (hard/soft). Post-2020, neural ensembles (e.g., Deep Ensembles for uncertainty). In CV/NLP, ensembles boost SOTA by 1-3%.

| Method | Pros | Cons | Scenarios |
|--------|------|------|-----------|
| Boosting | Low bias, adaptive | Slow, sensitive to outliers | Imbalanced data, regression |
| Bagging | Low variance, fast | High bias, needs large data | Classification, forests |

**Follow-up Q&A:**  
**Q: How does Voting differ from Bagging?**  
A: Voting is aggregation (majority/avg); Bagging adds bootstrap sampling for diversity.  

**Q: In deep learning, why use Snapshot Ensembles?**  
A: Cycle learning rates to "snapshot" models at minima, ensembling for free diversity; good for CNNs with limited compute.

### 8. What are time series models?

**Translated Reference Answer from AI Interview Guide:**  
1) Models using RNN for CTR estimation. RNN is suitable for data with sequence (time series) relationships, CTR estimation focuses on learning combined features.  
2) DeepFM model. This model considers both low-order and high-order features, can handle time-series related data in search ads.

**Deeper Dive with Extended Knowledge:**  
Time series models forecast sequential data:  
- **Classical**: ARIMA (autoregressive), Prophet (additive).  
- **DL**: RNN/LSTM (handle dependencies), Transformers (e.g., Informer for long seq). For CTR (ads): DIN (attention on user history), DIEN (GRU evolution). DeepFM: FM for low-order + DNN for high-order. Extended: Post-2022, diffusion models (TimeDiff) for generative TS; in ads, TST (TS Transformer) beats RNNs on long horizons. Challenges: Seasonality, non-stationarity.

**Follow-up Q&A:**  
**Q: Why prefer Transformers over RNNs for TS?**  
A: Parallel, no sequential bottleneck; attention captures global patterns.  

**Q: How does DeepFM handle TS in ads?**  
A: Embeds timestamps as features; combines with FM for interactions.

### 9. Scenario Question: In a TikTok (Douyin) scenario, how to build a search ad system, how to balance an ad's commercial value and user experience?

**Translated Reference Answer from AI Interview Guide:**  
Mainly consider from two aspects: 1) Let users not think; 2) Do not trigger users' defenses. For example, ads before opening the APP, time cannot be too long, ad content can be related to users' frequent searches.

**Deeper Dive with Extended Knowledge:**  
Building: 1) Retrieval (BM25/ES), 2) Ranking (LTR with GBDT/DNN), 3) Auction (eCPM = bid * pCTR). Balance:  
- **UX**: Relevance (personalization via embeddings), non-intrusive (e.g., native ads), frequency caps.  
- **Value**: Maximize revenue via second-price auctions, A/B testing. Extended: In 2025+ (e.g., TikTok's algorithm), use RL (bandits) for dynamic balancing; metrics like dwell time vs. eRPM. Challenges: Ad fatigue, privacy (federated learning).

**Follow-up Q&A:**  
**Q: What role does multi-objective optimization play?**  
A: Pareto optimize UX (retention) + revenue via scalarization or MOEA.  

**Q: How to measure balance?**  
A: KPIs: Churn rate, ad skip rate vs. conversion rate.

### 10. How to optimize "relevance" in ad search scenarios?

**No reference answer provided in the query, but based on context, infer optimization strategies.**

**Deeper Dive with Extended Knowledge:**  
Relevance: Match query-ad via semantics/cos-sim. Optimize:  
- **Embeddings**: BERT/Siamese for query-ad vectors.  
- **Feedback**: Implicit (clicks) for reranking.  
- **Diversification**: MMR to avoid redundancy. Extended: In Google Ads, use RLHF-like fine-tuning; post-2023, multimodal (CLIP) for image ads. Metrics: NDCG, pCTR uplift (5-15%).

**Follow-up Q&A:**  
**Q: How does position bias affect relevance optimization?**  
A: Higher positions inflate clicks; debias via IPS (inverse propensity).  

**Q: What about cold-start ads?**  
A: Use content-based (TF-IDF) + exploration (epsilon-greedy).




