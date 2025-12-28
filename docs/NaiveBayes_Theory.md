# Final Project

## Naive Bayes Classification

### 1. Introduction 

Naive Bayes classifiers are a part of probabilistic classifiers and supervised learning algorithms that assume that given a class, features are conditionally independent. This means that every feature contributes to the prediction without any connection with the other. Common applications are:
* spam detection : categorizes emails as spam or not spam
* sentiment analysis : categorizes documents using text classification
* medical diagnosis : categorizes diseases using symptoms
* weather prediction: categorizes forecast using different conditions.

### 2. Bayes’ Theorem and Assumptions
Bayes' theorem states that given a feature vector $X$ and class variable $C$, follwoing reationship is formed : 
$$P(C|X) = \frac{P(X|C).P(C)}{P(X)}$$
Where
* $P(C∣X)$ : posterior probability, prbability of class $C$ given features $X$.
* $P(C)$ : prior probability of class $C$.
* $P(X∣C)$ : likelihood, probability of features $X$ given class $C$.
* $P(X)$ : evidence or marginal likelihood.

The independence assumptions states that:
$$P(X \mid C) =P(x1​,x2​,…,x_{n}​∣C)= \prod_{i=1}^{n} P(x_i \mid C)$$

### 3. Mathematical Formulation of Naive Bayes
Given feature vector $X = (x_1, x_2, \ldots, x_d)$ and possible classes $(C_1, C_2, \ldots, C_K)$, mathematical foundations and predictions of naive bayes can be formed using the independence assumption:
$$P(C_k​∣X) = \frac{P(C_k) . \prod_{i=1}^{n} P(x_i \mid C)}{P(X)}$$

The joint probability of features given a class is
$$P(x_1,x_2,x_3 | C)$$

Using the chain rule of probability
$$P(x_1,x_2,x_3∣ C)=P(x_1∣x_2,x_3,C).P(x_2∣x_3,C).P(x_3∣C)$$

After appliying independence assumption
$$P(x_1∣x_2,x_3,C)=P(x_1∣C)$$
$$P(x_2∣x_3,C)=P(x_2∣C)$$

Finally, we get
$$P(x_1,x_2,x_3∣C)=P(x_1∣C).P(x_2∣C).P(x_3∣C)$$

After generalizing the feature, we get
$$P(x|C) =\prod_{i=1}^{n} P(x_i \mid C)$$

* For any given input, the demoninator $P(X)$ is constant, hence we can write:
$$P(C_k​∣X) ∝ P(C_k) . \prod_{i=1}^{n} P(x_i \mid C)$$
where the $\propto$ symbol shows that as the term on left side of equation increases, the numerator on the right side also increases and we get our predicition using the highest numerator for each class.
* Now we can construct a naive bayes classifier:
$$\hat{C} = \arg\max_{C_k} P(C_k)\prod_{i=1}^{d} P(x_i \mid C_k)$$
where $(k \in \{\text{1..}, \text{k}\})$
 
### 4. Fitting a Naive Bayes Model and making Predictions
We will use a dataset an an example to see how **categorical naive bayes** fits and predicts data.
| Sample | Weather | Temperature | Traffic | Attend? |
|--------|----------|-------------|----------|----------|
| 1 | Sunny   | Hot   | High  | Yes |
| 2 | Sunny   | Cool  | Low   | No  |
| 3 | Rainy   | Cool  | Medium| Yes |
| 4 | Sunny   | Hot   | Medium| Yes |
| 5 | Rainy   | Hot   | Low   | No  |

* There are 2 class variables $(C \in \{\text{Yes}, \text{No}\}$
* Features are 
$(x_1): Weather$ [Sunny, Rainy]  
$(x_2): Temperature$ [Hot, Cool]  
$(x_3): Traffic$ [High, Low, Medium]  

* Assuming that the features are conditionally independent
* Pre-computation from dataset of $5$ rows:
$P(Yes) = \frac{3}{5}$
$P(No) = \frac{2}{5}$

* Compute Likelihoods using:
$$
P(\text{Feature}=x_i \mid C=c) = \frac{\text{Count of samples where Feature = x_i and } C=c}{\text{Count}(C=c)}
$$

#### Weather

| Weather | Count in Yes | Count in No | P(Yes) | P(No)
|--------|----------------|--------------|--------------|--------------|
| Sunny | 2 | 1 | $\frac{2}{3}$ | $\frac{1}{2}$
| Rainy | 1 | 1 | $\frac{1}{3}$ | $\frac{1}{2}$
|Total  | 3 | 2 | $100 \%$ | $100 \%$

#### Temperature

| Temperature | Count in Yes | Count in No | P(Yes) | P(No)
|--------|----------------|--------------|--------------|--------------|
| Hot | 2 | 1 | $\frac{2}{3}$ | $\frac{1}{2}$
| Cool | 1 | 1 | $\frac{1}{3}$ | $\frac{1}{2}$
|Total  | 3 | 2 | $100 \%$ | $100 \%$

#### Traffic
| Weather | Count in Yes | Count in No | P(Yes) | P(No)
|--------|----------------|--------------|--------------|--------------|
| High | 1 | 0 | $\frac{1}{3}$ | $\frac{0}{2}$
| Medium | 1 | 1 | $\frac{1}{3}$ | $\frac{1}{2}$
| Low | 1 | 1 | $\frac{1}{3}$ | $\frac{1}{2}$
|Total  | 3 | 2 | $100 \%$ | $100 \%$

* Consider an example of features to predict:
$X = (Sunny, Hot, Low)$

#### Compute conditional probabilities:
| Feature | Value | $P(Value\| Yes)$ | $P(Value\| No)$ |
|--------|----------------|--------------|--------------|
| Weather | Sunny | $\frac{2}{3}$ | $\frac{1}{2}$ | $\frac{1}{2}$
| Temperature | Hot | $\frac{2}{3}$ | $\frac{1}{2}$ | $\frac{1}{2}$
|Traffic  | Low | $\frac{1}{3}$ | $\frac{1}{2}$ | 

#### Posterior Probabilities:
* For class = 'Yes'
$$P(\text{Yes} \mid X) \propto
\left(\frac{3}{5}\right)
\left(\frac{2}{3}\right)
\left(\frac{2}{3}\right)
\left(\frac{1}{3}\right).$$

$$= \frac{3}{5} \cdot \frac{4}{9} \cdot \frac{1}{3}
= \frac{12}{135}
= 0.0889$$

* For class = 'No'
$$P(\text{No} \mid X) \propto
\left(\frac{2}{5}\right)
\left(\frac{1}{2}\right)
\left(\frac{1}{2}\right)
\left(\frac{1}{2}\right).$$

$$= \frac{2}{5} \cdot \frac{1}{8}
= \frac{2}{40}
= 0.05$$

#### Normalize probabilities:
Normalizing converts these relative weights into a proper probability distribution
$$P(\text{Yes} \mid X) = 
\frac{0.0889}{0.0889 + 0.05}
= 0.64$$

$$P(\text{No} \mid X) = 
\frac{0.05}{0.0889 + 0.05}
= 0.36.$$

#### Final Prediction:
$$P(Yes|X)>P(No|X)$$
Hence the **Categorical Naive Bayes** model predicts that person will attent the event.

### 5. Variants of Naive Bayes Model
### 5.1 Gaussian:
In this model, the continuous features are assumed to have gaussian distributuion. When plotted, it represents a normal distribution because the curve is bell shaped and symmetirc around the mean of feature values:
![image](https://hackmd.io/_uploads/r1HpE-WfZg.png)

* Mathematically:
$$P(x_i | y) = \frac{1}{\sqrt{2.\pi.\sigma^2_y}}.exp\left(-\frac{(x_i - u_y)^2}{2.\sigma^2_y}\right)$$

Where, for class y:
* $u_y$ is the mean of feature $x_i$.
* $\sigma^2_y$ is the variance of feature $x_i$.

#### Short Example: Diamond clarity score and Grade
| Sample | Clarity Score | Grade |
|--------|----------------|--------|
| 1 | 9.0 | A |
| 2 | 8.5 | A |
| 3 | 9.5 | A |
| 4 | 6.0 | B |
| 5 | 6.5 | B |

#### Separate by Class
* Class $A : [9, 8.5, 9.5]$
$P(A) = \frac{3}{5}$

* Class $B : [6.0, 6.5]$
$P(B) = \frac{2}{5}$

* #### Calculate Mean and Variance:
* Class $A$ : 
$u_A = \frac{9+8.5+9.5}{3} = 9.0$
$\sigma^2_A =\frac{(9-9)^2+(8.5-9)^2+(9.5-9)^2}{3} = 0.1667$

* Class $B$ : 
$u_B = \frac{6+6.5}{2} = 6.25$
$\sigma^2_B =\frac{(6-6.25)^2+(6.5-6.25)^2}{3} = 0.0625$

#### Gaussian Likelihood
We want to predict grade for $Clarity \ Score = 8.8$
Using the equation:
* Class $A$
$P(8.8 | A) = \frac{1}{\sqrt{2.\pi.0.1667^2}}.exp\left(-\frac{(8.8 - 9.0)^2}{2.(1667^2)}\right) \approx 0.863$

* Class $B$
$P(8.8 | B) = \frac{1}{\sqrt{2.\pi.0.0625^2}}.exp\left(-\frac{(8.8 - 6.25)^2}{2.(0.0625^2)}\right) \approx 1.59 \times 10^{-23}$

#### Posterior probabilities
* Class $A$
$$P(\text{A} \mid 8.8) \propto
\left(\frac{3}{5}\right). 0.863 = 0.5178$$

* Class $B$
$$P(\text{B} \mid 8.8) \propto
\left(\frac{2}{5}\right). (1.59 \times 10^{-23}) = 6.36 \times 10^{-23}$$

#### Normalize probabilities
$$P(\text{A} \mid 8.8) = 
\frac{0.5178}{0.5178 + (6.36 \times 10^{-23})}
\approx 1$$

$$P(\text{B} \mid 8.8) = 
\frac{(6.36 \ x \ 10^{-23})}{(6.36 \times 10^{-23}) + 0.5178} \approx 0$$

#### Final Predcition
$$P(A|8.8)>P(B|8.8)$$
Hence gaussian naive bayes predicts that for clarity $8.8$, diamond is grade $A$.

#### Pseudocode Version:
##### Training and Fitting the model
```
function TRAIN_GAUSSIAN_NB(X, y):

    // X is an (n_samples × n_features) matrix
    // y is a vector of class labels

    classes = unique(y)
    
    for each class c in classes:

        // Extract all samples belonging to class c
        Xc = all rows of X where label == c
        Nc = number of rows in Xc
        N  = total number of samples

        // Compute prior probability
        priors[c] = Nc / N

        // Compute mean and variance for each feature
        for each feature j:
            means[c][j]     = mean of Xc[:, j]
            variances[c][j] = variance of Xc[:, j]

    return priors, means, variances
```
##### Prediction Phase
```
function PREDICT_GAUSSIAN_NB(x_new, priors, means, variances):

    // x_new is a 1 × n_features vector
    // priors, means, variances are learned from training

    classes = all keys in priors

    for each class c in classes:

        // Start with the prior probability
        posterior[c] = priors[c]

        // Multiply likelihoods of each feature
        for each feature j:

            μ  = means[c][j]
            σ2 = variances[c][j]

            // Compute Gaussian probability density
            p = (1 / sqrt(2 * π * σ2)) * exp( - ((x_new[j] - μ)^2) / (2 * σ2) )

            // Multiply with posterior
            posterior[c] = posterior[c] * p

    // Choose the class with the highest posterior probability
    predicted_class = class with maximum posterior[c]

    return predicted_class

```
### 5.2 Multinomial
This model is widely used in text classification for multinomially distributed data. Counts are used to represent frequency of words. Documents including spam emails can be detected using the frequency of words.
* It is assumed that each word is independent of the other in a feature and hence the final formula remains same.
$${P(C|X) \propto P(C) \times \prod_{i=1}^{n} P(x_i | C)}$$

* The differentiating part is how this model computes the likelihoods.
* Laplace smoothing is used in calculations of likelihood probabilities for handling $0$ counts.
$${P(x_i | C) = \frac{\text{Count}(x_i, C) + \alpha}{\text{Count}(C) + \alpha \cdot d}}$$

Where:
$\text{Count}(x_i, C)$: The number of times the feature $x_i$ occurs in a particular class $C$.
$\text{Count}(C)$: The total number of features accross all classes.
$\alpha$ (Alpha): The smoothing parameter that eliminates zero probabilities for features.
$d$: The number of unique features

Consider an example dataset to predict spam email from given words:
| ID | Message                  | Class     |
|----|--------------------------|-----------|
| 1  | win money now            | spam      |
| 2  | win prize now            | spam      |
| 3  | meeting schedule today   | not\_spam |
| 4  | schedule meeting today   | not\_spam |

#### Vocabulary set of unique words
Vocabulary = $(\text{'win', 'money','prize','now','meeting','schedule','today'})$
Vocabulary size $V = 7$

#### Word frequency by class
* spam class$(1,2)$
1. win : 2
2. money : 1 
3. prize : 1
4. now : 2

Total count : 6

* not_spam class $(3,4)$
1. meeting : 2
2. schedule : 2
3. today : 2

Total count : 6

#### Predicting following text message
$d = (\text{'win  now  schedule'})$

#### Prior probabilities
$P(spam) = \frac{1}{2} = 0.5$
$P(not\_spam) = \frac{1}{2} = 0.5$

#### Likelihoods ($\alpha = 1$)
* spam class
$P(\text{win}\mid \text{spam}) = \frac{2+1}{6+7}=\frac{3}{13}\approx 0.2308$
$P(\text{money}\mid \text{spam}) = \frac{1+1}{6+7}=\frac{2}{13}\approx 0.1538$
$P(\text{now}\mid \text{spam}) = \frac{2+1}{6+7}=\frac{3}{13}\approx 0.2308$
$P(\text{prize}\mid \text{spam}) = \frac{1+1}{6+7}=\frac{2}{13}\approx 0.1538$
$P(\text{meeting}\mid \text{spam}) = \frac{0+1}{6+7}=\frac{1}{13}\approx 0.07692$
$P(\text{schedule}\mid \text{spam}) = \frac{1}{13}\approx 0.07692$
$P(\text{today}\mid \text{spam}) = \frac{1}{13}\approx 0.07692$

* not\_spam class
$P(\text{meeting}\mid \text{not}) = \frac{2+1}{13}=\frac{3}{13}\approx 0.2308$
$P(\text{schedule}\mid \text{not}) = \frac{2+1}{13}=\frac{3}{13}\approx 0.2308$
$P(\text{today}\mid \text{not}) = \frac{2+1}{13}=\frac{3}{13}\approx 0.2308$
$P(\text{win}\mid \text{not}) = \frac{0+1}{13}=\frac{1}{13}\approx 0.07692$
$P(\text{money}\mid \text{not}) = \frac{1}{13}\approx 0.07692$
$P(\text{now}\mid \text{not}) = \frac{1}{13}\approx 0.07692$
$P(\text{prize}\mid \text{not}) = \frac{1}{13}\approx 0.07692.$

#### Posterior probabilites
* spam class
$$P(\text{spam} \mid d) \propto P(\text {spam}).P(\text{win}\mid \text{spam}).P(\text{now}\mid \text{spam}).P(\text{meeting}\mid \text{spam})$$
$$ =0.5.\frac{3}{13}.\frac{3}{13}.\frac{1}{13}$$
$$= \frac{9}{4394}\approx 0.002049$$

* not\_spam class
$$P(\text{not_spam} \mid d) \propto P(\text {not_spam}).P(\text{win}\mid \text{not_spam}).P(\text{now}\mid \text{not_spam}).P(\text{meeting}\mid \text{not_spam})$$
$$ =0.5.\frac{1}{13}.\frac{1}{13}.\frac{3}{13}$$
$$= \frac{3}{4394}\approx 0.000683$$

#### Normalize posteriors
$$\frac{9}{4394}+\frac{3}{4394}=\frac{12}{4394}$$
* $$P(\text{spam} \mid d)  = \frac{9/4394}{12/4394}=\frac{9}{12}=0.75$$
* $$P(\text{not_spam} \mid d)  = \frac{3/4394}{12/4394}=\frac{3}{12}=0.25$$

#### Final Prediction
$$\hat{C}=\text{spam} \quad(\text{since }0.75>0.25).$$

#### Pseudocode Version:
##### Training and Fitting the model
```
function TRAIN_MULTINOMIAL_NB(X, y, alpha):

    // X is n_samples × n_features matrix of counts
    // y is vector of class labels
    // alpha is Laplace smoothing parameter

    classes = unique(y)

    for each class c in classes:

        // Extract all samples belonging to class c
        Xc = all rows of X where label == c
        Nc = number of rows in Xc
        N  = total number of samples

        // Compute prior probability
        priors[c] = Nc / N

        // Compute total count of all features in class c
        total_count = sum of all elements in Xc

        for each feature j:

            // Count of feature j in class c
            feature_count = sum of Xc[:, j]

            // Compute likelihood with Laplace smoothing
            likelihoods[c][j] = (feature_count + alpha) / (total_count + alpha * n_features)

    return priors, likelihoods
```
##### Prediction Phase
```
function PREDICT_MULTINOMIAL_NB(x_new, priors, likelihoods):

    // x_new is a vector of feature counts
    // priors and likelihoods learned from training

    classes = all keys in priors

    for each class c in classes:

        posterior[c] = priors[c]

        for each feature j:

            // Multiply probability for feature j
            posterior[c] = posterior[c] * (likelihoods[c][j] ^ x_new[j])

    // Choose class with highest posterior
    predicted_class = class with maximum posterior[c]

    return predicted_class
```
### 5.3 Bernoulli
In this model, binary features like $(\text{yes/no, 0/1, and True/False})$ are used and data is distributed in a multivariate bernoulli distributuion. Used in scenarios where presence or absence of a feature contributes more than its frequency.
* Likelihood of each feature is calculated by defining bernouli distribution:
$${P(x_i | C) = \begin{cases} p_{ic} & \text{if } x_i = 1 \text{ (feature present)} \\ 1 - p_{ic} & \text{if } x_i = 0 \text{ (feature absent)} \end{cases}}$$
$${P(x_i | C) = p_{ic}^{x_i} (1 - p_{ic})^{(1-x_i)}}$$
$$P(X|C) = \prod_{i=1}^{n} p_{ic}^{x_i} (1 - p_{ic})^{(1-x_i)}$$

Consider the same dataset example for spam email.
| ID | Message                  | Class     |
|----|--------------------------|-----------|
| 1  | win money now            | spam      |
| 2  | win prize now            | spam      |
| 3  | meeting schedule today   | not\_spam |
| 4  | schedule meeting today   | not\_spam |

#### Vocabulary set of unique words
Vocabulary = $(\text{'win', 'money','prize','now','meeting','schedule','today'})$
Vocabulary size $V = 7$

#### Binary Feature Matrix (Absence = 0, Presence =1)
| ID | win |  money | prize | now | meeting | schedule | today | Class    | 
|----|-----|--------|-------|-----|---------|----------|-------|----------|
| 1  | 1   | 1      |   0   |  1  | 0       | 0        | 0     | spam     | 
| 2  | 1   | 0      |   1   |  1  | 0       | 0        | 1     | spam     | 
| 3  | 0   | 0      |   0   |  0  | 1       | 1        | 1     | not\_spam|
| 4  | 0   | 0      |   0   |  0  | 1       | 1        | 1     | not\_spam|    

#### Laplace smoothing
$P(w_i=1\mid C)=\frac{n_{w,C}+\alpha}{N_C + 2\alpha}$
where $(n_{w,C})$ = count of words in class and $N_C =2$ represents number of documents per class,

#### Prior probabilities
$P(spam) = \frac{1}{2} = 0.5$
$P(not\_spam) = \frac{1}{2} = 0.5$

#### Word probabilities
* spam class
$P(\text{win}\mid \text{spam}) = \frac{2+1}{4}=\tfrac{3}{4}=0.75$
$P(\text{money}\mid \text{spam}) = \frac{1+1}{4}=\tfrac{2}{4}=0.50$
$P(\text{now}\mid \text{spam}) = \frac{2+1}{4}=\tfrac{3}{4}=0.75$
$P(\text{prize}\mid \text{spam}) = \frac{1+1}{4}=\tfrac{2}{4}=0.50$
$P(\text{meeting}\mid \text{spam}) = \frac{0+1}{4}=\tfrac{1}{4}=0.25$
$P(\text{schedule}\mid \text{spam}) = \tfrac{1}{4}=0.25$
$P(\text{today}\mid \text{spam}) = \tfrac{1}{4}=0.25.$

* not\_spam class
$P(\text{meeting}\mid \text{not}) = \frac{2+1}{4}=\tfrac{3}{4}=0.75$
$P(\text{schedule}\mid \text{not}) = \tfrac{3}{4}=0.75$
$P(\text{today}\mid \text{not}) = \tfrac{3}{4}=0.75$
$P(\text{win}\mid \text{not}) = \frac{0+1}{4}=\tfrac{1}{4}=0.25$
$P(\text{money}\mid \text{not}) = \tfrac{1}{4}=0.25$
$P(\text{now}\mid \text{not}) = \tfrac{1}{4}=0.25$
$P(\text{prize}\mid \text{not}) = \tfrac{1}{4}=0.25.$

#### Predicting following text message
$d = (\text{'win  now  schedule'})$
$Present \ words: (\text{win},\text{now},\text{schedule}).$ 
$Absent \ words:(\text{money},\text{prize},\text{meeting},\text{today}).$


#### Posterior probabilities
* **spam class**
$$P(\text{spam} \mid d) \propto P(\text {spam}).P(\text{win}\mid \text{spam}).P(\text{now}\mid \text{spam}).P(\text{meeting}\mid \text{spam}).P(\text{absent}\mid \text{spam})$$
1. Present terms: $\theta_{\text{win, Spam}} \times \theta_{\text{now, Spam}} \times \theta_{\text{schedule, Spam}}$

$$= 0.5 \times0.75 \times 0.75 \times 0.25 = 0.140625.$$

2. Absent terms :$(1-\theta_{\text{money, Spam}}) \times (1-\theta_{\text{prize, Spam}}) \times (1-\theta_{\text{meeting, Spam}}) \times (1-\theta_{\text{today, Spam}})$
$$(1-0.50) \times (1-0.50) \times (1-0.25) \times (1-0.25) = 0.5 \times 0.5 \times 0.75 \times 0.75 = 0.140625$$
$$\text{Score}_{\text{Spam}} = 0.5 \times 0.140625 \times 0.140625 \approx {0.00988}$$

* not\_spam class
$$P(\text{not_spam} \mid d) \propto P(\text {not_spam}).P(\text{win}\mid \text{not_spam}).P(\text{now}\mid \text{not_spam}).P(\text{meeting}\mid \text{not_spam}).P(\text{absent}\mid \text{not_spam})$$
1. Present terms: $\theta_{\text{win, Not Spam}} \times \theta_{\text{now, Not Spam}} \times \theta_{\text{schedule, Not Spam}}$
$$=0.25 \times 0.25 \times 0.75 = 0.046875$$

2. Absent terms:$(1-\theta_{\text{money, Not Spam}}) \times (1-\theta_{\text{prize, Not Spam}}) \times (1-\theta_{\text{meeting, Not Spam}}) \times (1-\theta_{\text{today, Not Spam}})$
$$(1-0.25) \times (1-0.25) \times (1-0.75) \times (1-0.75) = 0.75 \times 0.75 \times 0.25 \times 0.25 = 0.03515625$$

$$\text{Score}_{\text{Not Spam}} = 0.5 \times 0.046875 \times 0.03515625 \approx \mathbf{0.00082}$$

#### Normalize Probabilities
$S = 0.00988 + 0.00082 = 0.0107$
$P(\text{spam}\mid\text{msg}) = \frac{0.00988}{S} \approx 0.923$
$P(\text{not}\mid\text{msg})  = \frac{0.00082}{S} \approx 0.077.$

#### Final Prediction
$$\hat{C} = \text{spam}\quad(\text{since }0.923>0.077)$$

#### Pseudocode Version:
##### Training and Fitting the model

```
function TRAIN_BERNOULLI_NB(X, y, alpha):

    // X is n_samples × n_features binary matrix (1 = present, 0 = absent)
    // y is vector of class labels
    // alpha is Laplace smoothing parameter

    classes = unique(y)

    for each class c in classes:

        // Extract all samples belonging to class c
        Xc = all rows of X where label == c
        Nc = number of rows in Xc
        N  = total number of samples

        // Compute prior probability
        priors[c] = Nc / N

        for each feature j:

            // Count how many documents have feature j present
            count_present = sum of Xc[:, j]

            // Compute probability of feature presence with Laplace smoothing
            prob_present[c][j] = (count_present + alpha) / (Nc + 2 * alpha)

    return priors, prob_present
```
##### Prediction Phase

```
function PREDICT_BERNOULLI_NB(x_new, priors, prob_present):

    // x_new is a binary feature vector
    // priors and prob_present learned from training

    classes = all keys in priors

    for each class c in classes:

        posterior[c] = priors[c]

        for each feature j:

            if x_new[j] == 1:
                posterior[c] = posterior[c] * prob_present[c][j]
            else:
                posterior[c] = posterior[c] * (1 - prob_present[c][j])

    // Choose class with highest posterior
    predicted_class = class with maximum posterior[c]

    return predicted_class
```


### 6. Strengths and Weaknesses of the Algorithm
#### Strengths
* Since the calculations are less complex, naive bayes model is **computationally efficient** and fast to train.
* The esitmation of parameters is done using assumption of independent features, hence increasing the size of data doesn't affect fiiting proccess and this makes the model easiliy **scalable**.
* The model performs well in **spaces of high dimension**, for example in text classification with hundreds of words as features.
* The model is very effective even with **small training set**.

#### Weaknesses
* The model makes predictions on the assumuption that features are **conditionally independent** and in real world scenario, they can be dependent on each other thus affecting the probability estimates.
* In a categorical dataset, some features may have a **zero probability** for a given class and this affects the likelihood value when those feature probabilities are multiplied together.
* In gaussian naive bayes, it is assumed that dataset follows the normal dsitribution strictly and if this is violated, the likelihood values will not be that accurate.

### 7. Glossary

* **Feature [X]** 
An individual measurable property or characteristic of a sample. In text classification, words represent features; in numerical datasets, we can have columns like lentgh, clarity_score, price etc.

* **Sample**
A single data point in the dataset, represented by a set of features and a class label.

* **Class**
The category that a sample belongs to. For example, in spam detection: spam or not_spam.

*  **Prior Probability (P(C))**
The probability of each unqiue class in a dataset.
$$P(C) = \frac{\text{Number of samples in class C}}{\text{Total number of Samples}}$$

*  **Likelihood (P(x|C))**
The probability of the features x given that they belong to class C.
$$P(X \mid C) =P(x1​,x2​,…,x_{n}​∣C)= \prod_{i=1}^{n} P(x_i \mid C)$$

* **Posterior Probability**
The probability that for given feature x, the sample belongs to class C. Computed using naive bayes theorem:

$$P(C|X) = \frac{P(C).P(X|C)}{P(X)}$$

* **Naive Bayes Variants**
Categorical, Gaussian, Multinomial, Bernouilli.

* **Laplace Smoothing (α)**
A method to eliminate zero probabilities when a feature does not appear in a class during training.

*  **Vocabulary**
The set of all unique features (words) considered in a text dataset.








