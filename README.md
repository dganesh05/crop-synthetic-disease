# Agrithon: An Analysis for Features Contributing to Crop Disease

## Introduction to the Problem

This project aims to solve a critical classification problem in agriculture: **predicting the presence of a plant disease based on observable symptoms and environmental factors**. Early and accurate detection of plant diseases is crucial for farmers to prevent crop loss, optimize the use of treatments like fungicides, and ensure food security.

The central question we are trying to answer is: **Can we build a reliable machine learning model that can accurately classify whether a plant has a disease, and if so, which specific symptoms are the most powerful predictors?** By answering this, we can move from a subjective assessment of plant health to a more data-driven diagnostic approach.

## The Dataset

The data for this project is a synthetic dataset sourced from Kaggle, designed to simulate a real-world agricultural survey.

**Source**: [Agri-thon Round 1: Synthetic Disease Presence Dataset](https://www.kaggle.com/datasets/mohammedarfathr/agrithon-round-1)

**About the Data**: The dataset consists of 150 samples and 31 columns. Each row represents an observation of a plant, and each column is a "Yes/No" question regarding a specific symptom (e.g., 'Is there a yellow halo around the spots?') or environmental condition (e.g., 'Was there poor air circulation in the field?'). 

The final column, 'Disease_Present', is our target variable, indicating whether the plant was diagnosed with the disease ('Present' or 'Not Present').

## Pre-processing the Data

Before any analysis or modeling, the data required a crucial pre-processing step: **numerical encoding**.

Machine learning algorithms perform mathematical operations and therefore require numerical input. Our dataset's features were categorical, with 'Yes' and 'No' answers. To prepare the data for analysis, we performed the following mapping:

* `'Yes'` was converted to `1`
* `'No'` was converted to `0`
* The target variable `'Present'` was converted to `1`
* The target variable `'Not Present'` was converted to `0`

This simple yet essential step transformed our entire dataset into a numerical format, making it possible to calculate correlations and train classification models.

## Data Understanding and Visualization

To understand the relationships within the data, we conducted an Exploratory Data Analysis (EDA) focused on two key questions: "Are any of our features redundant?" and "Which features are the most predictive of the disease?"

### A. Feature-to-Feature Correlation (Checking for Redundancy)

We started by creating a **heatmap** of a correlation matrix for all 30 features. This visualization helps us spot features that are highly correlated with each other (a problem known as multicollinearity).

![a heatmap representing the 30 features compared to each other. Most of the relationships are positive, but remained moderate and below 5.0](./img/feature_to_feature_correlation.png)

The heatmap revealed that most features had a weak positive correlation with each other. Crucially, there were no pairs with a very high correlation (e.g., > 0.7), which would have suggested they were redundant. This is an important "safety check." It told us that we didn't need to remove any features for being duplicates, and each one could potentially offer unique information to our model.

### B. Feature-to-Target Correlation (Finding the Best Predictors)

Next, we calculated the correlation of every feature directly with our target variable, `Disease_Present`, and visualized the results in a sorted bar chart.

*Note that if we had any redundant features from last step, we would have only chosen one of the set of redundant features based on simplicity.*

![a bar chart representing the 30 features scaled on their correlation to disease_present in descending order](./img/feature_to_target_correlation.png)

This chart clearly ranks the features by their predictive power. We observe that features like 'Are concentric rings visible clearly on the leaves?' and 'Are the leaf spots circular with concentric rings?' have the strongest relationship with the presence of the disease. Conversely, features like 'Is the farmer using resistant tomato varieties?' have almost no correlation. This visualization is the cornerstone of our feature selection strategy. Instead of using all 30 features blindly, this chart allowed us to make a data-driven decision to focus on a smaller subset of the most promising features.

Still, we don't know how many features we should select. That's where our next analysis comes in, where we select which top number of features work the best along with another crucial decision - the type of model.

## Modeling

Our modeling approach was designed to be systematic and experimental. We aimed to find the best-performing model by testing different feature sets and model architectures.

**Models Chosen**:
We started by testing a couple of common and powerful classification models to establish a performance baseline:

* Random Forest
* Logistic Regression
* Decision Tree
* XGBoost
* Naive Bayes
* SVM
* KNN

Our strategy was to first use these models to determine the optimal number of features to use, and then to fine-tune the best-performing model.

Because this is an excessive number of models, for future testing, analysts can choose to omit certain models. 

For this heatmap, we used cross-validation to get an averaged-out recall score for different top `n` values and types of models. We used a custom-made function to calculate the recall of each pair. 

![a heatmap representing 7 different classification models and n values from 1 to 25](./img/model_and_n_selection.png)

**Some factors that we look out for**

* Picking a too small `n` means that the model doesn't learn from other helpful factors, but a too bug `n` means that the model learns from too much noise, especially from columns that we established have a low correlation.

* A lot of combinations have a perfect recall of 1.00. This means that we have no false negatives in the prediction, which is a red flag especially for smaller datasets. It means it has perfectly memorized the dataset and will not perform well with new, unseen data. Also, note that these tend to be at `n > 14`, which aligns with the first point we made about incorporating too much noise. 

* Models that fluctuate are also red flags, because they are very sensitive to the data presented and have a lack of generalization. This is evident in the Decision tree, that fluctuates between 0.82 and 0.97. 

For our analysis, I chose to focus on a **Random Forest** model with `n=11`, because of its "high but not too high" n value, a consistent score, and a high enough recall that doesn't feel like overfitting (Beyond 11 is also flat and presents diminishing returns) However, users can feel free to choose a different model and n value and experiment with it. 

Because our dataset is small, tuning the hyperparameters is unlikely to yield any significant results. Again, analysts are encouraged to experiment with tuning, especially with larger datasets. 

## Evaluation

**Evaluation Metric: Recall**

For a disease detection problem, the most critical error is a false negative—failing to detect a disease when it is actually present. To minimize this specific error, we chose Recall as our primary evaluation metric.

Recall = True Positives / (True Positives + False Negatives)

A high recall score means our model is effective at finding all the positive cases.

![a confusion matrix representing 15 True Positives, 13 True Negatives, 2 False Positives, and 0 False Negatives](./img/confusion_matrix.png)

```
precision    recall  f1-score   support

           0       1.00      0.87      0.93        15
           1       0.88      1.00      0.94        15

    accuracy                           0.93        30
   macro avg       0.94      0.93      0.93        30
weighted avg       0.94      0.93      0.93        30
```

Because we only tested on 30 values, recall changed significantly from the 120 values we cross-fold validated on earlier. However, False Negatives are noticeably less than False positives, indicating a success for this analysis. More data can be used to test for a wider range. 

## Storytelling

Our journey began with a simple question: can we reliably predict plant disease from a list of symptoms? We started with 30 potential clues and, through exploratory analysis, quickly identified a smaller group of top predictors. Our initial modeling showed promise but also highlighted a critical danger: a simple Decision Tree could be easily tricked into "memorizing" the data, leading to a perfect but misleading score.

The key insight came from comparing multiple models. An ensemble model, the Random Forest, proved to be the superior choice. It was able to achieve a high recall score of over 90% without falling into the overfitting trap. This tells us that by combining the "opinions" of many decision trees, we can build a model that is both powerful and robust.

Also identifying the features that we determined contributed to disease presence: 

**Features that contribute to disease presence:**
* Is there any rotting seen on fruit?
* Are the lesions expanding over time?
* Are concentric rings visible clearly on the leaves?
* Are the leaf margins turning brown?
* Are the leaf spots circular with concentric rings?
* Is there a yellow halo around the spots?
* Are nearby tomato plants also showing similar symptoms?
* Was any fungicide recently applied?
* Is the disease more active during rainy days?
* Does the disease affect the whole plant?
* Is there any black moldy growth on the lesion?


**Features that did not contribute to disease presence:**
* Was there previous history of Early Blight in this field?
* Was the field irrigated from overhead sprinklers?
* Is the infection found only on mature leaves?
* Are the affected leaves wilting?
* Are stems or fruits also affected?
* Are the leaf veins visible through the lesion?
* Does the leaf show signs of early yellowing?
* Does the disease begin on the lower leaves?
* Was there poor air circulation in the field?
* Is the farmer using resistant tomato varieties?
* Is the damage uniform across the field?
* Are pruning and sanitation practices followed?
* Are multiple spots merging to form large blotches?
* Is the spot size more than 5mm in diameter?
* Are the lesions visible on both sides of the leaf?
* Is there any other crop in the field showing similar spots?
* Is the infection spreading upward on the plant?
* Is the center of the spot dry and brown?
* Is the plant under moisture stress?

According to the model, the features that contribute most to a diagnosis are the unambiguous, advanced symptoms of the disease.

* **Distinct Lesion Characteristics**: The model heavily weights the quality of the spots, not just their presence. Examples are 'Are concentric rings visible clearly', 'Is there a yellow halo around the spots?', and 'Is there any black moldy growth on the lesion?'

* **Severe Progression**: Symptoms like 'Is there any rotting seen on fruit?' and 'Does the disease affect the whole plant?' are signs of a well-established infection. 

* **Contagion and Environment**: The model also picked up on clues that the disease is active and spreading, such as 'Are nearby tomato plants also showing similar symptoms?' and 'Is the disease more active during rainy days?'.

* The **surprising feature** here is 'Was any fungicide recently applied?'. This is likely a strong predictor because farmers only apply fungicide when they already see or strongly suspect a disease. The model learned that this action is a powerful proxy for human observation of the disease. Future analyses could omit this feature and see if the recall changes. 


The model also found that many features which seem important are actually not reliable enough for a final diagnosis. These fall into two main categories:

* **General "Sick Plant" Symptoms**: Many non-contributing features are things that could be wrong with any plant for any number of reasons. 'Are the affected leaves wilting?', 'Does the leaf show signs of early yellowing?', and 'Is the plant under moisture stress?' are too generic and don't specifically point to this disease.

* **Background & Farming Practices**: The model concluded that the current, visible evidence is far more important than the general field conditions. Factors like 'Was there previous history of Early Blight', 'Was there poor air circulation', and 'Is the farmer using resistant varieties?' are risk factors, not diagnostic proof.

* The most **interesting insight** comes from the symptoms it ignored, like 'Are multiple spots merging to form large blotches?' and 'Is the spot size more than 5mm in diameter?'. This tells us the model learned that the specific appearance of a spot (like having rings or mold) is a much more powerful clue than its size or how many there are. Future analyses could include more appearance-based features to see if they improve predictions. 

## Limitations

This analysis would benefit from further exploration and includes several sections where readers can experiment and further improve analysis:

* Limited data: We only had 150 rows to work with, 120 for training and only 30 for testing. This made it suitable for exploratory analysis but needed more data so we can see the impact of the parameters we selected

* Parameter tuning: Because of limited size, experimentation with the parameters of our model would not lead to much useful results. However, analysts are encouraged to tinker with these parameters, as well as try other models and `n` sizes to visualize their outcome. 

## Impact

* **Positive Impact**: This project demonstrates a clear pathway to creating data-driven tools for agriculture. A reliable model could be integrated into a mobile app allowing farmers to quickly diagnose potential diseases by answering a simple questionnaire, or into Computer Vision infrastructure that could scan for symptoms and take action as needed  (such as [greenhouses in the Netherlands](https://youtu.be/lIvrIKaNCRE?feature=shared&t=632)). This could lead to more targeted and efficient use of fungicides, reducing both costs and environmental impact, and ultimately contributing to greater crop yields.

* **Potential Negative Impact**: Over-reliance on such a model could be a risk. If a new disease variant emerges with different symptoms, the model would fail to detect it. Furthermore, there's an ethical consideration regarding accessibility; if such a tool is only available on expensive devices or requires a paid subscription, it could create a technology gap that disadvantages smaller, less-resourced farms, widening the disparity in agricultural productivity. 

## References

Database used: [Agri-thon Round 1: Synthetic Disease Presence Dataset](https://www.kaggle.com/datasets/mohammedarfathr/agrithon-round-1)

```
.
├── img/ - all visualizations generated during the analysis
│ ├── confusion_matrix.png - the confusion matrix used to check for model performance at the end
│ ├── feature_to_feature_correlation.png - a heatmap used to perform feature-to-feature analysis to check for multicollinearity
│ ├── feature_to_target_correlation.png - a barplot used to perform feature-to-target analysis to check for correlation with targer
│ ├── model_and_n_selection.png - compares the performance of various classification models and n-values for top n features
├── project-notebook.ipynb - the main Jupyter Notebook that contains all Python code for data pre-processing, exploratory analysis, model selection, and evaluation.
└── README.md - this documentation, explaining the project's goals, methodology, findings, and repository structure.
```