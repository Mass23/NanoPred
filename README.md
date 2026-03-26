# NanoPred Project Documentation

## Overview
NanoPred is a comprehensive project aimed at predicting percent identity between high error-rate (e.g. Nanopore) sequencing data by modelling it based on sequences and quality data. The aim is to allow OTU clustering accounting for sequence quality. The project is divided into three main phases:

1. **In-Silico Dataset Creation**  
2. **Ensemble ML Model Building**  
3. **Statistical Analysis and Visualization**  

---

## Phase 1: In-Silico Dataset Creation with Adaptive Primer Trimming and K-Means Clustering
### 1.1 Dataset Creation
- Generate synthetic datasets that mimic pairwise comparisons of sequences based on SILVA database.
- Utilize adaptive primer trimming techniques to refine input sequences.

### 1.2 K-Means Clustering
- Implement K-means clustering to categorize the datasets based on specific features, improving the model's focus on relevant data patterns.
- Evaluate the clusters to ensure meaningful categorization of the dataset.

---

## Phase 2: Ensemble ML Model Building with 3-Fold CV and Feature Selection
### 2.1 Model Building
- Construct multiple machine learning models to form an ensemble approach, enhancing prediction accuracy.
- Techniques include Random Forest, Gradient Boosting, and Support Vector Machines.

### 2.2 Cross-Validation
- Utilize 3-fold cross-validation to assess model robustness and prevent overfitting.

### 2.3 Feature Selection
- Apply feature selection techniques to identify the most influential features on model performance, improving overall model efficiency.

---

## Phase 3: Statistical Analysis of Variable Selection Patterns and Visualization
### 3.1 Variable Selection Patterns
- Conduct statistical analysis to observe how variable selections vary across different model runs.
- Identify patterns that may indicate property relevance or model bias.

### 3.2 Quality vs. Accuracy Metrics Visualization
- Develop visualizations to compare quality metrics against accuracy metrics, allowing for intuitive understanding of model performance.
- Use graphical representations to communicate findings effectively.

---

## Conclusion
NanoPred aims to deliver a robust framework for sequence pairwise identity prediction through machine learning methods based on quality and sequence data. By documenting our approach comprehensively, we provide future users and researchers with insights into our process and outcomes.  

## Acknowledgments
- Special thanks to contributors and data scientists involved in the development of this project.
