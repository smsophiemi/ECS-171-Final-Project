# ECS-171-Final-Project

## Group Members
Pengcheng Cao, Zahira Ghazali, Denise Kwong, Sophie Mi, Lingfeng Pan, Zihan Wang

# Introduction

## Abstract
The dataset we are using records the standardized test scores of high school students in the United States in various subjects and some personal identification details of each student. (https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

# NOTE: the data was created by Royce Kimmons, who created it as a fictional data set. So all the observations are generated and do not represent real people.

In this project, we would like to use this data to understand the influence of the parents' background, test preparation, and other factors on a student's performance on high school standardized tests in order to create a model to predict how well students will do in the future.

Additionally, this will help us gain a better understanding as to why admissions officers argued standardized testing isn't a fair measure of admissions.

# Figures
- Figures (of your choosing to help with the narration of your story) with legends (similar to a scientific paper) For reference you search machine learning and your model in google scholar for reference examples.

Using the equations below, we determined the fit of our model by calculating the mispredictions

- Precision = $\frac{TP}{TP + FP}$
- Recall = $\frac{TP}{TP + FN}$
- Accuracy = $\frac{TP + TN}{TP + FP + TN + FN}$
- Mispredictions = $\frac{FP + FN}{TP + FP + TN + FN}$

[todo]
![pairplot](pairplot.png)

![heatmap](heatmap.png)

![comparing model complexity](model_complexity_graph.png)

# Methods
- Methods section (this section will include the exploration results, preprocessing steps, models chosen in the order they were executed. Parameters chosen. Please make sub-sections for every step. i.e Data Exploration, Preprocessing, Model 1, Model 2, (note models can be the same i.e. CNN but different versions of it if they are distinct enough). You can put links here to notebooks and/or code blocks using three ` in markup for displaying code. so it would look like this: ``` MY CODE BLOCK ```
Note: A methods section does not include any why. the reason why will be in the discussion section. This is just a summary of your methods
Results section. This will include the results from the methods listed above (C). You will have figures here about your results as well.
No exploration of results is done here. This is mainly just a summary of your results. The sub-sections will be the same as the sections in your methods section.

## Data Exploration
[link to figures](#figures)
[link to notebook](Project.ipynb)

### Observing Data
In this dataset, we have 1000 total observations. Running the `info()` command, we can see that our data has 1000 rows of non-null data for each column.

The column names of our data are: 
- Gender: student’s gender
- Race/Ethnicity: Represented by the groups defined in the Data Dictionary on National Codes for ethnicity that are as follows:
- Parental Level of Education: The 6 levels of education that the parents of the student could have are:
- Lunch: whether the student received standard or free/reduced lunch, which also provides an indicator of the student’s financial background
- Test preparation course: whether or not the student took and completed a test preparation course for exams
- Math Score: the student's score achieved on the math section of the standardized test
- Reading Score: the student’s score achieved on the reading section of the standardized test
- Writing Score: the student’s score achieved on the writing section of the standardized test

### Plots
We then used ```seaborn``` display all of the attributes in a pairplot, and calculated their correlation coefficients in a heatmap.


## Preprocessing
We preprocessed the data by changing categorical data to numerical or boolean values. For instance, in the case of gender - male was set to true and female to false. This was also done for the other attributes that contained qualitative data, such as lunch, test preparation courses, race/ethnicity, and parental level of education so that we could represent them as qualitative data. For race/ethnicity, we mapped the values by alphabetical order of groups, while for the parental level of education, we set the value from the lowest to the highest degree.

Transforms completed:
- gender
    - male: 1
    - female: 0
- lunch
    - standard: 1
    - free/reduced: 0
- test preparation course
    - completed: 1
    - none: 0
- race/ethnicity
    - Group A (White - British): 0
    - Group B (White - Irish): 1
    - Group C (White - Any other White background): 2
    - Group D (Mixed - White and Black Caribbean): 3
    - Group E (Mixed - White and Black African): 4
    - Group F (Mixed - White and Asian): 5
    - Group G (Mixed - Any other mixed background): 6

- parental level of education
    - some high school: 0
    - high school: 1
    - some college: 2
    - associate's degree: 3
    - bachelor's degree: 4
    - master's degree: 5

### Target
In standardized testing, the final score is reported as the average of the individual sections, thus, we created another column called ```avg score``` that averaged the math, reading, and writing section scores of each student. We then used this average score to define our target, ```passed```. We define pass to be if the student achieves a score higher than 75, which is approximately a letter grade of C.

## First Model
The first model we used was a Naive Bayes Classifier. We first split the data into two parts, the categorical attributes, and the numerical attributes (the test scores). We then scaled the numerical data using `MinMaxScaler()` and split up both parts separately into our training and testing sets with a ratio of 80:20, and average score as the y. We fit a Categorical Naive Bayes to the categorical test data and logged the accuracy, then a Gaussian Naive Bayes for the numerical test data. We then compared the testing and training errors using classification reports and calculations to determine the effectiveness of the models.

## Second Model
The second model we employed was a Neural Net using Keras. First, we split the data into the training and testing sets with a ratio of 80:20. We then used `Sequential()` to initialize the NN. The overall input dimension was 5 and we added 4 layers: one with 16 units and a `relu` activation function, another with 8 units and a `tanh` function, another with 6 units and a `linear` function, and the last sigmoid layer with 1 node. The loss was `binary_crossentropy`, and we ran the model for 10 epochs. We then thresholded the data such that 0.5 and above was considered to be reasonable (or 1), with the rest of the results representing 0. After applying XNOR to the attained `yhat_test` and `y_test` values, we summed the total predictions that were correct and incorrect. To view the degree of accuracy for our model, we printed out a confusion matrix and classification report.

# Discussion

## Data Exploration
We discovered that our dataset does not have any null data, so in our preprocessing, there is no need to drop any null data. 

Since we wanted to see the effect of changing each of the categorical attributes on the scores, we also did not drop any columns in preprocessing.

Some of the initial trends we saw from the pairplot and heatmap were:
- There is a high linear correlation between the test scores of the three different sections: math, reading, and writing
- The scatter plots of test scores with the rest of the attributes show a very high level of clustering, but looking at the correlation coefficients, there appears to be no correlation between the attributes
- There was no correlation between test prep, gender, race/ethnicity, parental level of education, lunch

## Preprocessing
Looking at the correlation values of the heatmap from our data exploration, the relationship between the categorical attributes is not as good as expected, and there were no specific attributes that show a high correlation, we concluded that any linear, polynomial or logistic regression model will not be a good fit. Thus, when creating our model, we plan to proceed with trying a classification model, an SVM, Neural Net, or unsupervised models to best represent our data.

The last part of preprocessing was choosing a target to reflect what we wanted to predict from our model - which was the probability of test success. Thus, because standardized testing reports the final score as the average of the individual sections, we created another column called ```avg score``` that averaged the math, reading, and writing section scores of each student. We then used this average score to define another column ```passed``` to be our target column. 

We define ```passed``` to be if the student achieves a score higher than 75, which is approximately a letter grade of C.

## First Model
We chose to use a Naive Bayes Classifier as our first model. Since our dataset contained both categorical and numerical attributes, we fit a Categorical Naive Bayes Classifier on the categorical attributes only. We then also fit a Gaussian Naive Bayes Classifier on the numerical attributes, but since the numerical attributes are the individual section scores that we directly used to obtain our target attribute, we expected this to be highly accurate - thus, we do our comparison of training and testing error primarily using the Categorical Naive Bayes Classifier.

### Comparing Training vs Testing Error for First Model
We printed classification reports for both the training and testing sets to compare the error for each. From the reports, we concluded that the overall precision and recall for determining whether the individual passed or failed was relatively the same for the testing and training data, with the results for the training data being slightly higher.

Running our model [link to eqn], we get the following values:

Testing Set:
- total positive = 31
- total negative = 200 - 31 = 169

- true positive = 16
- false positive = 31 - 16 = 15

- true negative = 120
- false negative = 169 - 120 = 49

Training Set
- total positive = 113
- total negative = 800 - 113 = 687

- true positive = 65
- false positive = 113 - 65 = 48

- true negative = 493
- false negative = 687 - 493 = 194

Thus, the misprediction for our training set was 0.3025 and the misprediction for our testing set was 0.32. Since the predictive error is similar for both training and testing, according to the fitting graph, our model is likely underfitting or is close to a good fit, with higher predictive error and simple model complexity placing it to the left of the ideal range for model complexity.

[link to figure]

## Second Model
To improve on our first model, we introduce Keras to define a Neural Network. Since none of the categorical attributes showed much correlation, we hoped that increasing the complexity with a neural net and adding layers would help define a more accurate model.

After testing we find the 4 layers separately with ```relu```, ```tanh```, ```linear``` and ```sigmoid``` as activation is expected to have highest accuracy.

### Comparing Training vs Testing Error for Second Model
We generated the classification report for training and testing set to compare the error for each.
By analyzing the report, we noticed that for the Neural Network we have a slightly higher accuracy 0.73 than the first model, and we concluded that the overall precision and recall for determining whether the individual passed or failed was relatively the same for the testing and training data, with the results for the training data being slightly higher.

Running our model [link to eqn], we get the following values:

Testing Set:
- total positive = 15
- total negative = 200 - 15 = 185

- true positive = 8
- false positive = 15 - 8 = 7

- true negative = 138
- false negative = 185 - 138 = 47

Training Set
- total positive = 101
- total negative = 800 - 101 = 699

- true positive = 58
- false positive = 101 - 58 = 43

- true negative = 488
- false negative = 699 - 488 = 211

Thus, the misprediction for our training set was 0.3175 and the misprediction for our testing set was 0.27. Since the predictive error is similar for both training and testing, according to the fitting graph, our model is likely underfitting or is close to a good fit, with higher predictive error and simple model complexity placing it to the left of the ideal range for model complexity.
[link to figure]

# Conclusion
## Summary of Results
We created two models to try and predict whether the student "passed" or failed the standardized test. This metric was created by taking the average of all three scores used and seeing if it was over 75% (our passing value). If it was, the student "passed", else the student failed. The first model we used was a Naive Bayes Classifier with a test accuracy 68% and training accuracy of 70%.

In the process of doing our first method and second method, the accuracy stayed always around 0.7, so a possible future direction is to explore if there’s any better model for our dataset, or add or change some of the hidden layers. 

Since we used avg score instead of separate three scores, we only considered a single measure of success in standardized tests. So in future models, we would like to predict them separately and see if the attributes in our dataset may have different correlations with different scores as well.

Overall, the two models performed well, with accuracies around 0.7, but the second model, using a Neural Net, performed better. One model that we'd like that we try in the future is clustering, which we think may reveal more similarities if we are not bounded by the given class labels.

However, we noticed very little overall correlation in our model, so we cannot conclude that the parents' background, test preparation, and other factors affect a student's performance on high school standardized tests.

It is worth noting that our model was created on a fictional dataset, so our model having no correlation cannot be used to justify why standardized testing is no longer used in college admissions.

# Collaboration
We completed each section of this project with all team members present on a voice call. Live coding does not work very well with running jupyter notebook plots, so we shared screen on one PC and all contributed code and ideas.

Sophie was the team leader of our group, gathering all members of the group to meet and facilitating communications, as well as directing the focus of the group and leading coding sessions.

