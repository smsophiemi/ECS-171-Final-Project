# ECS-171-Final-Project

## Group Members
Pengcheng Cao, Zahira Ghazali, Denise Kwong, Sophie Mi, Lingfeng Pan, Zihan Wang

# Introduction

## Abstract
The dataset we are using records the standardized test scores of high school students in the United States in various subjects, and some personal identification details. (https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)

# NOTE: the data was created by Royce Kimmons, who created it as a fictional data set. So all the observations are generated and do not represent real people.

In this project, we would like to use this data to understand the influence of the parents' background, test preparation, and other factors on a student's performance on high school standardized tests in order to create a model to predict how well students will do in the future. 

Additionally, this will help us gain a better understanding as to why admissions officers argued standardized testing isn't a fair measure of admissions.

## Summary
[todo]

# Figures
- Figures (of your choosing to help with the narration of your story) with legends (similar to a scientific paper) For reference you search machine learning and your model in google scholar for reference examples.
[todo]

# Methods
- Methods section (this section will include the exploration results, preprocessing steps, models chosen in the order they were executed. Parameters chosen. Please make sub-sections for every step. i.e Data Exploration, Preprocessing, Model 1, Model 2, (note models can be the same i.e. CNN but different versions of it if they are distinct enough). You can put links here to notebooks and/or code blocks using three ` in markup for displaying code. so it would look like this: ``` MY CODE BLOCK ```
Note: A methods section does not include any why. the reason why will be in the discussion section. This is just a summary of your methods
Results section. This will include the results from the methods listed above (C). You will have figures here about your results as well.
No exploration of results is done here. This is mainly just a summary of your results. The sub-sections will be the same as the sections in your methods section.

## Data Exploration

### Observing Data
In this dataset, we have 1000 total observations. Running the `info()` command, we can see that our data has 1000 rows of non-null data for each column, so in our preprocessing, there is no need to drop any null data. We also did not drop any columns, as we wanted to use each factor to test whether they had any influence on the test scores.

The column names of our data are: 
- Gender: student’s gender
- Race/Ethnicity: Represented by the groups defined in the Data Dictionary on National Codes for ethnicity that are as follows:
    - Group A - White - British
    - Group B - White - Irish
    - Group C - White - Any other White background
    - Group D - Mixed - White and Black Caribbean
    - Group E - Mixed - White and Black African
    - Group F - Mixed - White and Asian
    - Group G - Mixed - Any other mixed background
- Parental Level of Education: The 6 levels of education that the parents of the student could have are:
    - some high school
    - high school
    - some college
    - associate's degree
    - bachelor's degree
    - master's degree
- Lunch: whether the student received standard or free/reduced lunch, which also provides an indicator of the student’s financial background
- Test preparation course: whether or not the student took and completed a test preparation course for exams
- Math Score: the student's score achieved on the math section of the standardized test
- Reading Score: the student’s score achieved on the reading section of the standardized test
- Writing Score: the student’s score achieved on the writing section of the standardized test


### Plots
Plotting all of our data in a pairplot, and calculating the correlation for all the attributes, some of the initial trends we saw were:
- There is a high linear correlation between the test scores of the three different sections: math, reading, and writing
- The scatter plots of test scores with the rest of the attributes show a very level of clustering, but looking at the correlation coefficients, there appears to be no correlation between the attributes
- There was no correlation between test prep, gender, race/ethnicity, parental level of education, lunch

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
    - Group A: 0
    - Group B: 1
    - Group C: 2
    - Group D: 3
    - Group E: 4
    - Group F: 5
    - Group G: 6
- parental level of education
    - some high school: 0
    - high school: 1
    - some college: 2
    - associate's degree: 3
    - bachelor's degree: 4
    - master's degree: 5

After preprocessing our data, we concluded that any linear, polynomial and logistic regression model will not be a good fit, thus, when creating our model, we plan to proceed with trying a classification model, as well as both SVM and neural net to best represent our data.

### Target
In standardized testing, the final score is reported as the average of the individual sections, thus, we created another column called ```avg score``` that averaged the math, reading, and writing section scores of each student. We then used this average score to define our target, ```passed```. We define pass to be if the student achieves a score higher than 75, which is approximately a letter grade of C.

## First Model
We chose to use a Naive Bayes Classifier as our first model. Since our dataset contained both categorical and numerical attributes, we fit a Categorical Naive Bayes Classifier on the categorical attributes only. We then also fit a Gaussian Naive Bayes Classifier on the numerical attributes, but since the numerical attributes are the individual section scores that we directly used to obtain our target attribute, we expected this to be highly accurate - thus, we do our comparison of training and testing error primarily using the Categorical Naive Bayes Classifier.

## Second Model

# Discussion
- Discussion section: This is where you will discuss the why, and your interpretation and your though process from beginning to end. This will mimic the sections you have created in your methods section as well as new sections you feel you need to create. You can also discuss how believable your results are at each step. You can discuss any short comings. It's ok to criticize as this shows your intellectual merit, as to how you are thinking about things scientifically and how you are able to correctly scrutinize things and find short comings. In science we never really find the perfect solution, especially since we know something will probably come up int he future (i.e. donkeys) and mess everything up. If you do it's probably a unicorn or the data and model you chose are just perfect for each other!


### Comparing Training vs Testing Error for First Model
We printed classification reports for both the training and testing set to compare the error for each. From the reports, we concluded that the overall precision and recall for determining whether the individual passed or failed was relatively the same for the testing and training data, with the results for the training data being slightly higher. 

Using the equations below, we determined the fit of our model by calculating the mispredictions for the training and testing data of the Categorical Model:

- Precision = $\frac{TP}{TP + FP}$
- Recall = $\frac{TP}{TP + FN}$
- Accuracy = $\frac{TP + TN}{TP + FP + TN + FN}$
- Mispredictions = $\frac{FP + FN}{TP + FP + TN + FN}$

Running our model, we get the following values:

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


# Conclusion
- Conclusion section: This is where you do a mind dump on your opinions and possible future directions. Basically what you wish you could have done differently. Here you close with final thoughts
Collaboration section: This is a statement of contribution by each member. This will be taken into consideration when making the final grade for each member in the group. Did you work as a team? was there a team leader? project manager? coding? writer? etc.

