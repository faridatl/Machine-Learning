# Machine-Learning

🚀 Machine Learning Projects Repository

Welcome to my Machine Learning Projects repository! I'm Faridat, a data enthusiast with a passion for building intelligent systems. In this collection, you'll find a showcase of my hands-on experience in supervised learning, including classic algorithms and cutting-edge techniques.

🔍 Projects Overview:

## Naive Bayes:
The Breast Cancer Wisconsin Dataset from the UCI machine learning respository is a study of classification of 568 patients that are labelled as either (M) Malignant of (B)Benign based on 30 attributes/features computed from a digitized image of a fine needle aspirate of a breast mass. The Naive Bayes Classifier is a supervised learning algorithm that will be used to classify & test the performance of this model. 70% of the data will be reserved for training and the remaining 30% will be used for testing. For model evaluation a Confusion Matrix was created to determine models sensitivity and specifity. Overall model accuracy was 95.3% - model performed exceptoinally well at classifying which patients were Malignant or Benign.

## K-Nearest Neighbors (KNN):
The Wine Dataset from the UCI machine learning respository consists of the results of a chemical analysis of wines grown within the same region in Italy but derived from three different cultivers. The analysis determined 13 constituents (attributes) found in the three types of wine. The K-Nearest Neighbors Classifier is a supervised learning algorithm that will be used to classify and test the perfomance of this model. 80% of the data will be reserved for training and the remaining 20% will be used for testing. Features were standardized/scaled in order to prevent the dominance of one feature over another as well as providing all features with a consistent scale. This model evalution differs from a normal binary classification becasue instead of two classes we are now working with three. This required some adjustments to the Confusion Matrix calculation. Random state was also taken into consideration. Overall model accuracy with a random state of 42 was 94.4% but as noted on code when a random state of 73 was used model accuaracy improved to 100%.

## Decision Tree & Random Forest:
The Ionosphere Dataset from the UCI machine learning respository consists of a phased array of 16 high-frequency antennas with a total transmitted power on the order of 6.4 kilowatts. Received signals were processed using an autocorrelation function whose arguments are the time of a pulse and the pulse number. There were 17 pulse numbers for the system. Instances in this database are described by 2 attributes per pulse number, corresponding to the complex values returned by the function resulting from the complex electromagnetic signal. "Good" (g) radar returns are those showing evidence of some type of structure in the ionosphere. "Bad" (b) returns are those that do not; their signals pass through the ionosphere. This will be a binary classificstion.

Both the Decition Tree and Random Forest Classifier willl be employed to evaluate the classification performance. Both classifiers will be compared to determine which performed the best by recording the results of their Confusion matrix, sensitivity, specifity, total accuracy, F1-score, ROC and AUC scores. 5-Fold cross validation will also be employed to determine if the both models better.

[Highlight the Random Forest project, showcasing how it was utilized and its impact on predictive accuracy.]
Decision Trees:
[Provide insights into the Decision Trees project, discussing the context and any interesting findings.]

## Linear Discriminant Analysis(LDA) & Quadratic Discriminant Analysis(QDA):
The Iris Species dataset includes three types of iris species with 50 samples each as well as some properties about each flower. Therer are 4 features/attributes and three classes. A 5-Fold cross-validation to evaluate the classification performance of a Linear Discriminant Analysis and Quadratic Discriminant Analysis classifier will be carried out. The best classifier will be determined based on model performance as well as taking into account each classifiers Confusion Matrix, sensitivity, specificity, total accuracy, F1-score, ROC & AUC curve, and overall model performance.

🛠️ Technologies Used:

Python: Leveraging the power of Pandas, NumPy, and Scikit-Learn.
Jupyter Notebooks: Transparent and interactive documentation of the entire workflow.
Sublime Text: Efficient, versatile code editor with minimalist interface.

📈 Results:

[Include any performance metrics, accuracy scores, or visualizations that showcase the success of your projects.]
🔗 How to Use:

Each project is organized into separate directories. Feel free to reach out if you have questions, suggestions, or if you're interested in collaboration.

🌐 Connect with Me:

LinkedIn: [
](https://www.linkedin.com/in/faridatlawal/)

#### I'm continuously learning and expanding my skill set. Join me on this exciting journey through the world of machine learning! 🤖✨
