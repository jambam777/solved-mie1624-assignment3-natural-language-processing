Download Link: https://assignmentchef.com/product/solved-mie1624-assignment3-natural-language-processing
<br>
<strong>Sentiment Analysis </strong>is a branch of Natural Language Processing (NLP) that allows us to determine algorithmically whether a statement or document is “positive” or “negative”.

Sentiment analysis is a technology of increasing importance in the modern society as it allows individuals and organizations to detect trends in public opinion by analyzing social media content. Keeping abreast of socio-political developments is especially important during periods of policy shifts such as election years, when both electoral candidates and companies can benefit from sentiment analysis by making appropriate changes to their campaigning and business strategies respectively.

The purpose of this assignment is to compute the sentiment of text information – in our case, tweets posted recently on Canadian Elections – and answer the research question: <strong><em>“What can public opinion on Twitter tell us about the Canadian political landscape in 2019?” </em></strong>The goal is to essentially use sentiment analysis on Twitter data to get insight into the Canadian Elections

Central to sentiment analysis are techniques first developed in text mining. Some of those techniques require a large collection of classified text data often divided into two types of data, a training data set and a testing data set. The training data set is further divided into data used solely for the purpose of building the model and data used for validating the model. The process of building a model is iterative, with the model being successively refined until an acceptable performance is achieved. The model is then used on the testing data in order to calculate its performance characteristics.

<ul>

 <li><strong>Produce a report in the form of an IPython notebook detailing the analysis you performed to answer the research question. Your analysis must include the following steps: data cleaning, exploratory analysis, model preparation, model implementation, and discussion. This is an open-ended problem: there are countless different ways to approach each part of the analysis and therefore the motivation for each step is just as important as its implementation. When writing the report, make sure to explain (for each step) what it is doing, why it is important, and the pros and cons of that approach.</strong></li>

 <li><strong>Create 5 slides in PowerPoint and PDF describing the findings from exploratory analysis, model feature importance, model results and visualizations.</strong></li>

</ul>

Two sets of data are used for this assignment. The <em>sentiemnt_analysis.csv </em>file contains tweets that have had their sentiments already analyzed and recorded as binary values 0 (negative) and 1 (positive). Each line is a single tweet, which may contain multiple sentences despite their brevity. The comma-separated fields of each line are:

<ul>

 <li>ID Tweet ID</li>

 <li>text the text of the tweet</li>

 <li>label the polarity of each tweet (0 = negative sentiment, 1 = positive sentiment)</li>

</ul>

The second data set, <em>Canadian_elections_2019.csv </em>contains a list of tweets regarding the 2019 Canadian PM elections. The fields of each line are:

<ul>

 <li>text the text of the tweet</li>

 <li>sentiment can be “positive” or “negative”</li>

 <li>negative_reason reason for negative tweets. NaN for positive tweets</li>

</ul>

Both datasets have been collected directly from the web, so they may contain html tags, hashtags, and user tags.

<strong>Learning objectives:</strong>

<ol>

 <li>Implement functionality to parse and clean data according to given requirements.</li>

 <li>Understand how exploring the data by creating visualizations leads to a deeper understanding of the data.</li>

 <li>Learn about training and testing machine learning algorithms (logistic regression, k-NN, decision trees, random forest, XGBoost, etc.).</li>

 <li>Understand how to apply machine learning algorithms to the task of text classification.</li>

 <li>Improve on skills and competencies required to collate and present domain specific, evidence-based insights.</li>

</ol>

<strong>To do:</strong>

<ol>

 <li><strong>Data cleaning</strong></li>

</ol>

The tweets, as given, are not in a form amenable to analysis –– there is too much ‘noise’.

Therefore, the first step is to “clean” the data. Design a procedure that prepares the

Twitter data for analysis by satisfying the requirements below. o     All html tags and attributes (i.e., /&lt;[^&gt;]+&gt;/) are removed. o    Html character codes (i.e., &amp;…;) are replaced with an ASCII equivalent. o          All URLs are removed. o        All characters in the text are in lowercase. o          All stop words are removed. Be clear in what you consider as a stop word. o      If a tweet is empty after pre-processing, it should be preserved as such.

<ol start="2">

 <li><strong>Exploratory analysis </strong>

  <ul>

   <li>Design a simple procedure that determines the political party (Liberal, Conservatives or New Democratic Party (NDC)) of a given tweet and apply this procedure to all the tweets in the Canadian Elections dataset. A suggestion would be to look at relevant words and hashtags in the tweets that identify to certain political parties or candidates. What can you say about the distribution of the political affiliations of the tweets?</li>

   <li>Present a graphical figure (e.g. chart, graph, histogram, boxplot, word cloud, etc.) that visualizes some aspect of the generic tweets in <em>csv </em>and another figure for the 2019 Canadian Elections tweets. All graphs and plots should be readable and have all axes that are appropriately labelled.</li>

  </ul></li>

 <li><strong>Model preparation </strong></li>

</ol>

Split the generic tweets randomly into training data (70%) and test data (30%).

Prepare the data to try seven classification algorithms — logistic regression, k-NN, Naive Bayes, SVM, decision trees, Random Forest and XGBoost, where each tweet is considered a single observation/example. In these models, the target variable is the sentiment value, which is either positive or negative. Try two different types of features, Bag of Words (word frequency) and TF-IDF on all 7 models. (<em>Hint: Be careful about</em>

<em>when to split the dataset into training and testing set.</em>)

<ol start="4">

 <li><strong>Model implementation and tuning </strong></li>

</ol>

Train models on the training data from generic tweets and apply the model to the test data to obtain an accuracy value. Evaluate the same trained model with best performance on the Canadian Elections data. How well do your predictions match the sentiment labelled in the Canadian elections data?

Choose the model that has the best performance       and visualize the sentiment prediction results and the true sentiment for each of the 3 parties/candidates. Discuss whether NLP analytics based on tweets is useful for political parties during election campaigns.

Split the <strong>negative </strong>Canadian elections tweets into training data (70%) and test data

(30%). <strong>Use the true sentiment labels in the Canadian elections data instead of your predictions from the previous part. </strong>Choose three algorithms from classification algorithms (choose any 3 from logistic regression, k-NN, Naive Bayes, SVM, decision trees, ensembles (RF, XGBoost)), train multi-class classification models to predict the reason for the negative tweets. Tune the hyperparameters and chose the model with best score to test your prediction reason for negative sentiment tweets. There are 5 different negative reasons labelled in the dataset.

Feel free to combine similar reasons into fewer categories as long as you justify your reasoning. You are free to define input features of your model using word frequency analysis or other techniques.

<ol start="5">

 <li><strong>5<u>. </u></strong><strong>Results </strong></li>

</ol>

Answer the research question stated above based on the outputs of your first model. Describe the results of the analysis and discuss your interpretation of the results. Explain how each party is viewed in the public eye based on the sentiment value. For the second model, based on the model that worked best, provide a few reasons why your model may fail to predict the correct negative reasons. Back up your reasoning with examples from the test sets. For both models, suggest one way you can improve the accuracy of your models.

<strong>The order laid out here does not need to be strictly followed. Significant marks of each section are allocated to discussion. Use markdown cells as needed to explain your reasoning for the steps that you take.</strong>

<strong>Bonus:</strong>

We will give up to 10% bonus marks for innovative work going substantially beyond the minimal requirements. These marks can make up for marks lost in other sections of the assignment, but your overall mark for this assignment cannot exceed 100%. The obtainable bonus marks will depend on the complexity of the undertaking and are at the discretion of the marker. Importantly, your bonus work should not affect our ability to mark the main body of an assignment in any way. Any bonus work should be explicitly labelled as “Bonus” in its own section. You may decide to pursue any number of tasks of your own design related to this assignment, although you should consult with the TA before embarking on such exploration.

Certainly, the rest of the assignment takes higher priority. Some ideas:

<ul>

 <li>Try word embeddings (<a href="https://en.wikipedia.org/wiki/Word_embedding">https://en.wikipedia.org/wiki/Word_embedding</a>) and N-grams as feature engineering techniques in addition to WF and TF-IDF.</li>

 <li>Explore Deep Learning algorithms and compare their performance to that of your best performing classification model.</li>

 <li>Hyperparameter tuning for the models</li>

 <li>While the exploratory analysis section requires only two figures, you can explore the data further. You can also display the results of the model visually.</li>

</ul>