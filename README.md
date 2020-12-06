# Projects from CSE 4502 - Big Data Analytics

Textbooks **"End-to-End Machine Learning Project" of *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems* by Aurelien Geron** 2nd Edition


### ML Python packages used
- Scikit-learn
- Tensorflow-GPU (for GPU acceleration)
- Keras

### Project Synposes
1. **Boston Housting** - Following Chapter 2 of Textbook
   - Working with `sklearn.datasets.load_boston`
   - Pulled various specs from datasets and used `matplotlib` to plot histograms of data to understand trends
   - Scatter Matrix, Confusion Matrix
   - Implemented custom transformers
   - Trained and evaluated using `sklearn.linear_model.LinearRegression`
   - First look at Grid Search and Random Search for fine tuning hyperparameters
2. **Fashion MNIST** - Objective is to build a classifier for Fashion-MNIST dataset that achieves over 85% accuracy. Final accuracy = ~84% accuracy 
  - Single Class Classification
    - First time evoking StratifiedKFold train/test set split
    - Explored idea of a naive classifier "Never 5", where 5 is the index value for sandals. If the classifier predicted over the set of only sandals (can be applicable for any category of clothes), then it'll perform at 90% accuracy since sandals is only 1/10 of the categories of clothes in the dataset
    - Compared performance between SGD and Random Forest Classifiers using Precision vs. Recall Plots and ROC Curves
  - Multi Class Classification
    - Utilized `sklearn.svc.SVM` and `sklearn.preprocessing.StandardScaler`
    - Error analysis done through confusion matrix visualizations
3. **Fashion MNIST Part 2 & Letter Recognition** - First part is to use the same Fashion MNIST dataset but combine various classifiers into an ensemble that outperforms all of the classifiers. Second part is to use do letter classification and produce a classifier using any methods learned up to this point.
  - Part 1 - Fashion MNIST
    - Blending RandomForestClassifier, LinearSVC, ExtraTreesClassifier, MLPClassifier
    - Each performed at around 85% accurracy with LinearSVC performing the worst at 73%
    - Voting with `voting='soft'` performs better, uses argmax of sums of predicted probabilities. Accuracy = 87.45%
    - Stacking Ensemble using RandomForestClassifier with `oob_score=True` did not outperform the Voting Classifier
  - Part 2 - Letter Recognition
    - Tried RandomForestClassifier, ExtraTreesClassifier, SVM, MLP, SVC_OVO "One-Versus-One"
    - Used Voting Classifier to determine the possibility of a higher accuracy
4. **Fashion MINST Part 3** - Training custom MLP Classifier with Keras to see if it outperforms Programming Assignment 2. Train another classifier using the dimensionality reduced dataset, observe differences in training time, and accuracy. 
   - Part 1 - MLP using Keras
     - Played around with various keras Dense layers with different quantity of layers and neurons, with output being 10 neuron layer with `activation="softmax"`
     - MLP produced after 3 trials was found to outperform the RandomForestClassifier by 2-3%
       - 4 dense layers, 100 > 300 > 100 > 10, `activation="relu"` excluding the output layer. Compiled using  `loss="sparse_categorical_crossentropy", 
                       optimizer="sgd", 
                       metrics=["accuracy"]`
     - Hyperparameters fine tuned using `RandomizedSearchCV` , using the PCA reduced dataset will make the hyperparameter search less time consuming
     - loss, accuracy, validation loss, and validation accuracy were plotted to see training and validation behavior and trends
   - Part 2 - PCA reduction of the dataset
     - Reduced dataset to be 95% of its explained variance, dimensions = 188
     - Training time was significantly faster (86 sec vs. 136 sec), 1.58x faster, for a slight trade off in performance (87.26% vs. 87.9%)
   - Part 3 - Improvements
     - Improve MLP Neural Network with RandomizedSearchCV using the reduced data
     - Optimized learning rate `learning_rate`, number of hidden layers `n_hidden`, and number of neurons `n_neurons`
   - Final model accuracy = 88.33%
5. **Fashion MNIST Part 4** - Building a classifier performing over 94%, while learning to use convnet, Batch Normalization, Dropout to fight overfitting, and nadam optimizer in the process
   - Played around with quantity and types of layers such as Conv2D, the padding, BatchNormalization, MaxPooling2D, Dropout(%) 
   - Plotted training loss to validation loss to observe trends of overfitting to training set
   - Final model still had some overfitting issues that could not be ironed out between the various new methods with 
   - Final accuracy = 94.18%
     - Model compiled with `model.compile(optimizer='nadam',
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])`
6. **Celebrity Male and Female Faces** - Using Large-scale CelebFaces Attributes (CelebA) to continue explore Convolutional Neural Networks (CNN)
   - Modified and tried new things in convnet for new application (larger resolution color images)
   - Various iterations of model saved as h5 files for comparison
   - Model trained only 2000 photos, 1000 for validation, 1000 for testing, 100 epochs max. Followed Chapter 5.2 of Textbook with attention to the use of data augmentation and dropout to fight overfitting
     - Model was successful in fighting overfitting validation accuracy = ~93% and validation loss = ~20% +/- 10% towards the end of training
   - Loaded VGG16 as CNN base, experimented with additions of various amounts, types, and sizes of layers 
     - quantity of trainable weights in conv base (VGG16) frozen to avoid adjustable weights of preset conv base
   - Evidence of overfitting is still evidence when plotting training accuracy and validation accuracy 
   - Final accuracy = 94.3%
7. **IMDB Positive & Negative Reviews** - IMDB movie reviews classified as either a positive or negative review. Pre-trained GloVe word embeddings used to convert text into tensors. Word "value" evaluated by a list of positive words separated from a list of negative words. Compare GloVe embedding, with dense embedding and LSTM embedding. Best embedding is chosen built as the classifier of choice to try to obtain >90% accuracy. 
   - IDMB review dataset pulled from Stanford. Data tokenized, words are vectorized. Classifier will learn to classify the movie reviews after just 200 examples.
   - L2 norm used to relate "semantic" distance of associated words
   - GloVe "Global Vectors for Word Representation" developed by Stanford researchers in 2014. Embedding technique based on factorizing a matrix of word co-occurrence statistics. Precomputed embedding processed for use. 
   - distance of 2 words compared using GloVe, Dense, LSTM embeddings. Function written to parse words where its neighbor changes in 1 type of embedding but another 
   - All 3 embeddings visualized with how positive and negative words are spreaded throughout 2D space. 
   - Multiple models saved, best model reloaded and tested
   - Final accuracy = 90.61%
8. **COVID Testing Kit - Produce a classifier that has 100% recall but 70% precision. Idea is to focus on having perfect precision so as to not miss a single positive COVID case, but only 70% recall such that it is okay to have false positives. Fake data was generated for this exercise.
   - Data taken in from .csv, parsed, plotted. Numerical vs. categorical data prepared by pipelining, categorical is passed through OneHotEncoder. Data points split by training, validation, testing (5000, 2500, 2500)
   - RandomForestClassifier used to report the importance of each attribute
   - Trained ExtraTreesClf, SVC, RandomForestClf into Voting Classifier. ROC curves, precision vs. recall  plotted, ROC AUC scores calculated. Using recall = 1, found the precision values to find highest performing model.
   - SVC was best performing model. Best hyperparameters found with GridSearchCV with large area of search. 
     - At Recall = 1 = 100%, Precision = 71.85%
   - Applied to test set (2500 data points) Final precision = 70.72% at recall = 100%
9. **Celeb - Young Attractive Male Faces** - Using Celebrity Faces dataset, and 10,000 photos, train a classifier that can classify photos using 4 labels (young, attractive, smiling, male). 
   - Using VGG16 base, additional modifications made, namely Dropout layers to fight overfitting. 
   - Successful model saved as `celeb_ymas_2.h5`
   - Final accuracy = 87.52%

