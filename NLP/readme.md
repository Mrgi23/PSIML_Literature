# NLP workshop

The main goal of this workshop is to give an overview of the basic concepts of **natural language processing (NLP)**, **recurrent neural networks (RNNs)** and **transformers**. As it introduces all these concepts, the main emphasis is on the implementation. The **sentiment classification** of tweets is taken as an example task to demonstrate the end-to-end implementation of the machine learning models commonly used natural language processing.

Understanding the supervised learning paradigm, foundational components of neural networks and basics of *PyTorch* is needed to follow this workshop.

It is important to note that the material shown in this workshop has purely demonstrative purpose, which is why some steps might be simplified or omitted. The examples include reducing the size of the dataset to speed-up the training or lack of hyperparameter tuning. It is strongly recommended to exercise and modify the code, to explore other options, starting from the choice of hyperparameters to the models' architecure.

## 1. Neccesary installs

The list of *Python* packages used in this workshop is the following:

1) **pandas** - *conda install pandas*
2) **nltk** - *pip install nltk*
3) **matplotlib** - *pip install matplotlib*
4) **sklearn** - *pip install scikit-learn*
5) **scipy** - *pip install scipy*
6) **torch** - *conda install pytorch torchvision -c pytorch*

In order to install all required packages and create conda environment, run <code>setup.cmd</code> script.

## 2. Table of Contents

The notebooks for this workshop should be used in the following order:

1) **SentimentAnalysisIntroduction.ipynb** introduces the sentiment analysis problem that is going to be solved and the *sentiment140* dataset used for this task. It covers the basic preprocessing done on the input data (class **TextPreprocessor**) and dataset split into training, validation and test partition (class **SplitDataset**).

2) **SentimentAnalysisTraining.ipynb** notebook illustrates how to train a simple model for sentiment classification of tweets. It covers:
   - how to represent textual data in the numerical format (classes **Vocabulary**, **TwitterVectorizer** and its subclass **TwitterOneHotVectorizer**),
        - The **Vocabulary** class encapsulates the mapping between token and its index in the vocabulary. It can be built from scratch using some dataset or loaded from a file. Once initialized, the **Vocabulary** class has two main methods:

            <code>find_token(token): int</code> - returns index in the vocabulary for the input token

            <code>find_index(index): str</code> - returns the token that corresponds to the input index

        - The **TwitterVectorizer** encapsulates the conversion from input text sequence to numerical vector. This is an abstract class, whose main method in implemented in the subclasses:

            <code>vectorize(text, vector_length): np.ndarray</code> - creates a vector representation for the text

        - The **TwitterOneHotVectorizer** implements abstract **TwitterVectorizer** class. It represents input text sequence as one-hot encoding vector. It implements the <code>vectorize(text)</code> method that returns the collapsed one-hot encoding.

   - how to reprepresent dataset in that format (class **TwitterDataset**),
        - The **TwitterDataset** class implements PyTorch dataset class. It is an iterator through data points. Its main method is:

            <code>getitem(index): dict</code> - takes an index of a data point as input, and returns a dict of the data point's features (x_data) and label (y_target)

   - how to load that data for training (method in **TwitterDataLoader**),
        - Method <code>generate_batches</code> from the **TwitterDataLoader**, wraps the PyTorch DataLoader. It groups individual data points into mini-batches.

   - how to build a simple perceptron model for sentiment classification (class **ModelPerceptron**),
        - The **ModelPerceptron** encapsulates the architecture of a simple classification model with one linear layer.

   - how to implement the training loop (class **Trainer**).
        - The **Trainer** class is initialized with previously defined components: dataset, model, loss function and optimizer and it has two main methods: 

            <code>train(num_epochs, batch_size, device): void</code> - trains the model on the training set for the chosen number of epochs - executes the <code>train_epoch</code> method in a loop. This <code>train_epoch</code> method iterates over the batches to train the model and executes the following 5 steps for each batch:

                    1. Zero the gradients (clear the information about gradients from previous step)
                    2. Calculate the model output
                    3. Compute the loss, when compared with labels
                    4. Use the loss to calculate and backpropagate gradients
                    5. Use optimize to update weights of the model

            <code>evaluate(batch_size, device, split): void</code> - evaluates the model performance on the chosen dataset split</code>

    This notebook is the base for other steps in the workshop and it is important to understand it well before proceeding further with the workshop. We will cover only the key parts during the workshop but it is strongly recommended to take more detailed look as a <span style="color:orange">homework task</span>.

3) **SentimentAnalysisEvaluation.ipynb** notebook tackles with the evaluation of the sentiment classification model built in the previous step (SentimentAnalysisTraining.ipynb notebook).

    It repeats steps introduced in the previous notebook: definition of dataset, model, loss function and optimizer, and encapsulates training loop into the separate class (class **Trainer**). It further covers the evaluation of the trained model, including inference on a new data and evaluation on some held-out portion of the data (test set). The setting used in this notebook (initialization, model training, evaluation) is used in all further examples during the workshop.

4) <span style="color:orange">[Homework task]</span> **SentimentAnalysisMLP.ipynb** notebook showcases what happens if the simple perception model is replaced with more complex model (class **ModelMLP**). Since the only difference is in model architecture, this example will not be covered during the workshop and is recommended as a homework task.

5) **WordEmbeddings.ipynb** notebook introduces the concepts of word embeddings and demonstrates how to use pretrained word embeddings like GloVe (class **PreTrainedEmbeddings**).

    **Pretrained word embeddings** are trained on a large corpus of data and are available freely to download and use. There are many different varieties available, from the original *Word2Vec* to Stanford's *GloVe*, Facebook's *FastText*, and many others. Typically. embeddings come in a file with a following format:

    **dog** &nbsp;&nbsp;&nbsp;&nbsp; -1.231 -0.430 0.573 0.500 ... <br/>
    **cat** &nbsp;&nbsp;&nbsp;&nbsp; -0.947 0.637 -0.133 0.784 ...

    To efficiently load and process embeddings, we use utility class **PreTrainedEmbeddings**, that builds an in-memory index of all the word vector. It also enables quick lookups and nearest neighbor queries. We will not cover details of this class in the workshop, but we strongly recommend it as a <span style="color:orange">homework task</span>. Its key methods are:

    <code>get_embedding(word)</code> - returns an embedding vector for the given word

    <code>get_closest_to_vector(vector, n=1)</code> - returns <code>n</code> words nearest to the given vector

    The analogy task is chosen to showcase some properties of word embeddings. 

    This notebook concludes with the introduction of a new method of representing words in numerical format (classes **SequenceVocabulary** and **TwitterSequenceVectorizer**). **SequenceVocabulary** is a variant of a **Vocabulary** class, that bundles several tokens that are important for modeling sequences. The **TwitterSequenceVectorizer** demonstrates how to use this class.

6) **SentimentAnalysisMLPWithEmbeddings.ipynb** notebook explains how to use word embedding representation as an input to the multi-layer perceptron network for sentiment classification.

    It introduces the <code>nn.Embedding</code> layer, a PyTorch module that encapsulates an embedding matrix. That is followed by the description of the sentiment classification model (class **ModelMLPWithEmbeddings**), that covers two possible initializations of the <code>Embedding</code> layer: with and without pre-trained embedding vectors as initial values. During the workshop, we cover the case with initialized pre-trained *GloVe* embeddings, while the case when word embeddings are learnt from scratch is recommended as a <span style="color:orange">homework task</span>.

    The class **ModelMLPWithEmbeddings** covers the MLP model for sentiment classification. Input data is converted to embedding vectors as a result of <code>nn.Embedding</code> layer, followed by an aggregation function that creates one vector for the sequence, and couple of fully-connected layers.

    <span style="color:red">[Workshop task] Implement the forward method in the class **ModelMLPWithEmbeddings**, that executes the forward pass for the classifier model. </span>

    The rest of the notebook has standard flow: initialization of dataset, model, loss function and optimizer, model training and finally, evaluation.

7) **SentimentAnalysisElmanRNN.ipynb** notebook introduces the main concepts of recurrent neural networks and how they can be applied for the sentiment classification task.

    Firstly, it depicts the architecture of the *Elman RNN* module, that represents the base (vanilla) version of the RNN (class **ElmanRNN**). The PyTorch <code>nn.RNN</code> class implements the Elman RNN. We can use it directly, or we can construct our version, to showcase the RNN computations explicitly (class **ElmanRNN**). We will not cover the details of this class during the workshop, but we strongly recommend it as a <span style="color:orange">homework task</span>.

    Further, we showcase how to combine <code>nn.Embedding</code> layer, the <code>ElmanRNN</code> module and <code>Linear</code> layers into a model for sentiment classification of tweets (class **ModelElmanRNN**). While this workshop covers only the *Elman RNN* architecture, it is strongly recommended to explore what happens if it is replaced with some other RNN module such as *GRU* or *LSTM*, as a <span style="color:orange">homework task</span>. The only difference would be in the RNN initialization, while the rest of code remains unchanged.

    The rest of the notebook has standard flow: initialization of dataset, model, loss function and optimizer, model training and finally, evaluation.

8) <span style="color:orange">[Homework task]</span> **SentimentAnalysisTransformer.ipynb** notebook introduces how Transformer Encoder can be leveraged for the text classification task. The implementation is very similar to the recurrent neural network, while the key difference is using <code>nn.TransformerEncoder</code> layer instead of the <code>nn.RNN</code> layer. Therefore, we won't cover this task in detail in the workshop, but we strongly recommend it as a homework task.

9) **SentimentAnalysisBERT.ipynb** notebook concludes this workshop with the description of BERT model, a bidirectional model that is based on transformer architecture.

    It explains how to prepare the dataset (class **BERTDataLoader**), load the pre-trained BERT model and use it to produce the output (class **BERTPredictor**) for the sentiment classification task. The pre-trained BERT model is used from the standard implementation from [Hugging Face transformer library](https://huggingface.co/transformers/model_doc/bert.html), while the classification head is initialized with random weights.

    While the notebook does not cover how to fine-tune the BERT model for this task, it is highly recommended to explore this option, as a <span style="color:orange">homework task</span>.

    <span style="color:red">[Workshop task] Implement the **predict** method in the file **BERTPredictor**, that executes the trained model on the example text.</span>

    This method is executed in the very end of the notebook, similarly to other examples.

## 3. Tasks

This workshop has several tasks, to fill-in the missing code. The tasks are:

1) Implement <code>forward(x, apply_softmax)</code> method in the class **ModelMLPWithEmbeddings**. It should execute the forward pass for the classifier model based on the multi-layer perceptron network.

2) Implement <code>predict(text, model, tokenizer)</code> method in the file **BERTPredictor**, that executes the trained model on the example text. It should return the predicted sentiment for the input tweet.

## 4. References

1) Andrew Ng, "Deep Learning Specialization", Coursera

2) Delip Rao & Brian McMahan, "Natural Language Processing with PyTorch", O'Reilly Media Inc., 2019.
