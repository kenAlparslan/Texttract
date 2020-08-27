<img src="https://camo.githubusercontent.com/ed508e9c66d718f76333215a139af24f8bb8fa8d/68747470733a2f2f6d75736573636f72652e6f72672f73697465732f6d75736573636f72652e6f72672f66696c65732f4361707475726525323064253237652543432538316372616e253230323031362d30332d303125323030392e34382e31315f302e706e67" align="center"/>

## Google Summer of Code 2020 Final Work Product - 

## What problems we are tackling?

AWS TextTract solves a couple of big problems. It does Optical Character Recognition (OCR) to extract raw text from image. Our goal is to use open source community’s resources and build (or at least try to come close to AWS TextTract). By keeping this in mind, first , I need to handle the OCR subtask. I decide to use Python-tesseract which is an optical character recognition (OCR) tool for python. “That is, it will recognize and “read” the text embedded in images. Python-tesseract is a wrapper for Google’s Tesseract-OCR Engine.[ https://github.com/tesseract-ocr/tesseract] “[1] It is a machine learning technique that depends on feature extraction from a large dataset of documents, learn the patterns in those features, and later recognize them. It has been developed since 2006 by Google. It has been backed by years of research. I encourage the reader to look at some of the research that has been made to robustify the library. For example, rsearchers had been working on Table Extraction, tab detection, and layout analysis. For example, in this paper, the research study group works on  different segmentation ways for a page layout.
 
![here](./readme-assets/layout.png)

[ref: https://tesseract-ocr.github.io/docs/Table_detection_in_heterogeneous_documents.pdf]

## Beyond Raw Text Extraction

So far I don’t find any fields, table relationships or I don’t even look at the document type. So let’s fix that. So, I want to learn about the layout in the document and preserve that information. So, this brings us to identifying the document type. I want to identify the document type, and then say if it is a resume, I would like to pay specific attention to bullet points. Because most people use bullet points for their work experience for example. Then, layout analysis is needed as well. If I could annotate them that could be even better. So, first task is to classify the document type. This is not a trivial task. When were doing raw text extraction, we were recognizing strokes/text in an image. You could imagine that you are doing the very basic task of digit classification task on the MNIST dataset, but the input can have text and not just digits. Now, document type classification is a harder problem, because we need to learn patterns on the layout of a document. You can see why this is not a trivial task since there could be infinitely many document types, with many more layout variations. How could even we limit the problem? Luckily, there has been some research in this computer vision field. Specifically, this dataset is prepared for this document type classification problem.
The dataset is RVL-CDIP. https://www.cs.cmu.edu/~aharley/rvl-cdip/ The dataset is about 37 GB. It has 16 class labels like the following: 
1.	letter
2.	form
3.	email
4.	handwritten
5.	advertisement
6.	scientific report
7.	scientific publication
8.	specification
9.	file folder
10.	news article
11.	budget
12.	invoice
13.	presentation
14.	questionnaire
15.	resume
16.	memo


![here](./readme-assets/document.png)

For proof of concept, I trained a Convolutional Neural Network based on the InceptionV2 Architecture on only 5GB of the data (2.5k images per class instead of the original 25k images per class) with 70/10/20 training, validation and testing split, trained for 100 epochs and achieved 65% accuracy. 
loss: 1.1834 - acc: 0.6500 - categorical_crossentropy: 1.1834 - val_loss: 1.2591 - val_acc: 0.6075 - val_categorical_crossentropy: 1.2591. 
Training took 25 hours on Intel i3 CPU. For 16 classes, you can randomly guess, and you would have 1/16=6.25% chances of guessing the document type correctly. Compared to random guessing, 65% that I got is significantly better. Usually, in machine learning, multi -class classification problems get inherently harder as the number of classes grow significantly. The required resources also increase. There are some efforts to optimize the multi-class classifications in deep learning models. It is beyond the scope of this read, but you can find more info here [https://ieeexplore.ieee.org/document/7780615, https://projet.liris.cnrs.fr/imagine/pub/proceedings/CVPR-2009/data/papers/1036.pdf].
The dataset has seem some remarkable classification accuracies over various research group so far since that dataset has been publicly studied for a long time. One of the teams is Microsoft and they achieved 94.42% accuracy on the document type classification on the same dataset that I achieved 60%. They have open sourced the code here [https://github.com/microsoft/unilm/tree/master/layoutlm] and you can find their publication here[ https://arxiv.org/abs/1912.13318]. They also provide the pre-trained models that they have. This is very useful. Because it means the weights that they have fine-tuned until they got 94.42% accuracy are saved and by simply loading their models, you can run classification samples on the model. You can even change the last layer’s output and use the pre-trained model for your own task (for example, you might be working on a task to classifiy a document as email or not, you would simply keep the weights and change the number of nodes from 16 to 2 for your own task). Now,  I am currently using their pre-trained model and running each user’s submitted image on this pre-trained. We can always do raw text extraction from user images. However, knowing the document type has several benefits. We can have fine-tune our raw text extraction for specific document types. For example, we might have machine learning models that are trained to detect tables on a document. Once we know that it is document that is very less likely to have a table in it, we can maybe skip the table model, and save some resources/overhead when user hits our API endpoint. Such optimizations allow for better vertical scalability. 

## References

Python Wrapper for Google’s OCR Engine, https://pypi.org/project/pytesseract/
https://github.com/microsoft/unilm/tree/master/layoutlm
LayoutLM: Pre-training of Text and Layout for Document Image Understanding
, https://arxiv.org/abs/1912.13318



# TextTract
General Worflow is as follows:
![NLP Pipeline](./readme-assets/nlp_pipeline.png)

## General Resources
Pandas Text Processing: https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html

nltk.tokenize package: http://www.nltk.org/api/nltk.tokenize.html

Regex in Python: https://docs.python.org/3/library/re.html

Text Processing Example Python: https://github.com/udacity/AIND-NLP/blob/master/text_processing.ipynb

Beautiful Soup: https://www.crummy.com/software/BeautifulSoup/bs4/doc/

Library for fast text representation and classification: https://github.com/facebookresearch/fastText

GloVe: Global Vectors for Word Representation https://nlp.stanford.edu/projects/glove/
GloVe is better than Word2Vec. How they are different: https://www.quora.com/How-is-GloVe-different-from-word2vec

## Representational Learning
t-SNE: t-Distributed Stochastic Neighbor Embedding. t-SNE is a way to visualize the high dimensional word embeddings.
https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1

t-SNE works with different kinds of dataset too such as images. Take a look at the image ![here](./readme-assets/t-SNE_run_on_CIFAR_10_dataset.png) and the link here: https://medium.com/@pslinge144/representation-learning-cifar-10-23b0d9833c40

- Sample notebook for t-SNE run on CIFAR-10 image data: https://colab.research.google.com/drive/1r6o7GmIOTo_33JyyzbSWunGtunzGK5we?usp=sharing

Efficient Estimation of Word Representations in
Vector Space
:
https://video.udacity-data.com/topher/2018/October/5bc56d28_word2vec-mikolov/word2vec-mikolov.pdf

Distributed Representations of Words and Phrases
and their Compositionality:https://video.udacity-data.com/topher/2018/October/5bc56da8_distributed-representations-mikolov2/distributed-representations-mikolov2.pdf



## Datasets
- The RVL-CDIP Dataset: https://www.cs.cmu.edu/~aharley/rvl-cdip/
https://paperswithcode.com/paper/layoutlm-pre-training-of-text-and-layout-for 94.42% accuracy achieved, best so far on the RVL-CDIP dataset.
Sample notebook: https://www.kaggle.com/nbhativp/document-classification
Sample project 1: https://github.com/jpcrum/Final-Project-Group5
Sample project 2:https://github.com/Ilovemysr/rvl-cdip/blob/master/rvl_cdip_final_extraction.ipynb

- PubLayNet Dataset: https://developer.ibm.com/exchanges/data/all/publaynet/ 
https://dataplatform.cloud.ibm.com/analytics/notebooks/v2/43cb95d9-6c3e-479c-a189-8c9ff3524ec1/view?access_token=bb8ce645cf114b5f5512ae2eb9c7badcf0927f313e8f76b8138d0701289484e6



## OCR 101
A Gentle Intro to OCR:
https://towardsdatascience.com/a-gentle-introduction-to-ocr-ee1469a201aa?gi=58ff4a07494b

## Docker Usage
https://towardsdatascience.com/machine-learning-with-big-data-86bcb39f2f0b

## Very good Example Docker App:
Example Voting App with MySQL, Redis, Python Flask and HTML
https://github.com/dockersamples/example-voting-app

## Best Result achieved for RVL-CDIP
https://github.com/jdegange/Benchmarking-Modern-CNN-Architectures-to-RVL-CDIP
https://medium.com/@jdegange85/benchmarking-modern-cnn-architectures-to-rvl-cdip-9dd0b7ec2955


InceptionResNetV2 with penalized learning rate with vanilla SGD on 512x512 gave the best result

## Some Additional Papers
Modular Multimodal Architecture for Document Classification: https://www.groundai.com/project/modular-multimodal-architecture-for-document-classification/1

Real-Time Document Image Classification using Deep CNN and Extreme Learning Machines: https://www.groundai.com/project/real-time-document-image-classification-using-deep-cnn-and-extreme-learning-machines/1

Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval: https://www.cs.cmu.edu/~aharley/icdar15/

## Final Models
rvlcdip model had final accuracy 65%. 
loss: 1.1834 - acc: 0.6500 - categorical_crossentropy: 1.1834 - val_loss: 1.2591 - val_acc: 0.6075 - val_categorical_crossentropy: 1.2591

## New Model
https://github.com/microsoft/unilm/tree/master/layoutlm

python run_classification.py  --data_dir  data \
                              --model_type layoutlm \
                              --model_name_or_path path/to/pretrained/model/directory \
                              --output_dir path/to/output/directory \
                              --do_lower_case \
                              --max_seq_length 512 \
                              --do_train \
                              --do_eval \
                              --num_train_epochs 40.0 \
                              --logging_steps 5000 \
                              --save_steps 5000 \
                              --per_gpu_train_batch_size 16 \
                              --per_gpu_eval_batch_size 16 \
                              --evaluate_during_training \
                              --fp16 
Similarly, you can do evaluation by changing --do_train to --do_eval and --do_test

## Deployed -AWS

http://54.242.199.97:5000/

# TextTract Server - Client App
Front end: HTML
Backend: Python Flask
Database: None
Deployment: Docker

# Docker Install
Install Docker for Mac or Docker for Windows https://hub.docker.com/editions/community/docker-ce-desktop-windows/
Build the container and run the image...

# How to Run App Locally
```sh
$ cd ./OCR-Engine/flask_server
$ docker build --rm -t flask-ocr .
$ docker run -p 5000:5000 flask-ocr
```
If localhost:5000 is busy, simply change it to a different port number. First 5000 is your localhost port number, 5000 is the docker's binded port number.
For more info on dockers, check here:https://github.com/docker/getting-started

## Test images
https://files.realpython.com/media/ocr.930a7baf9137.jpg
https://files.realpython.com/media/sample1.a36a230755dc.jpg
https://files.realpython.com/media/sample2.36f8074c5273.jpg
https://files.realpython.com/media/sample3.8d93cef43018.jpg
https://files.realpython.com/media/sample4.c68c31b95ffb.jpg
https://files.realpython.com/media/sample5.ca470b17f6d7.jpg

## Current Docker Size
1.25GB


## To Do

- [x] Create UI
- [x] Add a trained model
- [x] Create the Server
- [x] Create API endpoint for recognition
- [x] Add OCR to the client side
- [x] Store Pre-trained models in backend
- [x] Deploy App (AWS)


## Continue to Work on

- [ ] Change UI
- [ ] Fix data extraction from Pdf file type


