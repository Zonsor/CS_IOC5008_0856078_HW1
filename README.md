# CS_IOC5008_0856078_HW1

**kaggle website:**
https://www.kaggle.com/c/cs-ioc5008-hw1/overview

**Brief introduction:**  
　The task of the homework is the scene recognition. I train a convolution neural network using transfer learning to complete this competition. Moreover, I do some data augmentation to improve accuracy. And I get the final score of **0.97211** in Kaggle competition.  

**Methodology:**  
　I choose “ResNext-101-32x8d” as my CNN model because it has the lowest classification error rate in ImageNet dataset. And I think that the tasks of these two datasets are slightly different, so when I do transfer learning, I don’t fix any layer. I just use pre-train model and retrain it. Besides, my input size is 256x256 because it is the size of most training set.  
　At first, I get the accuracy of 0.96346. Then I make use of data augmentation by random horizontal flip with 0.5 probability and obtain 96.826% accuracy. Finally, I also use random rotation with 7 angle degree to get the score of 0.97211.  
　The loss function is cross-entropy loss, and the optimizer is SGD. The following table is about the value of some hyperparameters:  

| Hyperparameter | Batch size | Learning rate | Momentum |
| :---           |     ---:   |          ---: |     ---: |
| value          |         20 |         0.001 |      0.9 |
