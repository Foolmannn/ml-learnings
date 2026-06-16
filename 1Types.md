1 .Type of ML


A . Supervised : Regression , Classification 
Data: Numerical: Age, weight and Categorical : Gender, Nation

For Numerical Data and Target. Regression. 
For Categorical Target Classfication

B . Unsupervised : Clustering, Dimensionality Reduction , Anamoly detection, Association Rule Learning

Clustering: For ex what are my customers groups by their buying habits . 

Dimensionality Reduction: For ex. no of rooms + no of washroom -> reducing the dimensions/ columns. For visualizations. https://colah.github.io/posts/2014-10-Visualizing-MNIST/

Anomoly Detection: Detecting outliers, for ex in factory some metrics are not normal then normal metrics. 

Association: For ex a supermarket: a customer buy egg then 60 percent of time they buy butter too. So an strong coorelation is found So we can derive the association. 



C . Semi supervised 
Mixed between supervised and un supervised : Labels requires human effort . So for some data supervised is done using the labels . Then apply it all data by using the un supervised . For ex first use the unsupervised to find the clusters then add labeling to the soem data in clusters which can be associated to all data in clusters


D . Reinforcement
Data does not exists. so learning by making mistakes .For ex self driving car. Chess playing games. 
Learnings is done by doing some action. Then We or someone gives reward or punishment. It works by increaing some reward Maximizing. Like training a dog, child . 


2. Based on training productions: 

    A. Batch / Offline learning
        Train ml model on our own machine. It is used for the large dataset and bigger model.And trained offline on own machine As training the model for huge data  on VPS is costly. 
        In this model is static. ie it remains same for the data trained. So in dynamic environment it will not up to date. 
        For ex. movie recommendation / spam detection. Movies get added. trends changes, spammer used new techniques. 
        So for data is not up to date .
        To solve this. Data is updated then again training is done testing then deployment is done. 
        It is done in entire Data. 

    B. Online learning
        Training is done dynamically. It is done incremental training. It uses mini batches of data for training. As data is done in small batches the training can be done in server too. 
        They are used if the product improves slowly more we use it. For ex Youtube alogrithm. it chnges based on what we consume . 

        #When to use?
        : Where there is concept drift: Fields where the problem space changes regulary
        : Cost Effective : 
        : Faster solution . 


        #How to implement: 
        For ex in sklearn there is partial_fit() in SGD regression. Or use the dedicated library like: River , Vowpal Wabbit

        #Learning Rate:
        We will not want to learn on every changes and not to learn very slowly 

        #Out of Core learning:
        It means when data is very huge it is not possible to train it on the current hardware at once. Then we can use technique of the  online learning for incremental learning. It is not trained online . But it is trained on own hardware but using the techniques of the online learning. Of out of Core learning means the using incremental learning for big data sets


        #Disadvantages:
        Tricky to use 
        Risky 
        For ex false data is being sent. So we have to be careful . Maybe use anomaly detection. Rollback features. 

        # Batch vs Online Learning

        ![alt text](onlineVSoffline.png)