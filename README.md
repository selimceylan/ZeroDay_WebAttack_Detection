# Zero-day Web Attack Detection
Cloud technologies arises and more people are beginning to migrate their applications and personal data to the cloud. This makes web-based applications an attractive target for cyber-attacks. This github repository includes a deep learning model to solve this problem, leveraging a recent article. Model takes http requests as an input and detects web attacks through them.

## Model
Model is sequence to sequence autoencoder for anomaly detection. Autoencoders use the property of a neural network in a special way to accomplish some efficient methods of training networks to learn normal behavior. When an outlier data point arrives, the auto-encoder cannot codify it well. It learned to represent patterns not existing in this data. Sequence to sequence autoencoder:
![image](https://user-images.githubusercontent.com/86148100/171147843-8d7c7c27-6549-4a84-a7f6-70d039549031.png)


A sequence to sequence model consists of two multilayered long short-term memory (LSTM) models: an encoder and a decoder. The encoder maps the input sequence to a vector of fixed dimensionality. The decoder decodes the target vector using this output of the encoder.

## Dataset
It is difficult to find malicious http requests in sufficient numbers and with varying characteristics. On the other hand, when a model is developed with the malicious dataset, it will be difficult to detect zero-day attacks.

Thanks to autoencoder that finds anomalies:
- Anomaly detection autoencoder models inputs and outputs are the same. Model trains with benign requests. There is no need to malicious requests.
- Model learns the semantic information of the benign requests. It means model can detect zero-day attacks.

In this repo dataset obtained from VulnBank. Model needs roughly 20000 benign request for training.

## Prepare Dataset
A json file created which includes vocabulary based on ASCII. Every character has an index value and some special charecters added.\
1- Decoding sequence starts with "GO" and ends with "EOS".\
2- The characters wihch are not defined in vocabulary are "UNK".\
3- Models takes constant input size therefore size is defined before training. "PAD" is the filler for empty parts.\
4- Link break is "/r" and tab is "t".\
![image](https://user-images.githubusercontent.com/86148100/171168386-2ce6ea47-aff8-4fcc-9596-65fdd41681b2.png)

Constant request length is specified as 1200 characters. The requests which has character more than 1200, eliminated.\
Every http requests starts with "ST@RT" and ends with "END". Requests take by one by with re.compile() method according to the "ST@RT" and "END" statements.

## Citation
Article:
> title={Network attack detection and visual payload labeling technology based on Seq2Seq architecture with attention mechanism},\
  authors={Fan Shi, Pengcheng Zhu, Xiangyu Zhou, Bintao Yuan and Yong Fang},\
  year={2020},\
  journal={International Journal of Distributed Sensor Networks},\
  howpublished={\url{https://journals.sagepub.com/doi/pdf/10.1177/1550147720917019 }}\
  
  Dataset:
  > @misc{vulnbank,\
  title={Vulnerable application for security issues demo},\
  authors={vulnbank, Mikhail Golovanov},\
  year={2019},\
  publisher={Github},\
  journal={GitHub repository},\
  howpublished={\url{https://github.com/vulnbank/vulnbank }}\

## Results
Model trained with these hyperparameters:\
Batch size = 10\
Epoch = 100\
Optimizer = Adam\
Loss function = MAE\
Validation split = 0.1

Loss Graph:\
![indir (1)](https://user-images.githubusercontent.com/86148100/171281276-908dd145-db1f-4243-8e76-b78e74bd74cc.png)

Malicious dataset consist of 1097 requests and model detects 1094 of them with threshold specified below.\
Benign dataset consist of 21991, of which 2462 were eliminated due to their length. 3906 benign requests is initially reserved and not used for training. Model finds correctly 3779 of them from 3906. False positives are 127.

### Determining of Optimum Threshold
Threshold classifies malicious and benign requests. Model predict a loss value for each request. Loss values that greater than the threshold are malicious and vice versa for benign. To find the most compatible threshold value, this formula can be used:\
threshold = mean(total loss)+ C * std(total loss)\
C value is depend on the dataset.

Threshold can find with loss distribution as you can see in this graph:
![indir (2)](https://user-images.githubusercontent.com/86148100/171281522-c6a8e93b-b498-40d3-a97f-a44ee9a68c87.png)

For this dataset, threshold was chosen as 10.


