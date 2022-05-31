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
4- Link break is "/r" and tab is "t".
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

