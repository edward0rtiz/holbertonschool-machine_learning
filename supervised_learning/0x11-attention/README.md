# 0x11 Attention :robot:

>The attention mechanism emerged as an improvement over the encoder decoder-based neural machine translation system in natural language processing (NLP). The encoder LSTM is used to process the entire input sentence and encode it into a context vector, which is the last hidden state of the LSTM/RNN. this project serves as introduction to transformes concept

At the end of this project I was able to solve these conceptual questions:

* What is the attention mechanism?
* How to apply attention to RNNs
* What is a transformer?
* How to create an encoder-decoder transformer model
* What is GPT?
* What is BERT?
* What is self-supervised learning?
* How to use BERT for specific NLP tasks
* What is SQuAD? GLUE?

## Tasks :heavy_check_mark:


- Class RNNEncoder that inherits from tensorflow.keras.layers.Layer to encode for machine translation
-  class SelfAttention that inherits from tensorflow.keras.layers.Layer to calculate the attention for machine translation based on this [paper](https://arxiv.org/pdf/1409.0473.pdf)
- Class RNNDecoder that inherits from tensorflow.keras.layers.Layer to decode for machine translation
- Function def positional_encoding(max_seq_len, dm): that calculates the positional encoding for a transformer
- Function def sdp_attention(Q, K, V, mask=None) that calculates the scaled dot product attention
- Class MultiHeadAttention that inherits from tensorflow.keras.layers.Layer to perform multi head attention
- Class EncoderBlock that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer 
- Class DecoderBlock that inherits from tensorflow.keras.layers.Layer to create an encoder block for a transformer
- Class Encoder that inherits from tensorflow.keras.layers.Layer to create the encoder for a transformer
- Class Decoder that inherits from tensorflow.keras.layers.Layer to create the decoder for a transformer
- Class Transformer that inherits from tensorflow.keras.Model to create a transformer network

## Results :chart_with_upwards_trend:

| Filename |
| ------ |
| [0-rnn_encoder.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x11-attention/0-rnn_encoder.py)|
| [1-self_attention.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x11-attention/1-self_attention.py)|
| [2-rnn_decoder.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x11-attention/2-rnn_decoder.py)|
| [4-positional_encoding.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x11-attention/4-positional_encoding.py)|
| [5-sdp_attention.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x11-attention/5-sdp_attention.py)|
| [6-multihead_attention.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x11-attention/6-multihead_attention.py)|
| [7-transformer_encoder_block.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x11-attention/7-transformer_encoder_block.py)|
| [8-transformer_decoder_block.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x11-attention/8-transformer_decoder_block.py)|
| [9-transformer_encoder.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x11-attention/9-transformer_encoder.py)|
| [10-transformer_decoder.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x11-attention/10-transformer_decoder.py)|
| [11-transformer.py](https://github.com/edward0rtiz/holbertonschool-machine_learning/blob/master/supervised_learning/0x11-attention/11-transformer.py)|


## Additional info :construction:
### Resources

- Python 3.5 / 3.8
- Pycharm Professional 2020.1
- h5py 2.10.0
- tensorflow 2.0
- Keras-Applications 1.0.8
- Keras-Preprocessing 1.1.2
- numpy 1.18.4
- pycodestyle 2.5.0
- scipy 1.4.1
- six 1.14.0
- tensorboard 1.12.2



### Try It On Your Machine :computer:
```bash
git clone https://github.com/edward0rtiz/holbertonschool-machine_learning.git
cd 0x11-attention
./main_files/[filename.py]
```

