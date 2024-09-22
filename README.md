<h1 align="center">Transformer</h1>

<p align="center">Based on the attention is all you need paper by Google</p>

### Encoder

**Input imbedding**: 
- A sentence of words are tokenized (split into single words). 
- The words are then mapped into numbers that repr the position of words in our vocabulary. Numbers(input ID) are mapped into a vector of 512 numbers. 
- The same word gets mapped to the same embedding. 
- The numbers (embeddings) are changed by the model and hence are not fixed. 
- They change according to the needs of the loss function.

**Positional Encoding**: 
- Each word should carry information about its pos in the sentence. 
- That is what positional encoding does.
- Words closer to each other as "close" and those distant from each other as "distant".
- $$PE(ps, 2i) = \sin(pos\10000^{2i\d} )$$
- $$PE(pos, 2i+1) = \cos(pos\10000^{2i\d})$$
- Positional encodings are computed once and reused.
- Trigonometric functions represent a pattern the model can recognize as continuous, so relative positions are easier to see for the model.

**Self-Attention**:
- it allows the model to relate words to each other
- $$Attention(Q, K, V) = softmax(\frac{QK^{T}}{\\\sqrt{ d }})V$$
- d = 512 / len(seq); T=transpose
- softmax keeps the values between 0 and 1 (they sum up to 1)

**Multi-Head Attention**:
$$Multihead(Q, K, V) = Concat(head_{1}\dots.head_{n})W $$
$$head_{i} = Attention(QW_{i}^{Q}, KW_{i}^{K}, VW_{i}^{V})$$
**Layer Normalization**:
- normalize to make new values in the range 0-1
- beta and gamma parameters are added
- model learns from this parameters to amplify values that need amplification

### Decoder

**Masked Multi-Head**:
- prevent model from seeing future words
- replace the future values with -inf
- the softmax sends the -inf to a very small value -> 0
- this is done after the matmul, just before the softmax function

### How inference and Training works!:

#### Training:

English -> Italian

eg. <\SOS> love you very much <\EOS>

- The example is sent to the encoder

eg. <\SOS> Ti amo molto

- This translation is fed to the decoder
- Padding words are added to make the sentence long enough to reach the sequence length

The output of the encoder is sent to the decoder

decoder -> linear (relate embedding to vocab) -> softmax

What we expect the model to output: 
- Ti amo molto <\EOS>

#### Inference:

- this repeats at time step 1
- at time step 2 we take the output of the previous sentence and append it to the decoder
- sentence did not change, therefore we do not recalculate the output of the encoder at every time step
- only the decoder changes
- this continues till we see the <\EOS> token
