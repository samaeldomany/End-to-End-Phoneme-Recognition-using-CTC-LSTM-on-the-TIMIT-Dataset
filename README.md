# End-to-End Phoneme Recognition using CTC-LSTM

An end-to-end Automatic Speech Recognition (ASR) project that performs phoneme recognition on the TIMIT speech corpus using a Bidirectional LSTM trained with Connectionist Temporal Classification (CTC) loss. The project implements both Greedy and Beam Search decoding for sequence prediction and evaluates performance using Phoneme Error Rate (PER).

---

## Overview

This project develops a deep learning-based phoneme recognition system capable of predicting phoneme sequences directly from speech without requiring frame-level alignments. Mel-Frequency Cepstral Coefficients (MFCCs) are extracted from audio signals and fed into a Bidirectional LSTM network optimized using CTC Loss.

To improve decoding quality, both Greedy Decoding and Beam Search Decoding are implemented and compared.

---

## Features

- End-to-end phoneme recognition
- MFCC feature extraction using Torchaudio
- Bidirectional LSTM acoustic model
- Connectionist Temporal Classification (CTC) Loss
- Greedy Decoder implementation
- Beam Search Decoder implementation
- Levenshtein-based Phoneme Error Rate (PER) evaluation
- Training loss visualization
- Posterior probability alignment visualization

---

## Model Architecture

```
Speech Audio
      │
      ▼
MFCC Feature Extraction
      │
      ▼
Bidirectional LSTM
      │
      ▼
Linear Classification Layer
      │
      ▼
Log Softmax
      │
      ▼
CTC Loss
      │
      ▼
Greedy / Beam Search Decoder
```

---

## Technologies Used

- Python
- PyTorch
- Torchaudio
- NumPy
- Matplotlib
- KaggleHub
- TIMIT Speech Dataset

---

## Workflow

1. Download and preprocess the TIMIT dataset
2. Extract MFCC features from speech audio
3. Encode phoneme labels
4. Train a Bidirectional LSTM using CTC Loss
5. Decode predictions using Greedy and Beam Search
6. Evaluate performance using Phoneme Error Rate (PER)
7. Visualize training curves and CTC posterior alignments

---

## Evaluation

The model is evaluated using:

- Training Loss
- Phoneme Error Rate (PER)
- Log-Likelihood Scores
- Greedy Decoder Performance
- Beam Search Decoder Performance

---

## Future Improvements

- Replace LSTM with Transformer or Conformer architectures
- Add attention-based decoding
- Train on larger speech datasets such as LibriSpeech
- Integrate language models for improved decoding
- Real-time speech inference deployment

---

## Author

Developed as a Deep Learning and Automatic Speech Recognition project demonstrating sequence modeling with Bidirectional LSTMs and CTC-based training for phoneme recognition.
