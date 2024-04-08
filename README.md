# TensorFlow Text Classification with ALBERT

This project demonstrates the use of ALBERT (A Lite BERT) for text classification on the IMDB Reviews dataset. It involves fine-tuning a pre-trained ALBERT model from TensorFlow Hub with TensorFlow and TensorFlow Text for sentiment analysis.

## Prerequisites

Before running this project, ensure you have the following packages installed:

- TensorFlow
- TensorFlow Text
- TensorFlow Datasets
- TensorFlow Hub
- Matplotlib (for plotting training history)

You can install the required packages using pip:

```sh
pip install tensorflow tensorflow_text tensorflow_datasets tensorflow_hub matplotlib
```

## Running the Project

The project is structured into two main parts:

1. Training a model with ALBERT as an embedding layer and fine-tuning it on the IMDB Reviews dataset.
2. Comparing the performance of the base model with the fine-tuned model on the dataset.

To run the project, execute the Python script containing the model definitions, training, and evaluation code.

### Structure and Explanation

- The script begins by installing `sentencepiece` and `tensorflow_text`, which are required for processing the text data and for the ALBERT model, respectively.
- It then sets up the ALBERT model from TensorFlow Hub, including both the preprocessor and the encoder.
- The input text is preprocessed and encoded before being fed into the ALBERT model to obtain embeddings.
- A sequential model is defined with the ALBERT embedding followed by Dense and BatchNormalization layers, optimized using the Adam optimizer.
- The model is trained on the `imdb_reviews` dataset for sentiment analysis, with the training and validation datasets prepared using TensorFlow Datasets.
- After training, the performance of the model is visualized by plotting the accuracy and validation accuracy over epochs.
- To demonstrate fine-tuning, a second model is set up with ALBERT layers set to be trainable, followed by additional Dense and BatchNormalization layers. It's trained for a smaller number of epochs to show the effects of fine-tuning.

### Visualization

Training history is visualized using Matplotlib, comparing the accuracy and validation accuracy of both the base and fine-tuned models.

## License

This project is open-source and available under the MIT License.
