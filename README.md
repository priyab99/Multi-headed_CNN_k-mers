# Enhancer Prediction using k-mer Features and Multi-Headed CNN

This project predicts DNA enhancer sequences using a deep learning model with four parallel CNN branches, each processing a different k-mer representation (k=1,2,3,4).

## Methodology
- k-mer-based embedding for DNA sequences
- 4-headed CNN with separate convolutional branches
- Trained on 2968 samples, tested on 400 independent sequences

## 📂 Folder Structure
- `dataset/`: Train and Test dataset
- `src/`: requirements, training, and testing scripts
- `logs/`: Training log
- `models/`: Saved model
- `result/`: Test result


## Experimental Results
Training log and test result can be found in the `log/` and `result/`.

## How to Run
```bash
python src/train_enhancers.py
python src/test_enhancers.py
