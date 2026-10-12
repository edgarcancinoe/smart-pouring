"""Fine-tune MODELS on the DART dataset (runs.MODELS: from the pretrained checkpoint) and push them to the Hub. See README.md.

Usage: ./run dart-train
"""
import train

MODELS = ["libero-4s-dart"]  # to train, in order


def main():
    train.OUT.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        train.train(name)


if __name__ == "__main__":
    main()
