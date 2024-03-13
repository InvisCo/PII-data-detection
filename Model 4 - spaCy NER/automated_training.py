import functools
import time
from pathlib import Path
from typing import Any

import spacy
from spacy.scorer import Scorer
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.cli.train import train


OUTPUT_DIR = Path("./output/")
CONFIG = Path("./config.cfg")
TRAINED_MODEL_DIR = Path("./trained_models/")
TEST_DATA = Path("./data/test.spacy")
RESULTS = Path("./Training-Results.md")

# ----------------------------- HYPERPARAMETER TUNING -----------------------------

# Maximum number of training sessions to run
MAX_TRAINING_SESSIONS = 2
# The factor of the total range by which to increment the hyperparameters
HYPERPARAMETER_INCREMENT_FACTOR = 0.1
# Hyperparameter (lower bound, upper bound)
HYPERPARAMETER_LIMITS = {
    "optimizer_learn_rate": (0.0001, 0.01),
    "training_dropout": (0.1, 0.5),
    "training_accumulate_gradient": (1, 5),
    "training_patience": (1000, 2000),
    "training_max_epochs": (10, 30),
    "training_max_steps": (10000, 30000),
    "training_eval_frequency": (100, 300),
    "batch_size_start": (50, 150),
    "batch_size_stop": (500, 1500),
    "batch_size_compound": (1.001, 1.01),
    "batcher_tolerance": (0.1, 0.3),
}
# The recommended order to update the hyperparameters
HYPERPARAMETER_UPDATE_ORDER = [
    "optimizer_learn_rate",
    "training_dropout",
    "batch_size_start",
    "batch_size_stop",
    "training_accumulate_gradient",
    "training_patience",
    "training_max_epochs",
    "training_max_steps",
    "training_eval_frequency",
    "batch_size_compound",
    "batcher_tolerance",
]

# ---------------------------------------------------------------------------------


def time_func(func):
    """Wraps a function to time its execution.
    Source: https://towardsdatascience.com/a-simple-way-to-time-code-in-python-a9a175eb0172
    """

    @functools.wraps(func)
    def time_closure(*args, **kwargs) -> tuple[str, Any]:
        """Times the execution of a function and returns the time elapsed along with the function result."""
        start = time.perf_counter()

        # Call the function
        result = func(*args, **kwargs)

        time_elapsed = time.perf_counter() - start

        # Format the time elapsed in mm:ss
        time_formatted = f"{int(time_elapsed // 60):02d}:{int(time_elapsed % 60):02d}"

        return time_formatted, result

    return time_closure


@time_func
def train_model(hyperparameters: dict[str, float | int]):
    """Trains a spaCy NER model using the specified hyperparameters.

    Hyperparameters:
        optimizer_learn_rate (float): The learning rate controls how much to update the model in response to the estimated error each time the model weights are updated. Crucial for convergence speed and quality.
        training_dropout (float): The dropout rate controls the amount of regularization during training, preventing overfitting by randomly setting a fraction of input units to 0 at each update.
        training_accumulate_gradient (int): Determines how many gradients to accumulate before updating model weights. Useful for effectively increasing batch size when limited by GPU memory.
        training_patience (int): Number of evaluation steps with no improvement after which training will be stopped, for early stopping.
        training_max_epochs (int): Sets the maximum number of training epochs. Limits training time and helps prevent overfitting.
        training_max_steps (int): Maximum number of update steps. Another way to control the duration of training.
        training_eval_frequency (int): How often to evaluate the model on the development set. Affects training speed and monitoring of model performance.
        batch_size_start (int): Initial batch size for the compounding batch size scheduler.
        batch_size_stop (int): Maximum batch size for the compounding batch size scheduler.
        batch_size_compound (float): Factor for compounding batch size increase per iteration, controlling how quickly the batch size grows.
        batcher_tolerance (float): Used in batch size calculation to allow for some variability in batch size, affecting memory usage and potentially training stability.
    """
    train(
        CONFIG,
        OUTPUT_DIR,
        use_gpu=0,
        overrides={
            "training.optimizer.learn_rate": hyperparameters["optimizer_learn_rate"],
            "training.dropout": hyperparameters["training_dropout"],
            "training.accumulate_gradient": hyperparameters[
                "training_accumulate_gradient"
            ],
            "training.patience": hyperparameters["training_patience"],
            "training.max_epochs": hyperparameters["training_max_epochs"],
            "training.max_steps": hyperparameters["training_max_steps"],
            "training.eval_frequency": hyperparameters["training_eval_frequency"],
            "training.batcher.size.start": hyperparameters["batch_size_start"],
            "training.batcher.size.stop": hyperparameters["batch_size_stop"],
            "training.batcher.size.compound": hyperparameters["batch_size_compound"],
            "training.batcher.tolerance": hyperparameters["batcher_tolerance"],
        },
    )


def update_hyperparameters(
    hyperparameters: dict[str, float | int]
) -> dict[str, float | int]:
    """Updates the hyperparameters for the next training session, adjusting only one parameter at a time."""

    # Clone the original hyperparameters to avoid side-effects
    new_hyperparameters = hyperparameters.copy()

    for key in HYPERPARAMETER_UPDATE_ORDER:
        current_value = hyperparameters[key]
        lower_bound, upper_bound = HYPERPARAMETER_LIMITS[key]

        # Calculate the increment
        increment = (upper_bound - lower_bound) * HYPERPARAMETER_INCREMENT_FACTOR
        new_value = current_value + increment

        # Ensure the new value does not exceed the upper bound
        if new_value > upper_bound:
            new_value = upper_bound

        # Update the hyperparameter if it has not reached its upper bound
        if current_value < upper_bound:
            new_hyperparameters[key] = new_value
            print(f"Updated {key} from {current_value} to {new_value}")
            # Stop loop once the first eligible hyperparameter is updated
            break

    return new_hyperparameters


def evaluate_model(model: Path, test_data: DocBin) -> dict[str, float]:
    # Load the trained model
    nlp = spacy.load(model)

    # Load the test docs
    test_docs = list(test_data.get_docs(nlp.vocab))

    # Initialize the scorer
    scorer = Scorer(default_lang=nlp.lang, default_pipeline=nlp.pipe_names)

    # Use the model to predict the document entities. Example is a tuple of (doc, gold) where gold is the original annotated document from the test set
    predictions = [Example(nlp(doc.text), doc) for doc in test_docs]

    # Score the predictions and filter the results to only include precision, recall, and F1
    scores = scorer.score(predictions)
    return {
        key: round(float(value), 5)
        for key, value in scores.items()
        if key in ["ents_p", "ents_r", "ents_f"]
    }


def move_model_to_trained_dir(model: Path, new_name: str) -> Path:
    return model.rename(TRAINED_MODEL_DIR / f"model-{new_name}")


def save_scores(
    model: Path,
    name: str,
    training_time: str,
    scores: dict[str, float],
    hyperparameters: dict[str, float | int],
) -> dict[str, dict[str, float | int]]:
    """Saves the scores and hyperparameters for the model to the results file, and returns the score and hyperparameters."""
    with RESULTS.open("a", encoding="UTF-8") as f:
        f.write(f"| [{name}]({TRAINED_MODEL_DIR}{model.name}) ")
        f.write(f"| {training_time} ")
        f.write(f"| {scores['ents_f']} ")
        f.write(f"| {scores['ents_p']} ")
        f.write(f"| {scores['ents_r']} ")
        for key in HYPERPARAMETER_UPDATE_ORDER:
            f.write(f"| {hyperparameters[key]} ")
        f.write("|\n")
    return {
        "name": name,
        "path": model.name,
        "scores": scores,
        "hyperparameters": hyperparameters,
    }


def main():
    # Start hyperparameters at lower bounds
    hyperparameters = {key: value[0] for key, value in HYPERPARAMETER_LIMITS.items()}

    # Store scores and hyperparameters for the all models
    all_scores = []

    # Create a new results file if it does not exist
    if not RESULTS.exists():
        with RESULTS.open("w", encoding="UTF-8") as f:
            f.write("# Results from Training Sessions\n")

    # Add a new section to the results file
    with RESULTS.open("a", encoding="UTF-8") as f:
        f.write("\n\n## Automated Results\n")
        col_headers = [
            "Model",
            "Training Time",
            "F1-Score",
            "Precision",
            "Recall",
        ] + HYPERPARAMETER_UPDATE_ORDER
        for header in col_headers:
            f.write(f"| {header} ")
        f.write(f"|\n{'| - ' * len(col_headers)}|\n")

    # Load the test data
    test_data = DocBin().from_disk(TEST_DATA)

    # Run the training and evaluation process
    for i in range(MAX_TRAINING_SESSIONS):
        model_name = f"{i:02d}"
        print(f"Training model {model_name}...")

        # Train the model with the current hyperparameters and measure the time it took
        training_time, _ = train_model(hyperparameters)
        print(f"Training model {model_name} complete in {training_time}.")

        # Evaluate the model against the test data and get the scores
        scores = evaluate_model(OUTPUT_DIR, test_data)

        # Move the trained model to the trained_models directory
        model_path = move_model_to_trained_dir(OUTPUT_DIR, model_name)

        # Save the scores and hyperparameters for the model
        all_scores.append(
            save_scores(model_path, model_name, training_time, scores, hyperparameters)
        )

        hyperparameters = update_hyperparameters(hyperparameters)

    # Highlight model with best score
    all_scores.sort(key=lambda x: x["scores"]["ents_f"], reverse=True)
    best = all_scores[0]
    with RESULTS.open("a", encoding="UTF-8") as f:
        f.write(
            f"\n*Best Model: [{best['name']}](./trained_models/{best['path']}) -> F1: {best['scores']['ents_f']}, Precision: {best['scores']['ents_p']}, Recall: {best['scores']['ents_r']}*\n"
        )


if __name__ == "__main__":
    print("Starting training...")
    main()
    print("Training complete.")
