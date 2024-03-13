import bisect
import functools
import math
import random
import time
from pathlib import Path
from typing import Any

import spacy
from spacy.cli.train import train
from spacy.scorer import Scorer
from spacy.tokens import DocBin
from spacy.training import Example

OUTPUT_DIR = Path("./output/")
CONFIG = Path("./config.cfg")
TRAINED_MODEL_DIR = Path("./trained_models/")
TEST_DATA = Path("./data/test.spacy")
RESULTS = Path("./Training-Results.md")
# Store results for all models
ALL_RESULTS = []

# ----------------------------- HYPERPARAMETER TUNING -----------------------------

# Maximum number of training sessions to run
MAX_TRAINING_SESSIONS = 50
# The factor of the total range by which to increment the hyperparameters
HYPERPARAMETER_INCREMENT_FACTOR = 0.2
# Floats are rounded to 4 decimal places to avoid floating point errors
HYPERPARAMETER_PRECISION = 4
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
    best_hyperparameters: dict[str, float | int], exploration_rate=0.1
) -> dict[str, float | int]:
    """Updates the hyperparameters for the next training session, adjusting based on the best-performing set.

    Args:
        best_hyperparameters (dict): The best set of hyperparameters found so far.
        exploration_rate (float): The rate at which to introduce randomness in the selection of the next hyperparameters.

    Returns:
        dict[str, float | int]: The new set of hyperparameters for the next training session.
    """
    # Clone the original hyperparameters to avoid side-effects
    new_hyperparameters = best_hyperparameters.copy()

    # Randomly decide whether to explore or exploit
    if random.random() < exploration_rate:
        # Explore: Randomly select a hyperparameter to adjust
        key = random.choice(HYPERPARAMETER_UPDATE_ORDER)

        lower_bound, upper_bound = HYPERPARAMETER_LIMITS[key]
        # Randomly increment or decrement the hyperparameter by a value within the range
        increment = (upper_bound - lower_bound) * HYPERPARAMETER_INCREMENT_FACTOR
        new_value = new_hyperparameters[key] + random.uniform(-increment, increment)

        new_hyperparameters[key] = round(
            max(min(new_value, upper_bound), lower_bound),
            HYPERPARAMETER_PRECISION,
        )
        print(f"Exploring {key}: {new_hyperparameters[key]}")
    else:
        # Exploit: Sequentially adjust hyperparameters based on performance
        for key in HYPERPARAMETER_UPDATE_ORDER:
            lower_bound, upper_bound = HYPERPARAMETER_LIMITS[key]
            # Randomly increment or decrement the hyperparameter
            increment = (
                (upper_bound - lower_bound)
                * random.choice([1, -1])
                * HYPERPARAMETER_INCREMENT_FACTOR
            )
            new_value = round(
                new_hyperparameters[key] + increment,
                HYPERPARAMETER_PRECISION,
            )

            if new_value > upper_bound or new_value < lower_bound:
                continue

            new_hyperparameters[key] = new_value
            print(f"Exploiting {key}: {new_hyperparameters[key]}")
            break

    return new_hyperparameters


@functools.cache
def check_if_hyperparameters_used(hyperparameters: dict[str, float | int]) -> bool:
    """Checks if the hyperparameters have been used in a previous training session."""
    return any(
        all(
            math.isclose(hyperparameters[k], v)
            for k, v in result["hyperparameters"].items()
        )
        for result in ALL_RESULTS
    )


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
) -> dict[str, Any]:
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
    # Default hyperparameters
    hyperparameters = {
        "optimizer_learn_rate": 0.001,
        "training_dropout": 0.1,
        "training_accumulate_gradient": 3,
        "training_patience": 1600,
        "training_max_epochs": 30,
        "training_max_steps": 20000,
        "training_eval_frequency": 200,
        "batch_size_start": 100,
        "batch_size_stop": 1000,
        "batch_size_compound": 1.001,
        "batcher_tolerance": 0.2,
    }

    # Track the best score and hyperparameters
    best_score = 0
    best_hyperparameters = hyperparameters.copy()

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
        bisect.insort(
            ALL_RESULTS,
            save_scores(model_path, model_name, training_time, scores, hyperparameters),
            key=lambda x: x["scores"]["ents_f"],
        )

        # Check if the current model is the best one so far
        current_f1_score = scores["ents_f"]
        if current_f1_score > best_score:
            best_score = current_f1_score
            best_hyperparameters = hyperparameters.copy()

        hyperparameters = update_hyperparameters(best_hyperparameters)

    # Highlight model with best score
    best = ALL_RESULTS[-1]
    with RESULTS.open("a", encoding="UTF-8") as f:
        f.write(
            f"\n*Best Model: [{best['name']}](./trained_models/{best['path']}) -> F1: {best['scores']['ents_f']}, Precision: {best['scores']['ents_p']}, Recall: {best['scores']['ents_r']}*\n"
        )


if __name__ == "__main__":
    print("Starting training...")
    main()
    print("Training complete.")
