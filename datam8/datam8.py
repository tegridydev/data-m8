import os
import json
import yaml
import random
import logging
from datetime import datetime
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import pandas as pd
from sqlitedict import SqliteDict
from tqdm import tqdm
from rich.console import Console
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

from transformers import AutoTokenizer, AutoModelForCausalLM, logging as transformers_logging
from ollama import Ollama

# Suppress warnings from transformers
transformers_logging.set_verbosity_error()

# Initialize Console
console = Console()

# Set up logging
logging.basicConfig(
    filename='datam8.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

CONFIG_FILE = "config_templates.yaml"


class DataM8:
    """Main DataM8 tool for managing datasets and models."""
    
    def __init__(self):
        self.config = self.load_config()
        self.output_path = self.config.get("output_path", "datasets")
        self.cache = SqliteDict("dataset_cache.sqlite", autocommit=True) if self.config.get("cache_enabled") else {}
        self.max_threads = min(self.config.get("max_threads", 4), cpu_count())
        self.llm_config = self.config.get("llm", {})
        self.templates = self.config.get("templates", {})
        
        self.ollama = None
        self.tokenizer = None
        self.model = None

        self.ensure_output_directory()
        self.initialize_llm()

    def load_config(self) -> dict:
        """Loads configuration from a YAML file."""
        try:
            with open(CONFIG_FILE, 'r') as f:
                config = yaml.safe_load(f)
            console.print("[bold green]Configuration loaded successfully.[/]")
            logger.info("Configuration loaded successfully.")
            return config
        except FileNotFoundError:
            console.print(f"[bold red]Configuration file '{CONFIG_FILE}' not found![/]")
            logger.error(f"Configuration file '{CONFIG_FILE}' not found!")
            raise
        except yaml.YAMLError as exc:
            console.print(f"[bold red]Error parsing YAML file: {exc}[/]")
            logger.error(f"Error parsing YAML file: {exc}")
            raise

    def ensure_output_directory(self):
        """Ensures the output directory exists."""
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            console.print(f"[bold green]Created output directory '{self.output_path}'.[/]")
            logger.info(f"Created output directory '{self.output_path}'.")

    def initialize_llm(self):
        """Initializes the appropriate LLM model."""
        model_name = self.llm_config.get("model_name", "gpt2")
        
        if model_name.startswith("ollama"):
            self.load_ollama_model(model_name)
        else:
            self.load_huggingface_model(model_name)

    def load_ollama_model(self, model_name: str):
        """Loads an Ollama model."""
        try:
            console.print(f"[bold blue]Loading Ollama model '{model_name}'...[/]")
            self.ollama = Ollama(model_name)
            console.print("[bold green]Ollama model loaded successfully![/]")
            logger.info(f"Ollama model '{model_name}' loaded.")
        except Exception as e:
            console.print(f"[bold red]Failed to load Ollama model '{model_name}': {e}[/]")
            logger.error(f"Failed to load Ollama model: {e}")
            raise

    def load_huggingface_model(self, model_name: str):
        """Loads a Hugging Face model."""
        try:
            console.print(f"[bold blue]Loading Hugging Face model '{model_name}'...[/]")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            console.print("[bold green]Hugging Face model loaded successfully![/]")
            logger.info(f"Hugging Face model '{model_name}' loaded.")
        except Exception as e:
            console.print(f"[bold red]Failed to load Hugging Face model '{model_name}': {e}[/]")
            logger.error(f"Failed to load Hugging Face model: {e}")
            raise

    def generate_text(self, prompt: str, max_length: int = 50) -> str:
        """Generates text from the loaded LLM."""
        try:
            if self.ollama:
                return self.ollama.generate(prompt)
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.model.generate(inputs["input_ids"], max_length=max_length)
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            console.print(f"[bold red]Text generation failed: {e}[/]")
            logger.error(f"Text generation failed: {e}")
            return ""

    def generate_dataset(self, template_name: str, num_samples: int) -> List[Dict]:
        """Generates a dataset based on a template."""
        template = self.templates.get(template_name, [])
        samples = []

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = [executor.submit(self.generate_sample, template) for _ in range(num_samples)]
            with Progress(
                SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                BarColumn(), TimeElapsedColumn(), console=console
            ) as progress:
                task = progress.add_task("Generating...", total=num_samples)
                for future in tqdm(futures, leave=False):
                    try:
                        sample = future.result()
                        samples.append(sample)
                        progress.advance(task)
                    except Exception as e:
                        console.print(f"[bold red]Error generating sample: {e}[/]")
                        logger.error(f"Error generating sample: {e}")

        return samples

    def generate_sample(self, template: List[Dict]) -> Dict:
        """Generates a single data sample."""
        sample = {}
        for field in template:
            try:
                key, expr = list(field.items())[0]
                sample[key] = eval(expr, {"random": random, "datetime": datetime})
            except Exception as e:
                logger.error(f"Error generating field '{key}': {e}")
                sample[key] = None
        return sample

    def save_dataset(self, data: List[Dict], format: str, filename: str):
        """Saves the dataset in the specified format."""
        filepath = os.path.join(self.output_path, f"{filename}.{format}")
        try:
            if format == "json":
                with open(filepath, "w") as f:
                    json.dump(data, f, indent=4)
            elif format == "csv":
                pd.DataFrame(data).to_csv(filepath, index=False)
            elif format == "parquet":
                pd.DataFrame(data).to_parquet(filepath)
            console.print(f"[bold green]Dataset saved: {filepath}[/]")
            logger.info(f"Dataset saved: {filepath}")
        except Exception as e:
            console.print(f"[bold red]Failed to save dataset: {e}[/]")
            logger.error(f"Failed to save dataset: {e}")

    def run(self):
        """Runs the interactive menu."""
        while True:
            console.print(Panel("[bold magenta]Main Menu[/]"))
            option = IntPrompt.ask(
                "1. Generate Dataset\n2. Exit\nSelect an option", choices=["1", "2"]
            )
            if option == 1:
                self.generate_and_save()
            else:
                console.print("[bold red]Goodbye![/]")
                break

    def generate_and_save(self):
        """Handles dataset generation and saving."""
        template_name = Prompt.ask("Template Name", choices=self.templates.keys())
        num_samples = IntPrompt.ask("Number of Samples", default=10, min=1)
        format = Prompt.ask("Format", choices=["json", "csv", "parquet"], default="json")
        filename = Prompt.ask("Filename", default=template_name)
        data = self.generate_dataset(template_name, num_samples)
        self.save_dataset(data, format, filename)


def main():
    tool = DataM8()
    tool.run()


if __name__ == "__main__":
    main()