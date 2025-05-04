# create_multi_concept_dataset.py

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import (
    load_dataset,
    DatasetDict,
    concatenate_datasets,
    load_from_disk,
    Value,
    Features,
    Sequence,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
import logging
import os
from typing import List, Dict, Any


# --- Helper: Hook class (copied from create_dataset.py) ---
class HookedTransformer:
    """Auxiliary class used to extract activations from transformer models."""

    def __init__(self, block: int) -> None:
        self.block = block
        self.site = None
        self.remove_handle = None
        self._features = None

    def register_with(self, model, site="mlp"):
        self.site = site
        # Adjust layer access based on actual model architecture if needed
        # This assumes a standard Llama-like structure
        layer_module = model.model.layers[self.block]

        if site == "mlp":
            target_module = layer_module.mlp
        elif site == "block":
            target_module = layer_module
        elif site == "attention":
            # Modify if you need attention outputs
            target_module = layer_module.self_attn
            logger.warning("Using self_attn output for 'attention' site.")
        else:
            raise NotImplementedError(
                f"Site '{site}' not implemented in HookedTransformer."
            )

        self.remove_handle = target_module.register_forward_hook(self)
        return self

    def pop(self) -> torch.Tensor:
        """Remove and return extracted feature from this hook."""
        assert self._features is not None, "Feature extractor was not called yet!"
        # Handle tuple outputs (e.g., from attention layers)
        if isinstance(self._features, tuple):
            # Choose the primary output (usually the first element)
            features = self._features[0]
        else:
            features = self._features
        self._features = None
        return features

    def __call__(self, module, inp, outp) -> None:
        self._features = outp


# --- End Helper ---


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_shakespeare_data(base_path: str) -> DatasetDict:
    """Loads Shakespeare data from specified path."""
    ds_dict = {}
    logger.info(f"Attempting to load Shakespeare data from base path: {base_path}")
    for stage in ["train", "valid", "test"]:
        texts = []
        labels = []  # 0 for modern, 1 for original
        has_data = False
        for label_str, label_val in [("modern", 0), ("original", 1)]:
            file_path = os.path.join(base_path, f"{stage}.{label_str}.nltktok")
            try:
                with open(file_path, "r") as f:
                    sents = [sent.strip() for sent in f if sent.strip()]
                    texts.extend(sents)
                    labels.extend([label_val] * len(sents))
                    logger.info(f"Loaded {len(sents)} sentences from {file_path}")
                    has_data = True
            except FileNotFoundError:
                logger.warning(f"File not found: {file_path}. Skipping.")
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")

        if has_data:
            ds_dict[stage] = Dataset.from_dict({"text": texts, "label_style": labels})
        else:
            logger.warning(f"No data loaded for Shakespeare stage: {stage}")

    if not ds_dict:
        raise FileNotFoundError(
            f"Could not load any Shakespeare data from {base_path}. Please check the path and files."
        )

    return DatasetDict(ds_dict)


def create_unified_dataset(
    rtp_config: Dict,
    shakespeare_config: Dict,
    tokenizer: Any,
    max_seq_length: int,
) -> DatasetDict:
    """Loads, preprocesses, and concatenates datasets."""

    # 1. Load RealToxicityPrompts
    logger.info("Loading RealToxicityPrompts dataset...")
    rtp_ds = load_dataset(rtp_config["path"], split=rtp_config["split"])

    # Preprocess RTP: Keep prompt text, extract toxicity score
    def preprocess_rtp(example):
        prompt = example["prompt"]
        # Use 0.0 if toxicity is None
        toxicity = prompt["toxicity"] if prompt["toxicity"] is not None else 0.0
        # Mark challenging prompts, potentially useful later (-1 if not challenging)
        challenge = 1.0 if example["challenging"] else -1.0
        return {
            "text": prompt["text"],
            "label_toxicity": float(toxicity),
            "label_challenge": challenge,  # Example of adding another potential label
            "source_dataset": "rtp",
        }

    rtp_ds = rtp_ds.map(preprocess_rtp, remove_columns=rtp_ds.column_names)
    logger.info(f"RTP dataset processed. Columns: {rtp_ds.column_names}")

    # 2. Load Shakespeare
    logger.info("Loading Shakespeare dataset...")
    try:
        sp_ds_dict = load_shakespeare_data(shakespeare_config["path"])
        # Combine splits if necessary, or select one (e.g., train)
        sp_ds = concatenate_datasets([sp_ds_dict[split] for split in sp_ds_dict.keys()])
        # Add source marker
        sp_ds = sp_ds.map(lambda x: {"source_dataset": "shakespeare"})
        logger.info(
            f"Shakespeare dataset loaded and combined. Columns: {sp_ds.column_names}"
        )
    except FileNotFoundError as e:
        logger.error(f"{e}. Skipping Shakespeare dataset.")
        sp_ds = None
    except Exception as e:
        logger.error(f"An unexpected error occurred loading Shakespeare data: {e}")
        sp_ds = None

    # 3. Concatenate
    datasets_to_combine = [ds for ds in [rtp_ds, sp_ds] if ds is not None]
    if not datasets_to_combine:
        raise ValueError("No datasets could be loaded or processed.")

    logger.info("Concatenating datasets...")
    combined_ds = concatenate_datasets(datasets_to_combine).shuffle(seed=42)

    # 4. Fill missing labels with -1 and tokenize
    label_columns = [
        "label_toxicity",
        "label_style",
        "label_challenge",
    ]  # Add any other labels here

    def fill_and_tokenize(examples):
        # Fill missing labels that weren't present in the original dataset part
        for col in label_columns:
            if col not in examples:
                examples[col] = [-1.0] * len(
                    examples["text"]
                )  # Create column if missing
            else:
                # Ensure existing column elements are floats, replace None with -1.0
                examples[col] = [
                    (float(x) if x is not None else -1.0) for x in examples[col]
                ]

        # Tokenize
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding=False,  # Pad later in dataloader if needed, or handle in extraction
        )
        # Add back labels - map() might drop non-standard columns
        for col in label_columns:
            tokenized[col] = examples[col]
        tokenized["source_dataset"] = examples["source_dataset"]
        return tokenized

    logger.info("Tokenizing combined dataset and filling missing labels...")
    # Determine columns to remove carefully - keep text for now, remove after tokenization if desired
    cols_to_remove = [
        c
        for c in combined_ds.column_names
        if c not in ["text"] + label_columns + ["source_dataset"]
    ]

    combined_ds_tokenized = combined_ds.map(
        fill_and_tokenize,
        batched=True,
        remove_columns=cols_to_remove,  # Remove original text, maybe keep labels?
        # num_proc=os.cpu_count() // 2 # Enable multiprocessing if beneficial
    )

    # Add sequence IDs
    combined_ds_tokenized = combined_ds_tokenized.map(
        lambda _, i: {"seq_id": i}, with_indices=True
    )

    # Create DatasetDict (e.g., 90/10 split)
    final_ds_dict = combined_ds_tokenized.train_test_split(test_size=0.1, seed=42)
    logger.info(f"Final unified dataset created and split: {final_ds_dict}")
    logger.info(f"Columns: {final_ds_dict['train'].column_names}")
    logger.info(f"Example train entry: {final_ds_dict['train'][0]}")

    return final_ds_dict


def extract_token_level_data(
    model: Any,
    hook: HookedTransformer,
    tokenized_dataset: Dataset,
    label_columns: List[str],
    batch_size: int,
    device: str,
) -> List[Dict]:
    """Extracts activations and aligns labels at the token level."""
    token_data_list = []
    dataloader = DataLoader(
        tokenized_dataset.with_format("torch"), batch_size=batch_size
    )

    logger.info("Starting activation extraction and token-level data alignment...")
    for batch in tqdm(dataloader, desc="Extracting Activations"):
        input_ids = batch["input_ids"].to(device)
        # Create attention mask on the fly if not present (assuming no padding in input_ids)
        attention_mask = batch.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = attention_mask.to(device)

        seq_ids = batch["seq_id"]
        # Get labels for the batch
        batch_labels = {col: batch[col] for col in label_columns if col in batch}

        with torch.no_grad():
            try:
                model(input_ids=input_ids, attention_mask=attention_mask)
                acts_batch = (
                    hook.pop().detach()
                )  # Shape: [batch_size, seq_len, hidden_dim]
            except Exception as e:
                logger.error(f"Error during model forward pass or hook pop: {e}")
                logger.error(
                    f"Input IDs shape: {input_ids.shape}, Attention Mask shape: {attention_mask.shape}"
                )
                continue  # Skip batch on error

        # Iterate through sequences in the batch
        for i in range(acts_batch.shape[0]):  # For each sequence
            # Determine actual sequence length using attention mask
            try:
                # Ensure attention_mask[i] is 1D tensor before calling sum()
                current_attention_mask = attention_mask[i]
                if current_attention_mask.ndim > 1:
                    # This case shouldn't happen if tokenization is done correctly
                    # but handle defensively. Assuming mask is [1, seq_len] or similar.
                    current_attention_mask = (
                        current_attention_mask.squeeze()
                    )  # Remove leading dims if possible
                    if current_attention_mask.ndim != 1:
                        logger.warning(
                            f"Unexpected attention mask shape: {attention_mask[i].shape}. Estimating seq_len."
                        )
                        # Fallback: Use the length of the activation sequence dim
                        seq_len = acts_batch.shape[1]
                    else:
                        seq_len = current_attention_mask.sum().item()
                else:
                    seq_len = current_attention_mask.sum().item()

                # Clamp seq_len to the actual dimension of acts_batch to prevent index errors
                seq_len = min(int(seq_len), acts_batch.shape[1])

            except Exception as e:
                logger.error(
                    f"Error calculating sequence length for item {i} in batch: {e}"
                )
                logger.warning(
                    f"Attention mask shape: {attention_mask[i].shape}. Using acts_batch.shape[1] as fallback."
                )
                seq_len = acts_batch.shape[1]  # Fallback

            if seq_len <= 0:
                logger.warning(f"Sequence length is {seq_len} for item {i}. Skipping.")
                continue

            seq_id = seq_ids[i].item()
            # Get the labels for this specific sequence
            sequence_labels = {col: batch_labels[col][i].item() for col in batch_labels}

            for j in range(seq_len):  # For each valid token in the sequence
                token_data = {
                    "seq_id": seq_id,
                    "token_idx": j,
                    "acts": acts_batch[i, j]
                    .cpu()
                    .half()
                    .tolist(),  # Store as list of float16
                }
                # Add all available labels for this token (inherited from sequence)
                token_data.update(sequence_labels)
                token_data_list.append(token_data)

    logger.info(f"Finished extraction. Total token entries: {len(token_data_list)}")
    if not token_data_list:
        logger.warning(
            "No token data was extracted. Check dataset and processing steps."
        )
        return []  # Return empty list

    # Determine all columns dynamically from the first element
    all_columns = list(token_data_list[0].keys())
    final_df = pd.DataFrame(token_data_list, columns=all_columns)

    # Convert DataFrame back to Hugging Face Dataset
    # Ensure 'acts' is handled correctly (list of floats)
    final_hf_dataset = Dataset.from_pandas(final_df)

    # Define features, especially for the 'acts' sequence
    act_dim = len(final_hf_dataset[0]["acts"])  # Get dimension from first item
    feature_dict = {
        "seq_id": Value("int64"),
        "token_idx": Value("int64"),
        "acts": Sequence(Value("float16"), length=act_dim),  # Specify length and dtype
    }
    for col in all_columns:
        if col not in feature_dict:  # Add label columns (assume float32)
            feature_dict[col] = Value("float32")

    final_hf_dataset = final_hf_dataset.cast(Features(feature_dict))

    logger.info(
        f"Token-level dataset created with columns: {final_hf_dataset.column_names}"
    )
    logger.info(f"Example token entry: {final_hf_dataset[0]}")
    return final_hf_dataset


# --- Main Script Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    BASE_MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
    TOKENIZER_NAME = "meta-llama/Meta-Llama-3-8B"
    HOOK_BLOCK_NUM = 25  # Example block number, adjust as needed
    HOOK_SITE = "mlp"  # Or "block" or "attention"
    MAX_SEQ_LENGTH = 512
    PROCESSING_BATCH_SIZE = 64  # Adjust based on GPU memory
    OUTPUT_DATASET_PATH = f"./datasets/{BASE_MODEL_NAME.split('/')[-1]}-multiconcept-B{HOOK_BLOCK_NUM:02d}-{HOOK_SITE}"
    CACHE_DIR = "./cache"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    RTP_CONFIG = {
        "path": "allenai/real-toxicity-prompts",
        "split": "train",  # Use train split for generating training data
    }
    # IMPORTANT: Replace with the actual path to your Shakespeare data directory
    SHAKESPEARE_CONFIG = {
        "path": "/path/to/your/Shakespearizing-Modern-English/data",  # ADJUST THIS PATH
        # Assumes files like train.modern.nltktok, train.original.nltktok exist here
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_DATASET_PATH), exist_ok=True)

    logger.info(f"Using device: {DEVICE}")
    if DEVICE == "cpu":
        logger.warning("Running on CPU, this will be very slow.")

    # --- Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("Tokenizer loaded.")

    # --- Create Unified Source Dataset ---
    try:
        unified_tokenized_dict = create_unified_dataset(
            rtp_config=RTP_CONFIG,
            shakespeare_config=SHAKESPEARE_CONFIG,
            tokenizer=tokenizer,
            max_seq_length=MAX_SEQ_LENGTH,
        )
    except Exception as e:
        logger.error(f"Failed to create unified tokenized dataset: {e}", exc_info=True)
        exit(1)

    # --- Load Model ---
    logger.info(f"Loading base model: {BASE_MODEL_NAME}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,  # Use float16 for efficiency
    ).to(DEVICE)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    logger.info("Base model loaded.")

    # --- Setup Hook ---
    hook = HookedTransformer(HOOK_BLOCK_NUM).register_with(model, HOOK_SITE)
    logger.info(f"Hook registered at block {HOOK_BLOCK_NUM}, site '{HOOK_SITE}'.")

    # --- Extract Activations and Labels per Token ---
    # Process train and test splits separately
    final_datasets = {}
    label_columns_to_extract = [
        "label_toxicity",
        "label_style",
        "label_challenge",
    ]  # Update if more labels added
    for split in unified_tokenized_dict.keys():
        logger.info(f"Processing split: {split}")
        final_datasets[split] = extract_token_level_data(
            model=model,
            hook=hook,
            tokenized_dataset=unified_tokenized_dict[split],
            label_columns=label_columns_to_extract,
            batch_size=PROCESSING_BATCH_SIZE,
            device=DEVICE,
        )

    final_dataset_dict = DatasetDict(final_datasets)

    # --- Save Final Dataset ---
    if final_dataset_dict:
        logger.info(f"Saving final token-level dataset to: {OUTPUT_DATASET_PATH}")
        final_dataset_dict.save_to_disk(OUTPUT_DATASET_PATH)
        logger.info("Dataset saved successfully.")
    else:
        logger.error("No data was processed. Final dataset is empty.")

    logger.info("Script finished.")
