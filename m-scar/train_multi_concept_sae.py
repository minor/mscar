# train_multi_concept_sae.py (Corrected for Training from Scratch)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)  # Use AutoConfig
from tqdm import tqdm
import argparse
import os
import json
import logging
from typing import Union, Sequence, Dict, Any, Callable, Optional, List

try:
    from llama3_SAE.modeling_llama3_SAE import (
        Autoencoder,
        TopK,
        JumpReLu,
        HeavyStep,
        normalized_mean_squared_error,
        normalized_L1_loss,
    )
except ImportError:
    print(
        "Error importing SCAR Autoencoder modules. Make sure the script is run from the project root"
        " or adjust PYTHONPATH."
    )
    exit(1)


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


# --- Loss Calculation (Copied from previous response - no change needed) ---
def calculate_multi_concept_loss(
    sae_module: Autoencoder,
    latents_pre_act: torch.Tensor,  # h
    latents: torch.Tensor,  # f = sigma_act(h)
    recons: torch.Tensor,  # x_hat
    acts: torch.Tensor,  # x
    batch: Dict[str, torch.Tensor],  # Pass the whole batch for label access
    concept_feature_map: Dict[str, int],
    concept_label_prefix: str,
    lambda_cond: float,
    sparsity_coeff: float,  # Gamma
) -> tuple:
    """
    Calculates the multi-concept loss based on paper equations.
    L_total = L_recon + lambda * L_cond + gamma * L_sparse
    """
    # L_recon (Eq 4) - Normalized MSE
    l2_loss_per_item = normalized_mean_squared_error(recons, acts)
    avg_l2_loss = l2_loss_per_item.mean()

    # L_cond (Eq 6) - Sum of masked BCEs over designated features
    total_cond_loss_per_item = torch.zeros_like(l2_loss_per_item)  # Shape: [batch_size]
    individual_cond_losses_avg = {}  # For logging average loss per concept

    for concept_name, feature_index in concept_feature_map.items():
        label_col_name = f"{concept_label_prefix}{concept_name}"
        if label_col_name not in batch:
            individual_cond_losses_avg[concept_name] = torch.tensor(
                0.0, device=acts.device
            )
            continue

        concept_labels = (
            batch[label_col_name].float().to(acts.device)
        )  # Ensure float and correct device
        valid_mask = (concept_labels != -1.0).float()  # Mask for valid labels

        num_valid = valid_mask.sum()
        if num_valid == 0:
            individual_cond_losses_avg[concept_name] = torch.tensor(
                0.0, device=acts.device
            )
            continue

        if feature_index >= latents_pre_act.shape[-1]:
            logger.error(
                f"Concept '{concept_name}' feature index {feature_index} is out of bounds"
                f" for latents_pre_act shape {latents_pre_act.shape}. Skipping concept."
            )
            individual_cond_losses_avg[concept_name] = torch.tensor(
                0.0, device=acts.device
            )
            continue

        concept_logits = latents_pre_act[:, feature_index]
        bce_loss = F.binary_cross_entropy_with_logits(
            concept_logits, concept_labels, reduction="none"
        )
        masked_bce_loss = bce_loss * valid_mask
        total_cond_loss_per_item += masked_bce_loss
        avg_concept_loss = masked_bce_loss.sum() / num_valid.clamp(min=1e-9)
        individual_cond_losses_avg[concept_name] = avg_concept_loss

    avg_cond_loss = total_cond_loss_per_item.mean()

    # L_sparse (Eq 5) - Optional L1 on post-activation latents f = sigma(h)
    avg_l1_loss = torch.tensor(0.0, device=acts.device)
    if sparsity_coeff > 0:
        l1_loss_per_item = normalized_L1_loss(latents, acts)
        avg_l1_loss = l1_loss_per_item.mean()

    # Combined Objective (Eq 7)
    avg_total_loss = (
        avg_l2_loss + lambda_cond * avg_cond_loss + sparsity_coeff * avg_l1_loss
    )

    return (
        avg_total_loss,
        avg_l1_loss,
        avg_l2_loss,
        avg_cond_loss,
        individual_cond_losses_avg,
    )


# --- Main Training Function ---
def train_new_sae(args, concept_feature_map):
    logger.info(f"Starting NEW SAE training with args: {args}")
    logger.info(f"Concept Feature Map: {concept_feature_map}")

    # --- Setup Device ---
    if not torch.cuda.is_available():
        logger.error("CUDA is not available. Training on GPU is required.")
        return
    device = torch.device("cuda")
    logger.info(f"Using device: {device}")

    # --- Load Base LLM (Frozen) ---
    logger.info(f"Loading BASE LLM (frozen) from: {args.base_llm_path}")
    try:
        # Load configuration first to get hidden size
        llm_config = AutoConfig.from_pretrained(args.base_llm_path)
        if args.hook_site in ["mlp", "block"]:
            n_inputs = llm_config.hidden_size
            logger.info(
                f"Inferred SAE n_inputs (LLM hidden_size) = {n_inputs} based on hook site '{args.hook_site}'"
            )
        elif args.hook_site == "attention":
            n_inputs = llm_config.hidden_size
            logger.warning(
                f"Assuming 'attention' hook site output dimension is hidden_size ({n_inputs}). Verify this."
            )
        else:
            logger.error(
                f"Cannot infer n_inputs for unknown hook_site '{args.hook_site}'. Provide explicitly or check site name."
            )
            return

        # Load the model itself (optional, technically only needed for data generation)
        # We don't *run* the LLM during SAE training, just use its activations from the dataset.
        # base_llm = AutoModelForCausalLM.from_pretrained(
        #     args.base_llm_path,
        #     torch_dtype=torch.float16,
        # ).to(device)
        # base_llm.eval()
        # for param in base_llm.parameters():
        #     param.requires_grad = False
        # logger.info("Base LLM loaded and frozen (though not used directly in SAE training loop).")

    except Exception as e:
        logger.error(f"Error loading base LLM config: {e}", exc_info=True)
        return

    # --- Load Dataset ---
    logger.info(f"Loading dataset from: {args.dataset_path}")
    try:
        if not os.path.isdir(args.dataset_path):
            raise FileNotFoundError(f"Dataset directory not found: {args.dataset_path}")
        dataset = load_from_disk(args.dataset_path)
        if not isinstance(dataset, DatasetDict):
            logger.error(
                "Loaded dataset is not a DatasetDict. Expected 'train' and optionally 'test' splits."
            )
            return
        if "train" not in dataset:
            logger.error("Dataset dictionary must contain a 'train' split.")
            return
        logger.info(f"Dataset loaded: {dataset}")
        # Verify necessary columns
        required_cols = ["acts"] + [
            f"{args.concept_label_prefix}{name}" for name in concept_feature_map
        ]
        missing_cols = [
            col for col in required_cols if col not in dataset["train"].column_names
        ]
        if missing_cols:
            logger.error(
                f"Dataset missing required columns: {missing_cols}. Check dataset generation and concept map/prefix."
            )
            return
        # Check activation dimension matches inferred n_inputs
        try:
            example_act_len = len(dataset["train"][0]["acts"])
            if example_act_len != n_inputs:
                logger.error(
                    f"Activation dimension in dataset ({example_act_len}) does not match "
                    f"inferred n_inputs from LLM config ({n_inputs}). Check hook site or LLM model."
                )
                return
        except Exception as e:
            logger.error(f"Could not verify activation dimension: {e}")

    except FileNotFoundError as e:
        logger.error(e)
        return
    except Exception as e:
        logger.error(f"Error loading dataset: {e}", exc_info=True)
        return

    dataset = dataset.with_format("torch")
    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = None
    if "test" in dataset:
        val_dataloader = DataLoader(
            dataset["test"],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        logger.info("Validation dataloader created.")
    else:
        logger.warning("No 'test' split found in dataset. Skipping validation.")

    # --- Initialize NEW SAE ---
    logger.info("Initializing NEW SAE module...")
    n_latents = (
        args.sae_n_latents
        if args.sae_n_latents
        else n_inputs * args.sae_expansion_factor
    )
    logger.info(f"SAE Configuration: n_inputs={n_inputs}, n_latents={n_latents}")

    # Select activation function
    if args.sae_activation_type == "relu":
        activation_fn = nn.ReLU()
        logger.info("Using ReLU activation.")
    elif args.sae_activation_type == "topk":
        if args.sae_activation_k is None:
            logger.error(
                "Argument --sae_activation_k is required when using --sae_activation_type='topk'"
            )
            return
        activation_fn = TopK(k=args.sae_activation_k, postact_fn=nn.ReLU())
        logger.info(
            f"Using TopK({args.sae_activation_k}) activation (with internal ReLU)."
        )
    elif args.sae_activation_type == "jumprelu":
        activation_fn = JumpReLu()
        logger.info("Using JumpReLU activation.")
    else:
        logger.error(f"Unsupported SAE activation type: {args.sae_activation_type}")
        return

    # Create the Autoencoder instance
    sae_module = Autoencoder(
        n_inputs=n_inputs,
        n_latents=n_latents,
        activation=activation_fn,
        tied=args.sae_tied_weights,
        normalize=args.sae_normalize_inputs,
    ).to(device)
    sae_module.train()  # Set to training mode

    trainable_params = sum(
        p.numel() for p in sae_module.parameters() if p.requires_grad
    )
    logger.info(f"Initialized new SAE. Trainable parameters: {trainable_params:,}")

    # --- Optimizer ---
    # Optimizer targets ONLY the parameters of the new sae_module
    optimizer = optim.Adam(sae_module.parameters(), lr=args.lr)
    logger.info(f"Optimizer initialized for SAE parameters: Adam with lr={args.lr}")

    # --- Training Loop ---
    best_val_loss = float("inf")
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f"Newly trained SAE weights will be saved to: {args.output_dir}")
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    logger.info(f"Automatic Mixed Precision (AMP) enabled: {args.use_amp}")

    for epoch in range(args.epochs):
        logger.info(f"--- Epoch {epoch + 1}/{args.epochs} ---")
        sae_module.train()
        train_loss_accum, train_l1_accum, train_l2_accum, train_cond_accum = (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        train_indiv_cond_accum = {name: 0.0 for name in concept_feature_map.keys()}

        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1} Training")
        for batch_idx, batch in enumerate(progress_bar):
            acts = batch["acts"].to(
                device, dtype=torch.float16
            )  # Activations from dataset

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                # *** Forward pass through the NEW SAE ***
                latents_pre_act, latents, recons = sae_module(acts)

                # Calculate loss using the same multi-concept function
                loss, l1_loss, l2_loss, cond_loss, indiv_cond_losses = (
                    calculate_multi_concept_loss(
                        sae_module,
                        latents_pre_act.float(),
                        latents.float(),
                        recons.float(),
                        acts.float(),
                        batch,
                        concept_feature_map,
                        args.concept_label_prefix,
                        args.lambda_cond,
                        args.sparsity_coeff,
                    )
                )

            scaler.scale(loss).backward()  # loss is computed on SAE outputs only
            scaler.unscale_(optimizer)

            if args.clip_grad_norm > 0:
                # Clip gradients of the SAE parameters
                torch.nn.utils.clip_grad_norm_(
                    sae_module.parameters(), args.clip_grad_norm
                )

            scaler.step(optimizer)  # Updates SAE parameters only
            scaler.update()

            # Accumulate metrics
            train_loss_accum += loss.item()
            train_l1_accum += l1_loss.item()
            train_l2_accum += l2_loss.item()
            train_cond_accum += cond_loss.item()
            for name, concept_loss in indiv_cond_losses.items():
                train_indiv_cond_accum[name] += concept_loss.item()

            log_dict = {
                "Loss": f"{loss.item():.4f}",
                "L1": f"{l1_loss.item():.4f}",
                "L2": f"{l2_loss.item():.4f}",
                "Cond": f"{cond_loss.item():.4f}",
            }
            progress_bar.set_postfix(log_dict)

        # Log epoch averages
        num_batches = len(train_dataloader)
        avg_train_loss = train_loss_accum / num_batches
        avg_l1 = train_l1_accum / num_batches
        avg_l2 = train_l2_accum / num_batches
        avg_cond = train_cond_accum / num_batches
        avg_indiv_cond_str = " | ".join(
            [
                f"{name}: {train_indiv_cond_accum[name] / num_batches:.4f}"
                for name in concept_feature_map.keys()
            ]
        )
        logger.info(
            f"Epoch {epoch + 1} Train Avg Loss: {avg_train_loss:.4f} | L1: {avg_l1:.4f} | L2: {avg_l2:.4f} | Cond: {avg_cond:.4f}"
        )
        logger.info(
            f"Epoch {epoch + 1} Train Avg Indiv Cond Losses: {avg_indiv_cond_str}"
        )

        # --- Validation Loop ---
        if val_dataloader:
            sae_module.eval()
            val_loss_accum, val_l1_accum, val_l2_accum, val_cond_accum = (
                0.0,
                0.0,
                0.0,
                0.0,
            )
            val_indiv_cond_accum = {name: 0.0 for name in concept_feature_map.keys()}

            val_progress_bar = tqdm(
                val_dataloader, desc=f"Epoch {epoch + 1} Validation"
            )
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    for batch in val_progress_bar:
                        acts = batch["acts"].to(device, dtype=torch.float16)
                        # *** Forward pass through SAE ***
                        latents_pre_act, latents, recons = sae_module(acts)
                        # Calculate loss
                        loss, l1_loss, l2_loss, cond_loss, indiv_cond_losses = (
                            calculate_multi_concept_loss(
                                sae_module,
                                latents_pre_act.float(),
                                latents.float(),
                                recons.float(),
                                acts.float(),
                                batch,
                                concept_feature_map,
                                args.concept_label_prefix,
                                args.lambda_cond,
                                args.sparsity_coeff,
                            )
                        )
                        # Accumulate validation metrics
                        val_loss_accum += loss.item()
                        val_l1_accum += l1_loss.item()
                        val_l2_accum += l2_loss.item()
                        val_cond_accum += cond_loss.item()
                        for name, c_loss in indiv_cond_losses.items():
                            val_indiv_cond_accum[name] += c_loss.item()
                        val_progress_bar.set_postfix({"Val_Loss": f"{loss.item():.4f}"})

            num_val_batches = len(val_dataloader)
            avg_val_loss = val_loss_accum / num_val_batches
            avg_val_l1 = val_l1_accum / num_val_batches
            avg_val_l2 = val_l2_accum / num_val_batches
            avg_val_cond = val_cond_accum / num_val_batches
            avg_val_indiv_cond_str = " | ".join(
                [
                    f"{name}: {val_indiv_cond_accum[name] / num_val_batches:.4f}"
                    for name in concept_feature_map.keys()
                ]
            )
            logger.info(
                f"Epoch {epoch + 1} Val Avg Loss: {avg_val_loss:.4f} | L1: {avg_val_l1:.4f} | L2: {avg_val_l2:.4f} | Cond: {avg_val_cond:.4f}"
            )
            logger.info(
                f"Epoch {epoch + 1} Val Avg Indiv Cond Losses: {avg_val_indiv_cond_str}"
            )

            # --- Save Best Model ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(args.output_dir, "sae_best.pth")
                # *** Save only the SAE's state_dict ***
                torch.save(sae_module.state_dict(), save_path)
                logger.info(
                    f"New best SAE saved with validation loss {best_val_loss:.4f} to {save_path}"
                )
        else:
            # Save periodically if no validation
            if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
                save_path = os.path.join(args.output_dir, f"sae_epoch_{epoch + 1}.pth")
                torch.save(sae_module.state_dict(), save_path)
                logger.info(f"SAE state_dict saved at epoch {epoch + 1} to {save_path}")

    # --- Save Final Model ---
    final_save_path = os.path.join(args.output_dir, "sae_final.pth")
    torch.save(sae_module.state_dict(), final_save_path)
    logger.info(f"Final trained SAE state_dict saved to {final_save_path}")

    # --- Save Training Args & Concept Map ---
    args_dict = vars(args)
    args_dict["concept_feature_map"] = concept_feature_map
    # Add inferred/calculated values
    args_dict["sae_n_inputs"] = n_inputs
    args_dict["sae_actual_n_latents"] = n_latents
    args_save_path = os.path.join(args.output_dir, "training_args.json")
    try:
        with open(args_save_path, "w") as f:
            json.dump(args_dict, f, indent=4)
        logger.info(f"Training arguments saved to {args_save_path}")
    except TypeError as e:
        logger.error(f"Could not serialize training arguments to JSON: {e}")

    logger.info("SAE Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a NEW Multi-Concept Sparse Autoencoder (SAE)"
    )

    # --- Paths ---
    parser.add_argument(
        "--base_llm_path",
        type=str,
        required=True,
        help="Path to the BASE LLM (e.g., 'meta-llama/Meta-Llama-3-8B') used for activation generation",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the preprocessed multi-concept activation dataset directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./trained_sae_multi",
        help="Directory to save trained SAE weights and logs",
    )
    parser.add_argument(
        "--hook_site",
        type=str,
        default="mlp",
        choices=["mlp", "block", "attention"],
        help="Site in the LLM where activations were extracted (used to infer n_inputs)",
    )

    # --- SAE Architecture ---
    parser.add_argument(
        "--sae_expansion_factor",
        type=int,
        default=4,
        help="SAE expansion factor (n_latents = n_inputs * expansion_factor). Overridden by --sae_n_latents.",
    )
    parser.add_argument(
        "--sae_n_latents",
        type=int,
        default=None,
        help="Explicit number of latent dimensions for the SAE (overrides expansion_factor).",
    )
    parser.add_argument(
        "--sae_activation_type",
        type=str,
        default="relu",
        choices=["relu", "topk", "jumprelu"],
        help="Activation function type for the SAE.",
    )
    parser.add_argument(
        "--sae_activation_k",
        type=int,
        default=None,
        help="Value of 'k' if using 'topk' activation.",
    )
    parser.add_argument(
        "--sae_tied_weights",
        action="store_true",
        help="Use tied weights for encoder/decoder in SAE.",
    )
    parser.add_argument(
        "--sae_normalize_inputs",
        action="store_true",
        help="Apply Layer Normalization to SAE inputs.",
    )

    # --- Training Hyperparameters ---
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate"
    )  # Often higher for SAEs than LLM finetuning
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Batch size per GPU (larger often better for SAEs)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=8, help="Number of dataloader workers"
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (0 to disable)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs (if no validation)",
    )
    parser.add_argument(
        "--use_amp", action="store_true", help="Use Automatic Mixed Precision (AMP)"
    )

    # --- Concept Conditioning & Sparsity ---
    parser.add_argument(
        "--concept_map",
        type=str,
        required=True,
        help='JSON string mapping concept name to designated feature index (e.g., \'{"toxicity": 0, "style": 1}\')',
    )
    parser.add_argument(
        "--concept_label_prefix",
        type=str,
        default="label_",
        help="Prefix for concept label columns in the dataset (e.g., 'label_toxicity')",
    )
    parser.add_argument(
        "--lambda_cond",
        type=float,
        default=20.0,
        help="Weighting factor (lambda) for the total concept conditioning loss term (L_cond)",
    )
    parser.add_argument(
        "--sparsity_coeff",
        type=float,
        default=2.0,
        help="Coefficient (gamma) for the sparsity loss term (L_sparse - usually L1)",
    )

    args = parser.parse_args()

    # --- Argument Validation ---
    if args.sae_activation_type == "topk" and args.sae_activation_k is None:
        parser.error("--sae_activation_k is required when --sae_activation_type='topk'")
    if args.sae_n_latents is None and args.sae_expansion_factor is None:
        parser.error(
            "Either --sae_n_latents or --sae_expansion_factor must be provided."
        )
    if args.sae_n_latents and args.sae_expansion_factor:
        logger.warning("--sae_n_latents provided, ignoring --sae_expansion_factor.")

    # Parse the concept map
    try:
        concept_feature_map = json.loads(args.concept_map)
        if not isinstance(concept_feature_map, dict):
            raise ValueError("Concept map must be a JSON dictionary.")
        # Check if indices are unique and valid integers
        indices = list(concept_feature_map.values())
        if len(indices) != len(set(indices)):
            raise ValueError("Feature indices in concept map must be unique.")
        if not all(isinstance(i, int) and i >= 0 for i in indices):
            raise ValueError(
                "Feature indices in concept map must be non-negative integers."
            )
        # Check if indices are within bounds (can only be fully checked after n_latents is known)
    except json.JSONDecodeError:
        logger.error(
            f"Invalid JSON string provided for --concept_map: {args.concept_map}"
        )
        exit(1)
    except ValueError as e:
        logger.error(f"Error in concept map: {e}")
        exit(1)

    train_new_sae(args, concept_feature_map)
