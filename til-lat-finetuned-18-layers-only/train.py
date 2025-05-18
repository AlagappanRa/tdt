#!/usr/bin/env python3
import os
import json
import torch
import torch.distributed as dist
from pathlib import Path
import soundfile as sf
from sklearn.model_selection import train_test_split
import nemo.collections.asr as nemo_asr
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from nemo.utils import logging

def build_manifests():
    ASR_DIR = Path("/kaggle/input/til-asr/asr")
    IN_MANIFEST = ASR_DIR / "asr.jsonl"
    TRAIN_MAN = Path("train_manifest.jsonl")
    EVAL_MAN  = Path("eval_manifest.jsonl")

    if dist.get_rank() == 0:
        entries = []
        with open(IN_MANIFEST, "r") as fin:
            for idx, line in enumerate(fin):
                # if idx >= 10:
                #     break
                if not line.strip():
                    continue
                item = json.loads(line)
                wav = ASR_DIR / item["audio"]
                info = sf.info(str(wav))
                txt = item["transcript"].strip()
                entries.append({
                    "audio_filepath": str(wav),
                    "duration": float(info.duration),
                    "text": txt,
                })
        train, val = train_test_split(entries, test_size=0.1, random_state=42)
        with open(TRAIN_MAN, "w") as fout:
            for e in train:
                fout.write(json.dumps(e, ensure_ascii=False) + "\n")
        with open(EVAL_MAN, "w") as fout:
            for e in val:
                fout.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"Wrote {len(train)} train and {len(val)} eval samples.")
    dist.barrier()
    return TRAIN_MAN, EVAL_MAN

def freeze_layers(model):
    for p in model.encoder.pre_encode.parameters():
        p.requires_grad = False
    for i, layer in enumerate(model.encoder.layers):
        if i < 18:
            for p in layer.parameters():
                p.requires_grad = False
    for p in model.decoder.prediction["embed"].parameters():
        p.requires_grad = False

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["OMP_NUM_THREADS"] = "1"
    dist.init_process_group(backend="nccl")

    TRAIN_MAN, EVAL_MAN = build_manifests()
    logging.set_verbosity(logging.INFO)

    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2",
        map_location="cpu"
    )
    freeze_layers(model)

    sr     = 16000
    labels = model.joint.vocabulary
    train_cfg = {
        "manifest_filepath": str(TRAIN_MAN),
        "sample_rate": sr,
        "labels": labels,
        "batch_size": 2,
        "shuffle": True,
        "num_workers": 2,
        "pin_memory": False,
    }
    val_cfg = dict(train_cfg, shuffle=False, manifest_filepath=str(EVAL_MAN))
    model.setup_training_data(train_cfg)
    model.setup_validation_data(val_cfg)

    tb_logger = TensorBoardLogger(
        save_dir="/kaggle/working/tb_logs",
        name="parakeet_finetune",
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=2,
        strategy="ddp",
        precision=16,
        max_epochs=5,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
        logger=tb_logger,
        log_every_n_steps=10,
    )
    trainer.fit(model)

    if dist.get_rank() == 0:
        model.save_to("parakeet_finetuned.nemo")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
