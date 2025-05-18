#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import logging
import nemo.collections.asr as nemo_asr
from jiwer import Compose, ToLowerCase, SubstituteRegexes, RemovePunctuation, ReduceToListOfListOfWords, wer

def main():
    logging.getLogger("nemo").setLevel(logging.WARNING)
    model = nemo_asr.models.ASRModel.restore_from("parakeet_finetuned.nemo", map_location="cuda:0").cuda().eval()

    EVAL_MAN = Path("eval_manifest.jsonl")
    with open(EVAL_MAN, encoding="utf-8") as f:
        instances = [json.loads(l) for l in f if l.strip()]

    audio_paths = [i["audio_filepath"] for i in instances]
    refs        = [i["text"] for i in instances]
    keys        = [i.get("key", idx) for idx,i in enumerate(instances)]

    hyps = []
    with torch.no_grad():
        for i in tqdm(range(0, len(audio_paths), 2), desc="Inference on GPU0"):
            batch = audio_paths[i:i+2]
            outb  = model.transcribe(batch, batch_size=2)
            hyps.extend([h.text if hasattr(h, "text") else h for h in outb])

    trans = Compose([ToLowerCase(), SubstituteRegexes({"-":" "}), RemovePunctuation(), ReduceToListOfListOfWords()])
    score = 1 - wer(refs, hyps, truth_transform=trans, hypothesis_transform=trans)
    print()
    print(f"✅ 1 - WER on eval set: {score:.3f}")
    print()

    mismatches = []
    for k,r,h in zip(keys, refs, hyps):
        if trans(r) != trans(h):
            mismatches.append((k,r,h))
            if len(mismatches) >= 10:
                break

    print("\n🔍 First 10 mismatches:")
    for k, r, h in mismatches:
        print(f"- {k}\n    REF: {r}\n    HYP: {h}")

    pd.DataFrame({"key": keys, "ref": refs, "hyp": hyps}).to_csv("full_results.csv", index=False)

if __name__ == "__main__":
    main()
