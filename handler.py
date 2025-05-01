# handler.py

import os
import io
import argparse
import base64
from typing import Dict, Any

import soundfile as sf
import torch

# Import the V2 functions from inference_v2.py
from inference_v2 import load_v2_models, convert_voice_v2

class EndpointHandler:
    def __init__(self, model_dir: str):
        """
        model_dir: the root of your HF repo where your configs and weights live.
        Hugging Face will set this automatically.
        """
        # 1. Build the same CLI args inference_v2 expects (with defaults)
        parser = argparse.ArgumentParser()
        parser.add_argument("--source",           type=str,  default=None)
        parser.add_argument("--target",           type=str,  default=None)
        parser.add_argument("--output",           type=str,  default=".")
        parser.add_argument("--diffusion-steps",  type=int,  default=30)
        parser.add_argument("--length-adjust",    type=float,default=1.0)
        parser.add_argument("--compile",         action="store_true")
        parser.add_argument("--intelligibility-cfg-rate", type=float, default=0.7)
        parser.add_argument("--similarity-cfg-rate",      type=float, default=0.7)
        parser.add_argument("--top-p",                   type=float, default=0.9)
        parser.add_argument("--temperature",             type=float, default=1.0)
        parser.add_argument("--repetition-penalty",      type=float, default=1.0)
        parser.add_argument("--convert-style",    type=str2bool, default=False)
        parser.add_argument("--anonymization-only", type=str2bool, default=False)
        parser.add_argument("--ar-checkpoint-path", type=str, default=None)
        parser.add_argument("--cfm-checkpoint-path", type=str, default=None)

        # Parse an empty list to just load defaults
        args = parser.parse_args([])

        # 2. Override any filesystem paths so that the model_dir is used
        #    HF runtime checks out your repo into model_dir
        args.source = None
        args.target = None
        # output is unused in handler

        # 3. Load the V2 wrapper once
        self.vc_wrapper = load_v2_models(args)

        # 4. Keep args around for inference call
        self.args = args

    def __call__(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Expects JSON:
        {
          "inputs": {
            "source_audio": "<base64-wav>",
            "reference_audio": "<base64-wav>"
          }
        }
        Returns:
        {
          "outputs": {
            "audio": "<base64-wav>"
          }
        }
        """
        # --- 1. Decode base64 inputs ---
        src_b64 = payload["inputs"]["source_audio"]
        ref_b64 = payload["inputs"]["reference_audio"]
        src_bytes = base64.b64decode(src_b64)
        ref_bytes = base64.b64decode(ref_b64)

        src_arr, sr = sf.read(io.BytesIO(src_bytes))
        ref_arr, _  = sf.read(io.BytesIO(ref_bytes))

        # 2. Write arr to temp files so convert_voice_v2 can read paths
        src_path = os.path.join("/tmp", "hf_src.wav")
        ref_path = os.path.join("/tmp", "hf_ref.wav")
        sf.write(src_path, src_arr, sr)
        sf.write(ref_path, ref_arr, sr)

        # 3. Run conversion
        out_audio = convert_voice_v2(
            source_audio_path=src_path,
            target_audio_path=ref_path,
            args=self.args
        )
        # convert_voice_v2 returns tuple (save_sr, waveform_array)
        save_sr, wav_array = out_audio

        # 4. Encode output back to base64
        buf = io.BytesIO()
        sf.write(buf, wav_array, save_sr, format="WAV")
        out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {"outputs": {"audio": out_b64}}


# Optional local test
if __name__ == "__main__":
    # Load small example files
    with open("examples/source.wav","rb") as f:
        src = base64.b64encode(f.read()).decode()
    with open("examples/reference.wav","rb") as f:
        ref = base64.b64encode(f.read()).decode()

    handler = EndpointHandler(model_dir=".")
    response = handler({"inputs": {"source_audio": src, "reference_audio": ref}})
    print("Output b64 length:", len(response["outputs"]["audio"]))
