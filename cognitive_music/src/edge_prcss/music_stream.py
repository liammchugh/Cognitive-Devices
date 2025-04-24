from queue import Queue
from threading import Thread
import sounddevice as sd   # or pyaudio / soundcard
import torch
from transformers import MusicgenForConditionalGeneration, MusicgenProcessor
from app import MusicgenStreamer
import time
import math
import threading

import queue          # NEW
import sounddevice as sd
import threading, time, math
from queue import Queue
from threading import Thread
import torch
from transformers import MusicgenForConditionalGeneration, MusicgenProcessor
from app import MusicgenStreamer


class MUSE_Activity_Player:
    def __init__(self, model_id="facebook/musicgen-small", device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.processor = self._load_model(model_id, device)
        self.device   = device
        self.fs       = self.model.audio_encoder.config.sampling_rate

        # runtime state
        self.chunk_q  = Queue(maxsize=32)
        self.stop_evt = threading.Event()
        self.gen_thread = None

        # audio output
        sd.default.samplerate = self.fs
        self.out = sd.OutputStream(
            channels=2,
            dtype="float32",
            samplerate=self.fs,   # ← pass samplerate here
        )
        self.out.start()          # ← start it explicitly

    @staticmethod
    def _load_model(model_id, device):
        model = MusicgenForConditionalGeneration.from_pretrained(model_id).to(device).half()
        model.eval()
        processor = MusicgenProcessor.from_pretrained(model_id)
        return model, processor

    def _start_generation(self, prompt, play_steps=10):
        # 1) cancel previous run
        old_evt = self.stop_evt          # flag object old threads are using
        old_evt.set()                    # tell them to stop
        while not self.chunk_q.empty():
            try: self.chunk_q.get_nowait()
            except queue.Empty: break
        

        # 2) build new streamer **before** starting thread
        inputs   = self.processor(text=prompt, return_tensors="pt").to(self.device)
        streamer = MusicgenStreamer(self.model, device=self.device, play_steps=play_steps)
        new_evt = threading.Event()      # fresh flag for the new run

        def _cb(audio, stream_end=False):
            if self.stop_evt.is_set():
                return
            try:
                self.chunk_q.put_nowait(audio.astype("float32"))
            except queue.Full:
                pass  # drop if playback is lagging

        streamer.on_finalized_audio = _cb

        # 3) launch generator thread
        self.gen_thread = Thread(
            target=self._run_generate,
            kwargs=dict(
                inputs=inputs,
                streamer=streamer,
                max_new_tokens= int(play_steps * 2),   # ≈ 1.6 s of audio
            ),
            daemon=True,
        )
        self.gen_thread.start()
        self.stop_evt = new_evt # swap the flag on the player instance

    def _run_generate(self, *, inputs, streamer, max_new_tokens):
        """Wrapper to use inference_mode to save memory."""
        with torch.inference_mode():
            self.model.generate(**inputs, streamer=streamer, max_new_tokens=max_new_tokens)

    def push_activity(self, prompt, play_steps_s=1.0):
        frame_rate = self.model.audio_encoder.config.frame_rate   # 50
        play_steps = int(play_steps_s * frame_rate)               # 1 s × 50 = 50
        self._start_generation(prompt, play_steps)

    def audio_loop(self):
        try:
            while True:
                chunk = self.chunk_q.get()
                self.out.write(chunk)
        finally:
            self.out.close()


if __name__ == "__main__":
    player = MUSE_Activity_Player()

    def heartbeat_updater():
        base_prompt = "Cycling in the park"
        while True:
            hr = (math.sin(time.time()) + 1) * 45 + 60  # 60–150 BPM
            prompt = f"{base_prompt}. Heartrate indicator: {int(hr)} BPM."
            print(prompt)
            player.push_activity(prompt)
            time.sleep(5)

    threading.Thread(target=heartbeat_updater, daemon=True).start()
    player.audio_loop()
