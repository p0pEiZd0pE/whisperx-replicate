from cog import BasePredictor, Input, Path, BaseModel
import whisperx
import torch
import gc
import json
from typing import Optional

class Output(BaseModel):
    transcription: str
    segments: str
    word_timestamps: str

class Predictor(BasePredictor):
    def setup(self):
        """Load WhisperX model at startup"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load WhisperX model
        self.model = whisperx.load_model(
            "large-v2",
            self.device,
            compute_type="float16" if self.device == "cuda" else "int8"
        )
        print("WhisperX model loaded successfully!")
    
    def predict(
        self,
        audio: Path = Input(description="Audio file to transcribe"),
        language: str = Input(
            description="Language code (e.g., 'en', 'es', 'fr'). Leave empty for auto-detection",
            default=None
        ),
        batch_size: int = Input(
            description="Batch size for transcription",
            default=16,
            ge=1,
            le=32
        ),
        diarize: bool = Input(
            description="Enable speaker diarization (requires HF token)",
            default=False
        ),
        hf_token: str = Input(
            description="HuggingFace token for diarization (optional)",
            default=None
        )
    ) -> Output:
        """Transcribe audio file with WhisperX"""
        
        # Load audio
        audio_path = str(audio)
        print(f"Loading audio from: {audio_path}")
        audio_data = whisperx.load_audio(audio_path)
        
        # Transcribe
        print("Transcribing...")
        result = self.model.transcribe(
            audio_data,
            batch_size=batch_size,
            language=language
        )
        
        # Align whisper output
        print("Aligning timestamps...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=self.device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio_data,
            self.device,
            return_char_alignments=False
        )
        
        # Optional diarization
        if diarize and hf_token:
            print("Running speaker diarization...")
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token,
                device=self.device
            )
            diarize_segments = diarize_model(audio_data)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # Format output
        full_transcription = " ".join([seg["text"] for seg in result["segments"]])
        
        # Clean up
        gc.collect()
        torch.cuda.empty_cache()
        
        return Output(
            transcription=full_transcription,
            segments=json.dumps(result["segments"], indent=2),
            word_timestamps=json.dumps(result.get("word_segments", []), indent=2)
        )
