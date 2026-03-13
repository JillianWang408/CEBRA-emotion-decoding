"""
CEBRA encoding and decoding adapted for calc/pred data split.

Uses calc for training, pred for testing (same as DPAD_valence, DPAD_9emotion).
Supports: 9emotion, arousal, valence, categories.

Run from project root:

  # 9-emotion (all patients)
  export PATIENT_ID=9
  python -m src.CEBRA.encoding --target 9emotion
  python -m src.CEBRA.encoding_finetune --target 9emotion  # optional two-stage heads
  python -m src.CEBRA.decoding_finetune --target 9emotion

  # Valence/Arousal (patients 9, 27, 28, 15, 22, 24, 29, 30, 31)
  # encoding_finetune is for 9emotion only; for arousal/valence/categories skip it
  python -m src.CEBRA.encoding --patient-id 9 --target arousal
  python -m src.CEBRA.decoding_finetune --patient-id 9 --target arousal
"""
