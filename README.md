# traffic-sign-recognition
Automatic detection and classiffication of polish traffic signs

## Converting models
1. Train model using appropriate script (`train.py`)
2. Use `convert_model.py` to output frozen graph (adapt script for your needs)
3. Use node names to get placeholders for model output.