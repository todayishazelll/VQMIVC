import torch

from model_encoder import Encoder, Encoder_lf0
from model_decoder import Decoder_ac
from model_encoder import SpeakerEncoder as Encoder_spk


from torchinfo import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model instance
model = Decoder_ac().to(device)

# Dummy input tensors
batch_size = 2
seq_len = 140
dim_neck = 64
dim_lf0 = 1
dim_emb = 256

z = torch.randn(batch_size, seq_len // 2, dim_neck).to(device)  # (B, 70, 64) -> will be upsampled to (B, 140, 64)
lf0_embs = torch.randn(batch_size, seq_len, dim_lf0).to(device)  # (B, 140, 1)
spk_embs = torch.randn(batch_size, dim_emb).to(device)  # (B, 256)

# Print model summary

def model_summary():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate models
    encoder = Encoder(in_channels=80, channels=512, n_embeddings=512, z_dim=64, c_dim=256).to(device)
    encoder_lf0 = Encoder_lf0().to(device)
    encoder_spk = Encoder_spk().to(device)
    decoder = Decoder_ac(dim_neck=64).to(device)

    # Print summaries
    print("\nEncoder Summary:")
    summary(encoder, input_size=(1, 80, 100))  # Example input shape (batch_size=1, channels=80, time_steps=100)

    print("\nEncoder_lf0 Summary:")
    summary(encoder_lf0, input_size=(1, 100))  # Example input shape (batch_size=1, time_steps=100)

    print("\nEncoder_spk Summary:")
    summary(encoder_spk, input_size=(1, 80, 100))  # Example input shape (batch_size=1, channels=80, time_steps=100)

    print("\nDecoder Summary:")
    summary(model, input_data=(z, lf0_embs, spk_embs))
    # Decoder receives three inputs: z, lf0_embs, and spk_emb

# Call model summary function before training or inference
if __name__ == "__main__":
    start_time = time.time()  # Start time
    model_summary()
    end_time = time.time()  # End time

    elapsed_time = end_time - start_time
    print(f"\nModel summary execution time: {elapsed_time:.4f} seconds")
