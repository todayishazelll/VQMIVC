import torch

from model_encoder import Encoder, Encoder_lf0
from model_decoder import Decoder_ac
from model_encoder import SpeakerEncoder as Encoder_spk


from torchinfo import summary

class FullModel(torch.nn.Module):
    def __init__(self, encoder_cfg, encoder_lf0_cfg, encoder_spk_cfg, decoder_cfg):
        super(FullModel, self).__init__()
        self.encoder = Encoder(**encoder_cfg)
        self.encoder_lf0 = Encoder_lf0()
        self.encoder_spk = Encoder_spk(**encoder_spk_cfg)
        self.decoder = Decoder_ac(**decoder_cfg)

    def forward(self, mel, lf0):
        # Encoder processes the mel spectrogram
        z, c, z_beforeVQ, vq_loss, perplexity = self.encoder(mel)

        # Encoder_lf0 processes the lf0
        lf0_embs = self.encoder_lf0(lf0)

        # Encoder_spk processes the mel spectrogram to get speaker embeddings
        spk_embs = self.encoder_spk(mel)

        # Decoder generates the output mel spectrogram
        output = self.decoder(z, lf0_embs, spk_embs)
        return output


# Define the model configurations
encoder_cfg = {
    "in_channels": 80,  # Assuming 80 mel bins
    "channels": 512,  # Example value, adjust as needed
    "n_embeddings": 512,  # Example value, adjust as needed
    "z_dim": 64,  # Example value, adjust as needed
    "c_dim": 256  # Example value, adjust as needed
}

encoder_spk_cfg = {
    "c_in": 80,  # Assuming 80 mel bins
    "c_h": 128,  # Example value, adjust as needed
    "c_out": 256,  # Example value, adjust as needed
    "kernel_size": 5,  # Example value, adjust as needed
    "bank_size": 8,  # Example value, adjust as needed
    "bank_scale": 1,  # Example value, adjust as needed
    "c_bank": 128,  # Example value, adjust as needed
    "n_conv_blocks": 6,  # Example value, adjust as needed
    "n_dense_blocks": 6,  # Example value, adjust as needed
    "subsample": [1, 2, 1, 2, 1, 2],  # Example value, adjust as needed
    "act": "relu",  # Example value, adjust as needed
    "dropout_rate": 0  # Example value, adjust as needed
}

decoder_cfg = {
    "dim_neck": 64,  # Example value, adjust as needed
    "dim_lf0": 1,  # Assuming lf0 is a single dimension
    "dim_emb": 256,  # Example value, adjust as needed
    "dim_pre": 512  # Example value, adjust as needed
}

# Create the full model
full_model = FullModel(encoder_cfg, {}, encoder_spk_cfg, decoder_cfg)

# Define the input size for the summary
input_mel = torch.randn(1, 80, 128)  # (batch_size, channels, length)
input_lf0 = torch.randn(1, 128)  # (batch_size, length)

# Print the summary of the full model
print("Full Model Summary:")
summary(full_model, input_data=[input_mel, input_lf0], device="cpu")

# Measure the execution time
print("\nMeasuring execution time...")
start_time = time.time()

# Simulate a forward pass
with torch.no_grad():
    full_model(input_mel, input_lf0)

end_time = time.time()
execution_time = end_time - start_time

print(f"Execution time: {execution_time:.6f} seconds")
