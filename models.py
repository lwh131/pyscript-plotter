import torch
import torch.nn as nn
import torch.nn.functional as F

def init_weights_xavier(m):
    """
    Applies Xavier (Glorot) initialization to the layers of a model.
    
    Usage:
        model = MyModel()
        model.apply(init_weights_xavier)
    """
    # Check if the module is an instance of a convolutional or linear layer
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        # Apply Xavier uniform initialization to the weights
        nn.init.xavier_uniform_(m.weight)
        
        # Initialize biases to a small constant value (e.g., 0.01) or zero.
        # Initializing to a small constant can sometimes help break symmetry,
        # but zero is also a common and safe choice.
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.01)


class ContactPatchRegressor(nn.Module):
    def __init__(self, num_scalars):
        super().__init__()
        # If scalar features are broadcasted and concatenated as channels
        input_channels = 1 + num_scalars # 1 for seabed_z, plus scalars
        
        self.conv_net = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(64, 1, kernel_size=5, padding=2, padding_mode='reflect') # Output 1 channel for contact probability
        )
        # NO SIGMOID HERE if using BCEWithLogitsLoss
        # If using BCELoss, add nn.Sigmoid() here as self.sigmoid = nn.Sigmoid() and apply it in forward

    def forward(self, seabed_profile_input, scalar_features):
        # seabed_profile_input shape: [batch_size, num_profile_points]
        # scalar_features shape: [batch_size, num_scalar_features]

        # 1. Expand scalar features to match profile length
        #    From [B, S] to [B, S, L]
        scalar_features_expanded = scalar_features.unsqueeze(2).expand(-1, -1, seabed_profile_input.size(1))
        
        # 2. Add channel dimension to seabed_profile_input: [B, 1, L]
        seabed_profile_input_chan = seabed_profile_input.unsqueeze(1)
        
        # 3. Concatenate along the channel dimension: [B, 1+S, L]
        combined_input = torch.cat([seabed_profile_input_chan, scalar_features_expanded], dim=1)
        
        # Pass through convolutional layers
        logits = self.conv_net(combined_input) # Output shape [batch, 1, num_profile_points]
        return logits.squeeze(1) # Return [batch, num_profile_points]


class FiLMLayer(nn.Module):
    """
    A standard FiLM (Feature-wise Linear Modulation) layer.
    """
    def __init__(self, conditioning_dim, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.generator = nn.Linear(conditioning_dim, num_channels * 2)

    def forward(self, x, conditioning_features):
        gamma_beta = self.generator(conditioning_features)
        gamma = gamma_beta[:, :self.num_channels]
        beta = gamma_beta[:, self.num_channels:]

        gamma = gamma.unsqueeze(2)
        beta = beta.unsqueeze(2)

        return (x * (gamma + 1)) + beta


class AttentionPooling(nn.Module):
    """ An attention-based pooling layer. """
    def __init__(self, in_features):
        super().__init__()
        # A small network to compute attention scores
        self.attention_net = nn.Sequential(
            nn.Conv1d(in_features, in_features // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(in_features // 2, 1, kernel_size=1)
        )

    def forward(self, x):
        # x shape: [batch, channels, length]
        
        # 1. Compute scores
        # scores shape: [batch, 1, length]
        scores = self.attention_net(x)
        
        # 2. Convert scores to weights (probabilities)
        # weights shape: [batch, 1, length]
        weights = torch.softmax(scores, dim=2)
        
        # 3. Compute weighted average of input features
        # (x * weights) shape: [batch, channels, length]
        # torch.sum(...) shape: [batch, channels]
        weighted_avg = torch.sum(x * weights, dim=2)
        
        return weighted_avg


class SpanRegressor(nn.Module):
    """
    A SpanRegressor that uses FiLM for some scalars and concatenates the rest.
    """
    def __init__(self, num_scalars):
        super().__init__()
        num_film_features = 4
        num_concat_features = num_scalars - num_film_features

        # Input channels = 1 (seabed) + remaining scalars
        input_channels = 1 + num_concat_features
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1, padding_mode='reflect')
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv4 = nn.Conv1d(64, 64, kernel_size=3, padding=1, padding_mode='reflect')

        self.film1 = FiLMLayer(conditioning_dim=num_film_features, num_channels=32)
        self.film2 = FiLMLayer(conditioning_dim=num_film_features, num_channels=64)
        self.film3 = FiLMLayer(conditioning_dim=num_film_features, num_channels=64)
        
        # Activation and dropout
        self.relu = nn.ReLU()

        # Regression head for final output
        self.regression_head = nn.Sequential(
            AttentionPooling(64),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, seabed_profile_input, scalar_features):
        # The first 4 are for FiLM
        film_features = scalar_features[:, :4]
        # The rest are to be concatenated to the input
        concat_features = scalar_features[:, 4:]

        # Add channel dim to seabed: [B, L] -> [B, 1, L]
        seabed_chan = seabed_profile_input.unsqueeze(1)

        # Expand the concat features to match the length of the seabed profile
        # From [B, num_concat] -> [B, num_concat, L]
        concat_features_expanded = concat_features.unsqueeze(2).expand(-1, -1, seabed_profile_input.size(1))

        # Combine seabed and the remaining scalars along the channel dimension
        combined_input = torch.cat([seabed_chan, concat_features_expanded], dim=1)

        # Process through the network
        # First Block
        x = self.conv1(combined_input) # Use the combined input here
        x = self.film1(x, film_features) # Modulate with FiLM features
        x = self.relu(x)

        # Second Block
        x = self.conv2(x)
        x = self.film2(x, film_features) # Modulate with FiLM features
        x = self.relu(x)

        # Second Block
        x = self.conv3(x)
        x = self.film3(x, film_features) # Modulate with FiLM features
        x = self.relu(x)

        # Third Block
        x = self.conv4(x)
        x = self.relu(x)

        # Regression Head
        output = self.regression_head(x)

        return output


class WindowRegressor(nn.Module):
    """
    A WindowRegressor that uses FiLM to condition the network on scalar features.
    This corrected version removes the problematic feature concatenation.
    """
    def __init__(self, num_scalars, num_outputs=2):
        super().__init__()

        # --- FIX 1: The first convolution only needs to see the profile (1 channel) ---
        # The scalar information will be injected by FiLM later.
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv4 = nn.Conv1d(128, 64, kernel_size=3, padding=1, padding_mode='reflect')
        self.conv5 = nn.Conv1d(64, 32, kernel_size=3, padding=1, padding_mode='reflect')

        # The FiLM layers are correctly defined to accept all scalars for conditioning.
        self.film1 = FiLMLayer(conditioning_dim=num_scalars, num_channels=32)
        self.film2 = FiLMLayer(conditioning_dim=num_scalars, num_channels=64)
        self.film3 = FiLMLayer(conditioning_dim=num_scalars, num_channels=128)
        self.film4 = FiLMLayer(conditioning_dim=num_scalars, num_channels=64)
        
        self.relu = nn.ReLU()

        # The regression head is correct.
        self.regression_head = nn.Sequential(
            AttentionPooling(32),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_outputs)
        )

    def forward(self, seabed_profile_input, scalar_features):
        
        # --- FIX 2: Do not concatenate scalars to the input ---
        # The input to the convolutional network is just the profile itself.
        
        # Add channel dim: [B, L] -> [B, 1, L]
        seabed_chan = seabed_profile_input.unsqueeze(1)

        # Process through the network
        # The `scalar_features` are now ONLY used for FiLM modulation.
        x = self.conv1(seabed_chan) # Pass only the profile to the first convolution
        x = self.film1(x, scalar_features)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.film2(x, scalar_features)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.film3(x, scalar_features)
        x = self.relu(x)
        
        x = self.conv4(x)
        x = self.film4(x, scalar_features)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.relu(x)

        output = self.regression_head(x)

        return output


class WindowRegressorNoFilm(nn.Module):
    """
    A simplified 1D CNN Regressor. It combines the seabed profile and scalar features
    by concatenating them along the channel dimension. This is a more standard and
    robust architecture than the previous FiLM-based model, serving as a strong baseline.
    """
    def __init__(self, num_scalars, num_outputs=2):
        super().__init__()

        # The number of input channels for the first convolution is 1 (for the seabed profile)
        # plus the number of scalar features.
        combined_input_channels = 1 + num_scalars

        # A sequential block of convolutional layers for feature extraction.
        self.conv_net = nn.Sequential(
            nn.Conv1d(combined_input_channels, 32, kernel_size=5, padding=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2, padding_mode='reflect'),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2, padding_mode='reflect'), # Added another layer for more depth
            nn.ReLU()
        )

        # The regression head takes the pooled features and maps them to the final outputs.
        self.regression_head = nn.Sequential(
            AttentionPooling(in_features=64), # Pool features across the window length
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_outputs) # Final output layer
        )

    def forward(self, seabed_profile_input, scalar_features):
        # seabed_profile_input shape: [batch_size, num_profile_points]
        # scalar_features shape:      [batch_size, num_scalar_features]

        # 1. Add a channel dimension to the seabed profile input.
        #    Shape: [batch_size, num_profile_points] -> [batch_size, 1, num_profile_points]
        seabed_chan = seabed_profile_input.unsqueeze(1)
        
        # 2. Expand the scalar features to match the length of the profile.
        #    Shape: [batch_size, num_scalars] -> [batch_size, num_scalars, 1] -> [batch_size, num_scalars, num_profile_points]
        scalars_expanded = scalar_features.unsqueeze(2).expand(-1, -1, seabed_profile_input.size(1))

        # 3. Concatenate the profile and expanded scalars along the channel dimension.
        #    This creates a multi-channel input for the CNN.
        #    Shape: [batch_size, 1 + num_scalars, num_profile_points]
        combined_input = torch.cat([seabed_chan, scalars_expanded], dim=1)
        
        # 4. Pass the combined input through the convolutional network.
        #    Shape: [batch_size, 64, num_profile_points]
        conv_features = self.conv_net(combined_input)
        
        # 5. Pass the features through the regression head to get the final prediction.
        #    Shape: [batch_size, num_outputs]
        output = self.regression_head(conv_features)

        return output
    

class FiLMLayer(nn.Module):
    """
    A Feature-wise Linear Modulation layer.
    This layer takes a feature map `x` and applies an affine transformation
    to it, conditioned on parameters `gamma` and `beta` generated from
    another input (e.g., physical parameters).
    
    The operation is: output = gamma * x + beta
    """
    def forward(self, x, gamma, beta):
        # Reshape gamma and beta for broadcasting over the sequence length
        # x.shape:      [batch_size, channels, sequence_length]
        # gamma.shape:  [batch_size, channels] -> [batch_size, channels, 1]
        # beta.shape:   [batch_size, channels] -> [batch_size, channels, 1]
        return gamma.unsqueeze(-1) * x + beta.unsqueeze(-1)


class FiLMGenerator(nn.Module):
    """
    An MLP that generates the `gamma` and `beta` parameters for all FiLM layers
    in the network, based on the scalar physical inputs.
    """
    def __init__(self, num_mlp_inputs, total_film_channels):
        super(FiLMGenerator, self).__init__()
        self.generator = nn.Sequential(
            nn.Linear(num_mlp_inputs, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            # The output layer must produce 2 values (gamma, beta) for each channel
            # across all FiLM'd layers.
            nn.Linear(128, total_film_channels * 2) 
        )

    def forward(self, scalar_input):
        return self.generator(scalar_input)


class InceptionModule1D(nn.Module):
    """
    The Inception module from the 'InceptCurves' paper, adapted for 1D data.
    It applies multiple convolutions of different kernel sizes in parallel and concatenates their outputs.
    This is based on the diagram in Figure 4 of the paper.
    """
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super(InceptionModule1D, self).__init__()

        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, n1x1, kernel_size=1),
            nn.BatchNorm1d(n1x1),
            nn.ReLU(True),
        )

        # Branch 2: 1x1 convolution followed by 3x3 convolution
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm1d(n3x3_reduce),
            nn.ReLU(True),
            nn.Conv1d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm1d(n3x3),
            nn.ReLU(True),
        )

        # Branch 3: 1x1 convolution followed by 5x5 convolution
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm1d(n5x5_reduce),
            nn.ReLU(True),
            nn.Conv1d(n5x5_reduce, n5x5, kernel_size=5, padding=2),
            nn.BatchNorm1d(n5x5),
            nn.ReLU(True),
        )

        # Branch 4: 3x3 max pooling followed by 1x1 convolution
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm1d(pool_proj),
            nn.ReLU(True),
        )
    
    def forward(self, x):
        # Process input through all four parallel branches
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        # Concatenate the outputs along the channel dimension (dim=1)
        return torch.cat([b1, b2, b3, b4], 1)


class InceptCurvesFiLM(nn.Module):
    def __init__(self, num_scalars, num_outputs):
        super(InceptCurvesFiLM, self).__init__()

        # Moderate reduction throughout
        self.pre_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=24, kernel_size=7, stride=2, padding=3),  # 32→24
            nn.BatchNorm1d(24),
            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        self.inception1 = InceptionModule1D(in_channels=24, n1x1=12, n3x3_reduce=18, n3x3=24, n5x5_reduce=3, n5x5=6, pool_proj=6)  # ~25% reduction
        self.film1_channels = 12 + 24 + 6 + 6  # 48 (was 64)
        
        self.inception2 = InceptionModule1D(in_channels=self.film1_channels, n1x1=48, n3x3_reduce=48, n3x3=96, n5x5_reduce=12, n5x5=24, pool_proj=24)  # ~25% reduction
        self.film2_channels = 48 + 96 + 24 + 24  # 192 (was 256)
        
        self.inception3 = InceptionModule1D(in_channels=self.film2_channels, n1x1=48, n3x3_reduce=48, n3x3=96, n5x5_reduce=12, n5x5=24, pool_proj=24)
        self.inception4 = InceptionModule1D(in_channels=192, n1x1=48, n3x3_reduce=48, n3x3=96, n5x5_reduce=12, n5x5=24, pool_proj=24)
        
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.film_layer1 = FiLMLayer()
        self.film_layer2 = FiLMLayer()

        total_film_channels = self.film1_channels + self.film2_channels  # 48 + 192 = 240
        self.film_generator = FiLMGenerator(num_scalars, total_film_channels)

        # Proportional regressor
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(192, 96),  # ~60% reduction
            nn.ReLU(True),
            nn.Dropout(0.25),
            nn.Linear(96, num_outputs)
        )

    def forward(self, seabed_input, scalar_input):
        if seabed_input.dim() == 2:
            # Input shape: [batch_size, seq_len] -> add channel dimension
            seabed_input = seabed_input.unsqueeze(1)  # [batch_size, 1, seq_len]

        # --- Step A: Generate all FiLM parameters from scalar inputs ---
        # film_params shape: [batch_size, (64*2 + 256*2)] = [batch_size, 640]
        film_params = self.film_generator(scalar_input)
        
        # Split the generated parameters for each FiLM layer
        # The split sizes must be (gamma1_size, beta1_size, gamma2_size, beta2_size, ...)
        gamma1, beta1, gamma2, beta2 = torch.split(film_params, 
                                                   [self.film1_channels, self.film1_channels, 
                                                    self.film2_channels, self.film2_channels], dim=1)
        
        # --- Step B: Process seabed profile and apply FiLM modulation ---
        x = self.pre_conv(seabed_input)
        
        # Block 1 + FiLM 1
        x = self.inception1(x)
        x = self.pool(x)
        x = self.film_layer1(x, gamma1, beta1)
        
        # Block 2 + FiLM 2
        x = self.inception2(x)
        x = self.pool(x)
        x = self.film_layer2(x, gamma2, beta2)

        # Remaining blocks (unmodulated)
        x = self.inception3(x)
        x = self.pool(x)
        x = self.inception4(x)

        # --- Step C: Final prediction from modulated features ---
        x = self.avg_pool(x)
        x = x.squeeze(-1)
        output = self.regressor(x)
        
        return output