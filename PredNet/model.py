import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, padding=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.Gates = nn.Conv2d(
            input_size + hidden_size, 4 * hidden_size, kernel_size=kernel_size, padding=padding
        )

    def forward(self, x, prev_state):
        batch_size, _, height, width = x.shape
        if prev_state[0] is None and prev_state[1] is None:
            zeros = torch.zeros(batch_size, self.hidden_size, height, width, device=x.device)
            prev_state = (zeros, zeros)
        
        prev_hidden, prev_cell = prev_state
        
        stacked_inputs = torch.cat((x, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)

        # Chunk across channel dimension: input, remember, output, cell
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        cell_gate = torch.tanh(cell_gate)

        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
        return hidden, cell


class PredNet(nn.Module):
    def __init__(self, channels=[3, 48, 96, 192]):
        super().__init__()
        self.channels = channels
        self.n_layers = len(channels)
        
        self.a_conv_layers = nn.ModuleList()
        for i in range(self.n_layers - 1):
            self.a_conv_layers.append(nn.Sequential(
                nn.Conv2d(channels[i] * 2, channels[i + 1], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            ))

        self.conv_lstm_layers = nn.ModuleList()
        self.upsample = nn.Upsample(scale_factor=2)
        
        for i in range(self.n_layers):
            input_dim = channels[i] * 2
            if i < self.n_layers - 1:
                input_dim += channels[i + 1]
            
            self.conv_lstm_layers.append(
                ConvLSTMCell(input_size=input_dim, hidden_size=channels[i])
            )

        self.pred_conv_layers = nn.ModuleList()
        for i in range(self.n_layers):
            self.pred_conv_layers.append(nn.Sequential(
                nn.Conv2d(channels[i], channels[i], kernel_size=3, padding=1),
                nn.ReLU()
            ))

    def forward(self, x_seq):
        batch_size, time_steps, _, h, w = x_seq.shape
        
        R_states = [(None, None) for _ in range(self.n_layers)] # (hidden, cell)
        E_states = [None for _ in range(self.n_layers)] 

        current_h = h
        current_w = w
        for i in range(self.n_layers):
            E_states[i] = torch.zeros(batch_size, 2*self.channels[i], current_h, current_w, device=x_seq.device)
            current_h //= 2
            current_w //= 2

        total_loss = 0
        
        for t in range(time_steps):
            lambda_t = 0 if t == 0 else 1
            A = x_seq[:, t]
            
            # top down
            for l in reversed(range(self.n_layers)):
                lstm_input = E_states[l]
                
                if l < self.n_layers - 1:
                    prev_hidden_upper = R_states[l+1][0]
                    upsampled_R = self.upsample(prev_hidden_upper)
                    lstm_input = torch.cat((lstm_input, upsampled_R), dim=1)
                
                hidden, cell = self.conv_lstm_layers[l](lstm_input, R_states[l])
                R_states[l] = (hidden, cell)

            # bottom up
            for l in range(self.n_layers):
                lambda_l = 1 if l == 0 else 0.1
                    
                A_hat = self.pred_conv_layers[l](R_states[l][0])
                
                if l == 0:
                     A_hat = torch.clamp(A_hat, min=0, max=1.0)

                pos = F.relu(A - A_hat)
                neg = F.relu(A_hat - A)
                E_states[l] = torch.cat([pos, neg], dim=1)

                bs, chan, h, w = E_states[l].size()
                current_layer_loss = E_states[l].sum() / (bs * chan * h * w)
                total_loss += current_layer_loss * lambda_l * lambda_t
                
                if l < self.n_layers - 1:
                    A = self.a_conv_layers[l](E_states[l])

        return total_loss
    
    def extrapolate(self, input_seq, future_steps):
        batch_size, context_steps, channels, h, w = input_seq.shape
        device = input_seq.device
        
        R_states = [(None, None) for _ in range(self.n_layers)]
        E_states = [None for _ in range(self.n_layers)]
        
        current_h, current_w = h, w
        for i in range(self.n_layers):
            E_states[i] = torch.zeros(batch_size, 2*self.channels[i], current_h, current_w, device=device)
            current_h //= 2
            current_w //= 2

        generated_frames = []

        with torch.no_grad():
            for t in range(context_steps):
                A = input_seq[:, t]
                self.step_one_frame(A, R_states, E_states)

        with torch.no_grad():
            prev_prediction = self.pred_conv_layers[0](R_states[0][0]) # hidden first layer
            prev_prediction = torch.clamp(prev_prediction, min=0, max=1.0)
            
            for t in range(future_steps):
                A_hallucinated = prev_prediction
                generated_frames.append(A_hallucinated)
                self.step_one_frame(A_hallucinated, R_states, E_states)
                
                next_prediction = self.pred_conv_layers[0](R_states[0][0])
                next_prediction = torch.clamp(next_prediction, min=0, max=1.0)
                
                prev_prediction = next_prediction

        return torch.stack(generated_frames, dim=1)

    def step_one_frame(self, A_target, R_states, E_states):
        for l in reversed(range(self.n_layers)):
            lstm_input = E_states[l]
            if l < self.n_layers - 1:
                prev_hidden_upper = R_states[l+1][0]
                upsampled_R = self.upsample(prev_hidden_upper)
                lstm_input = torch.cat((lstm_input, upsampled_R), dim=1)
            
            hidden, cell = self.conv_lstm_layers[l](lstm_input, R_states[l])
            R_states[l] = (hidden, cell)

        A_current = A_target
        
        for l in range(self.n_layers):
            A_hat = self.pred_conv_layers[l](R_states[l][0])
            if l == 0:
                A_hat = torch.clamp(A_hat, min=0, max=1.0)

            pos = torch.nn.functional.relu(A_current - A_hat)
            neg = torch.nn.functional.relu(A_hat - A_current)
            E_states[l] = torch.cat([pos, neg], dim=1)
            
            if l < self.n_layers - 1:
                A_current = self.a_conv_layers[l](E_states[l])
