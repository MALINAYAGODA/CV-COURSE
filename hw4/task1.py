import statistics
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128, device="cuda")
    y = torch.randint(0, 2, (10000,), device="cuda")
    return TensorDataset(X, y)


def create_model(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim), 
        nn.ReLU(),
        nn.Linear(hidden_dim, input_dim), 
        nn.ReLU(),
        nn.Linear(input_dim, output_dim)
    ).cuda().train()


def train():
    dataset = prepare_data()
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    model = create_model(input_dim=128, hidden_dim=512, output_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    forward_timing_events = []
    backward_timing_events = []
    accumulated_loss = 0.0
    num_batches = 0

    for batch_idx, (input_batch, label_batch) in enumerate(dataloader):
        augmentation_noise = torch.randn_like(input_batch)
        augmented_input = input_batch + augmentation_noise

        optimizer.zero_grad(set_to_none=True)

        fwd_event_start = torch.cuda.Event(enable_timing=True)
        fwd_event_end = torch.cuda.Event(enable_timing=True)
        fwd_event_start.record()
        
        predictions = model(augmented_input)
        batch_loss = criterion(predictions, label_batch)
        
        fwd_event_end.record()

        bwd_event_start = torch.cuda.Event(enable_timing=True)
        bwd_event_end = torch.cuda.Event(enable_timing=True)
        bwd_event_start.record()
        
        batch_loss.backward()
        
        bwd_event_end.record()
        
        optimizer.step()

        forward_timing_events.append((fwd_event_start, fwd_event_end))
        backward_timing_events.append((bwd_event_start, bwd_event_end))
        
        accumulated_loss += batch_loss.item()
        num_batches += 1

    torch.cuda.synchronize()

    forward_durations = [start.elapsed_time(end) / 1000.0 for start, end in forward_timing_events]
    backward_durations = [start.elapsed_time(end) / 1000.0 for start, end in backward_timing_events]
    mean_loss = accumulated_loss / num_batches

    print(f"Training epoch completed")
    print(f"Average loss: {mean_loss:.4f}")
    print(f"Mean forward pass time: {statistics.mean(forward_durations):.4f} s")
    print(f"Mean backward pass time: {statistics.mean(backward_durations):.4f} s")


if __name__ == '__main__':
    train()
