import os
import torch

def save_model_state(model, optimizer, epoch, save_path, **kwargs):
    """
    Saves the state of a model, optimizer, and current epoch to a specified file.

    Parameters:
    - model (torch.nn.Module): The model whose state is to be saved.
    - optimizer (torch.optim.Optimizer): The optimizer whose state is to be saved.
    - epoch (int): The current epoch number.
    - save_path (str): The file path where the model state should be saved.
    - **kwargs: Additional state information to be saved.
    """
    # Ensure the directory for save_path exists
    dir_path = os.path.dirname(save_path)
    os.makedirs(dir_path, exist_ok=True)
    
    # Prepare the save dictionary
    save_dict = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(), 
        'epoch': epoch
    }
    
    # Include any additional information provided
    save_dict.update(kwargs)
    
    # Save the model state
    torch.save(save_dict, save_path)
    # Uncomment the next line to print the save location
    # print(f"Model saved to {save_path}")

def load_model_state(model, optimizer, save_path):
    """
    Loads a model state, optimizer state, and epoch from a specified file.

    Parameters:
    - model (torch.nn.Module): The model to load the state into.
    - optimizer (torch.optim.Optimizer): The optimizer to load the state into.
    - save_path (str): The file path from where the model state is to be loaded.

    Returns:
    - int: The epoch number loaded from the file.
    """
    # Load the saved state
    saved_model = torch.load(save_path)
    
    # Restore model and optimizer states
    model.load_state_dict(saved_model['model_state_dict'])
    optimizer.load_state_dict(saved_model['optimizer_state_dict'])
    
    # Return the epoch number
    return saved_model['epoch']
