import torch
import torch.nn.functional as F


def apply_tta_transform(images, transform_type):
    if transform_type == 'original':
        return images
    elif transform_type == 'hflip':
        return torch.flip(images, dims=[3])
    elif transform_type == 'vflip':
        return torch.flip(images, dims=[2])
    elif transform_type == 'rot90':
        return torch.rot90(images, k=1, dims=[2, 3])
    elif transform_type == 'rot180':
        return torch.rot90(images, k=2, dims=[2, 3])
    elif transform_type == 'rot270':
        return torch.rot90(images, k=3, dims=[2, 3])
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


def reverse_tta_transform(predictions, transform_type):
    if transform_type == 'original':
        return predictions
    elif transform_type == 'hflip':
        return torch.flip(predictions, dims=[3])
    elif transform_type == 'vflip':
        return torch.flip(predictions, dims=[2])
    elif transform_type == 'rot90':
        return torch.rot90(predictions, k=-1, dims=[2, 3])
    elif transform_type == 'rot180':
        return torch.rot90(predictions, k=-2, dims=[2, 3])
    elif transform_type == 'rot270':
        return torch.rot90(predictions, k=-3, dims=[2, 3])
    else:
        raise ValueError(f"Unknown transform type: {transform_type}")


def predict_with_tta(model, images, device, transforms=None):
    if transforms is None:
        transforms = ['original', 'hflip', 'vflip', 'rot90', 'rot180', 'rot270']
    
    with torch.no_grad():
        for transform_type in transforms:
            transformed_images = apply_tta_transform(images, transform_type)
            outputs = model(transformed_images)
            probs = F.softmax(outputs, dim=1)
            reversed_probs = reverse_tta_transform(probs, transform_type)
            all_probs.append(reversed_probs)
    
    averaged_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    predictions = torch.argmax(averaged_probs, dim=1)
    return averaged_probs, predictions


def get_tta_config(mode='standard'):
    configs = {
        'standard': ['original', 'hflip', 'vflip', 'rot90', 'rot180', 'rot270'],
        'flips_only': ['original', 'hflip', 'vflip'],
        'rotations_only': ['original', 'rot90', 'rot180', 'rot270'],
        'minimal': ['original'],
    }
    
    return configs.get(mode, configs['standard'])


if __name__ == '__main__':
