import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    def __init__(self, theta=5.0):
        super(BoundaryLoss, self).__init__()
        self.theta = theta
    
    def forward(self, inputs, targets):
        boundaries = self.get_boundaries(targets)
        distance_map = self.compute_distance_map(boundaries)
        boundary_weights = torch.exp(-distance_map / self.theta)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        weighted_loss = boundary_weights * ce_loss
        return weighted_loss.mean()
    
    def get_boundaries(self, masks):
        boundaries = []
        
        for mask in masks:
            num_classes = int(mask.max().item()) + 1
            mask_one_hot = F.one_hot(mask.long(), num_classes=num_classes)  # (H, W, C)
            mask_one_hot = mask_one_hot.permute(2, 0, 1).float()  # (C, H, W)
            
            class_boundaries = []
            for c in range(num_classes):
                class_mask = mask_one_hot[c].unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
                dilated = F.max_pool2d(class_mask, kernel_size=3, stride=1, padding=1)
                eroded = -F.max_pool2d(-class_mask, kernel_size=3, stride=1, padding=1)
                boundary = (dilated - eroded).squeeze()
                class_boundaries.append(boundary)
            combined_boundary = torch.stack(class_boundaries).sum(dim=0)
            boundaries.append(combined_boundary > 0)
        
        return torch.stack(boundaries).float()
    
    def compute_distance_map(self, boundaries):
        distance_maps = []
        max_iterations = 15
        
        for boundary in boundaries:
            dist_map = torch.zeros_like(boundary)
            current_region = boundary.clone()
            for distance in range(1, max_iterations + 1):   
                dilated = F.max_pool2d(
                    current_region.unsqueeze(0).unsqueeze(0),
                    kernel_size=3,
                    stride=1,
                    padding=1
                ).squeeze()
                new_pixels = (dilated > current_region).float()
                dist_map = dist_map + new_pixels * distance
                current_region = dilated
                if new_pixels.sum() == 0:
                    break
            distance_maps.append(dist_map)
        return torch.stack(distance_maps)


class EdgeWeightedLoss(nn.Module):
    def __init__(self, edge_weight=10.0):
        super(EdgeWeightedLoss, self).__init__()
        self.edge_weight = edge_weight
        self.register_buffer('sobel_x', torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3))
    
    def forward(self, inputs, targets):
        edge_mask = self.extract_edges(targets)
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        weights = torch.ones_like(edge_mask)
        weights = weights + edge_mask * (self.edge_weight - 1.0)
        weighted_loss = weights * ce_loss
        return weighted_loss.mean()
    
    def extract_edges(self, masks):
        edges = []
        for mask in masks:
            mask_float = mask.float().unsqueeze(0).unsqueeze(0)
            grad_x = F.conv2d(mask_float, self.sobel_x, padding=1)
            grad_y = F.conv2d(mask_float, self.sobel_y, padding=1)
            gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
            edge = (gradient_magnitude > 0.1).float().squeeze()
            edges.append(edge)
        return torch.stack(edges)
