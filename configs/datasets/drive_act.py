"""
Drive and Act Dataset Configuration
Configuration for multi-modal car action recognition using Drive and Act dataset.
"""

# Action classes in Drive and Act dataset
ACTION_CLASSES = [
    'closing_bottle',
    'closing_door_inside',
    'closing_door_outside',
    'closing_laptop',
    'drinking',
    'eating',
    'entering_car',
    'exiting_car',
    'fastening_seat_belt',
    'fetching_an_object',
    'interacting_with_phone',
    'looking_or_moving_around',
    'opening_backpack',
    'opening_bottle',
    'opening_door_inside',
    'opening_door_outside',
    'opening_laptop',
    'placing_an_object',
    'preparing_food',
    'pressing_automation_button',
    'putting_laptop_into_backpack',
    'putting_on_jacket',
    'putting_on_sunglasses',
    'reading_magazine',
    'reading_newspaper',
    'sitting_still',
    'taking_laptop_from_backpack',
    'taking_off_jacket',
    'taking_off_sunglasses',
    'talking_on_phone',
    'unfastening_seat_belt',
    'using_multimedia_display',
    'working_on_laptop',
    'writing'
]

# Dataset configuration
DATASET_CONFIG = {
    'name': 'DriveAndAct',
    'num_classes': len(ACTION_CLASSES),
    'modalities': ['rgb', 'kir'],  # RGB and thermal infrared
    'splits': ['split0', 'split1', 'split2'],
    'video_types': ['video_train', 'video_val', 'video_test'],
    'data_path_template': '{root_dir}/{split}/{video_type}/{modality}/{class_name}/',
    'video_ext': '.mp4',
    'img_size': 224,
    'num_frames': 16,
    'sampling_rate': 4,
    'fps': 30,
    'mean': [0.485, 0.456, 0.406],  # ImageNet normalization for RGB
    'std': [0.229, 0.224, 0.225],
    'kir_mean': [0.5],  # Single channel for thermal
    'kir_std': [0.5]
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 8,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'scheduler': 'cosine',
    'warmup_epochs': 5,
    'save_interval': 10,
    'log_interval': 10
}

# Model configuration
MODEL_CONFIG = {
    'backbone': 'uniformerv2_b16',
    'fusion_type': 'early',
    'dropout': 0.1,
    'hidden_dim': 768,
    'num_heads': 12,
    'depth': 12
}