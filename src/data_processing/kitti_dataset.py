import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class KittiDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, num_classes=4):
        """
        Args:
            image_dir (str): Path to the root directory of training images.
            label_dir (str): Path to the root directory of label text files.
            transform (callable, optional): Transformations to apply to the images.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.samples = []

        self.desiredWidth = 1220
        self.desiredHeight = 365

        self.num_classes = num_classes

        # Change to True to recompute croppings if they already exist
        self.shouldCrop = False

        # Parse all scenes and frames
        scenes = sorted(os.listdir(image_dir))
        for scene in scenes:
            if not self.sceneValid(scene):
                continue

            scene_image_dir = os.path.join(image_dir, scene)
            scene_label_dir = os.path.join(label_dir, f"{scene}.txt")
            cropped_image_dir = os.path.join(scene_image_dir, 'cropped')

            if not os.path.exists(scene_label_dir):
                continue

            if not os.path.exists(cropped_image_dir):
                print("\nMissing cropped dataset image directory, recomputing all following croppings...")
                os.makedirs(cropped_image_dir)
                self.shouldCrop = True

            # Parse the label file for this scene
            with open(scene_label_dir, 'r') as f:
                labels = [line.strip().split() for line in f.readlines()]

            # Create a mapping of frame indices to labels
            frame_labels = {}
            for label in labels:
                frame_idx = int(label[0])
                bbox = list(map(float, label[6:10]))
                class_name = label[2]
                class_id = self.getClassMapping(class_name)
                if class_id != -1:
                    if frame_idx not in frame_labels:
                        frame_labels[frame_idx] = []
                    frame_labels[frame_idx].append({"bbox": bbox, "class_id": class_id})

            # Associate frames with labels
            frame_files = sorted(os.listdir(scene_image_dir))
            for frame_file in frame_files:
                if not self.frameValid(frame_file):
                    continue
                
                frame_idx = int(os.path.splitext(frame_file)[0])
                if frame_idx in frame_labels:
                    cropped_image_path = os.path.join(cropped_image_dir, frame_file)

                    if self.shouldCrop:
                        image_path = os.path.join(scene_image_dir, frame_file)
                        self.cropImage(image_path, cropped_image_path)

                    # Update the samples with the new image path
                    self.samples.append({
                        "image_path": cropped_image_path,
                        "labels": frame_labels[frame_idx],
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Format the labels
        bboxes = torch.tensor([label["bbox"] for label in sample["labels"]], dtype=torch.float32)
        class_ids = torch.tensor([label["class_id"] for label in sample["labels"]], dtype=torch.long)

        return image, torch.cat([bboxes, class_ids.unsqueeze(1)], dim=1)


    def sceneValid(self, scene):
        _, file_extension = os.path.splitext(scene)
        file_extension = file_extension.lower()

        if scene == ".DS_Store":
            return False

        return True

    def frameValid(self, frame):
        _, file_extension = os.path.splitext(frame)
        file_extension = file_extension.lower()

        if frame == ".DS_Store":
            return False
        elif frame == "cropped":
            return False

        return True

    def cropImage(self, imagePath, croppedImagePath):
        # Load the image
        image = Image.open(imagePath)
        width, height = image.size

        # Calculate coordinates for cropping the center
        left = (width - self.desiredWidth) / 2
        upper = (height - self.desiredHeight) / 2
        right = (width + self.desiredWidth) / 2
        lower = (height + self.desiredHeight) / 2

        # Adjust coordinates if they are out of bounds
        left = max(0, left)
        upper = max(0, upper)
        right = min(width, right)
        lower = min(height, lower)

        # Crop and resize the image
        image_cropped = image.crop((left, upper, right, lower))
        image_cropped = image_cropped.resize((self.desiredWidth, self.desiredHeight))

        # Save the cropped image
        image_cropped.save(croppedImagePath)


    def getClassMapping(self, classStr):
        if self.num_classes == 5:
            CLASS_MAPPING = {
                "NoObj": 0,
                "Car": 1,
                "Van": 2,
                "Pedestrian": 3,
                "Cyclist": 4,
            }

            return CLASS_MAPPING.get(classStr, -1)
        else:
            CLASS_MAPPING = {
                "Car": 0,
                "Van": 1,
                "Pedestrian": 2,
                "Cyclist": 3,
            }
            return CLASS_MAPPING.get(classStr, -1)
