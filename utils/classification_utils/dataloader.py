from torch.utils.data import Dataset 
from torchvision import transforms
from PIL import Image

class SkinDataset(Dataset):
    def __init__(self, dataframe, transform=None, is_train=True, label_mapping=None):
        self.dataframe = dataframe.to_pandas() if hasattr(dataframe, 'to_pandas') else dataframe
        self.transform = transform
        self.is_train = is_train
     
        if label_mapping is not None:
            self.label_to_idx = label_mapping
            self.labels = list(label_mapping.keys())
        else:
            self.labels = sorted(self.dataframe['disease'].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        
        print(f"Number of classes: {len(self.labels)}")
        if is_train:
            print(f"Training Classes: {self.labels}")
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_path = row['image_path']
        label = row['disease']

        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        label_idx = self.label_to_idx[label]
        return image, label_idx

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
 
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(180),   
 
        transforms.ColorJitter(
            brightness=0.3,   
            contrast=0.3,       
            saturation=0.2,     
            hue=0.05          
        ),
    
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),   
            scale=(0.9, 1.1),      
            shear=10               
        ),
         
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.3,               
            scale=(0.02, 0.15),  
            ratio=(0.3, 3.3),
            value='random'
        ),
        
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform