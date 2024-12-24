import torch.nn as nn
from torchvision.models import alexnet, resnet18
from models.blocks import *

class MyModel(nn.Module):
    def __init__(self, model_type, batch_size, num_anchors, num_classes):
        super(MyModel, self).__init__()
        
        # Calculate the output dimension
        self.conv_output_dim = 10 #NEED TO CHANGE
        self.batch_size = batch_size
        self.model_type = model_type
        
        # Define a simple fully connected neural network
        self.linear_input_dim = 3072 #NEED TO CHANGE
        self.linear_output_dim = 3072 #NEED TO CHANGE
        
        self.out_channels = num_anchors * (5 + num_classes)
        
    def simpleYOLO(): # "simpleYOLO"

        # Conv block with kernel size 3x3, stride 2 for downsampling
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1,inplace=True)
            )

        # Additional convolution block without downsampling
        def extra_conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1,inplace=True)
            )

        # Final block with kernel_size=2, stride=2 to get exactly 6x19
        def final_conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1,inplace=True)
            )

        # Downsample and increase depth
        # Input: 3 x 365 x 1220
        self.features = nn.Sequential(
            conv_block(3, 32),          # 32 x 182 x 610
            conv_block(32, 64),         # 64 x 91 x 305
            conv_block(64, 128),        # 128 x 46 x 153
            extra_conv_block(128, 128), # 128 x 46 x 153 (extra block)
            conv_block(128, 256),       # 256 x 23 x 77
            extra_conv_block(256, 256), # 256 x 23 x 77 (extra block)
            conv_block(256, 512),       # 512 x 12 x 39
            extra_conv_block(512, 512), # 512 x 12 x 39 (extra block)
            final_conv_block(512, 512), # 512 x 6 x 19
        )

        # A simple 1x1 conv for predictions
        self.pred = nn.Conv2d(512, out_channels, kernel_size=1)
        
    def tinyYOLO(): # "tinyYOLO"

        # Conv block with kernel size 3x3, stride 2 for downsampling
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True)
            )

        # Final block to adjust depth and output size
        def final_conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True)
            )

        # Downsample and increase depth
        # Input: 3 x 365 x 1220
        self.features = nn.Sequential(
            conv_block(3, 16),          # 16 x 182 x 610
            conv_block(16, 32),         # 32 x 91 x 305
            conv_block(32, 64),         # 64 x 46 x 153
            conv_block(64, 128),        # 128 x 23 x 77
            conv_block(128, 256),       # 256 x 12 x 39
        )

        # Final prediction layer
        self.pred = nn.Sequential(
            final_conv_block(256, out_channels),
            nn.AdaptiveAvgPool2d((6, 19))  # Ensure output size is 6x19
        )
            
    def encoderdecoderYOLO(): # "encoderdecoderYOLO"

        # Encoder block
        def conv_block(in_channels, out_channels, kernel_size=3, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True)
            )

        # Decoder block
        def upsample_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True)
            )

        # Encoder: Downsampling and increasing depth
        self.encoder = nn.Sequential(
            conv_block(3, 32),          # 32 x 182 x 610
            conv_block(32, 64),         # 64 x 91 x 305
            conv_block(64, 128),        # 128 x 46 x 153
            conv_block(128, 256),       # 256 x 23 x 77
            conv_block(256, 512),       # 512 x 12 x 39
        )

        # Bottleneck: Reduce spatial size further
        self.bottleneck = conv_block(512, 1024, kernel_size=2, stride=2, padding=0)  # 1024 x 6 x 19

        # Decoder: Upsampling and reducing depth
        self.decoder = nn.Sequential(
            upsample_block(1024, 512),  # 512 x 12 x 39
            upsample_block(512, 256),   # 256 x 24 x 78
            upsample_block(256, 128),   # 128 x 48 x 156
            upsample_block(128, 64),    # 64 x 96 x 312
            upsample_block(64, 32),     # 32 x 192 x 624
        )

        # Final 1x1 conv to get desired output size
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)  # Adjust depth to num_anchors * (5 + num_classes)

        # Output reshape layer
        self.output_reshape = nn.Conv2d(out_channels, 18, kernel_size=1)  # Final depth adjustment
        
    def attentionYOLO(self): # "attentionYOLO"

        # Stem
        self.layer1 = conv_block(3, 32, stride=2)  # Downsample: 3->32
        self.layer2 = ResidualBlock(32, 64, stride=2)  # Downsample: 32->64
        self.layer3 = ResidualBlock(64, 128, stride=2)  # Downsample: 64->128

        # Add SE block to enhance channel attention here
        self.se1 = SEBlock(128)

        # An extra residual block instead of extra_conv_block
        self.layer4 = ResidualBlock(128, 128, stride=1)

        self.layer5 = ResidualBlock(128, 256, stride=2)  # 128->256
        self.se2 = SEBlock(256)
        self.layer6 = ResidualBlock(256, 256, stride=1)

        self.layer7 = ResidualBlock(256, 512, stride=2)  # 256->512
        self.se3 = SEBlock(512)
        self.layer8 = ResidualBlock(512, 512, stride=1)

        # Instead of the final_conv_block, we directly downsample to 6x19 if needed
        # Or you can keep a final downsampling block as-is for original dimensions
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(512),
            SiLU()
        )

        # SPP before final prediction: Increases receptive field
        self.spp = SPP()
        # After SPP, channels: 512 * 4 = 2048 because SPP concatenates features
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(512 * 4, 512, kernel_size=1, stride=1),
            nn.BatchNorm2d(512),
            SiLU()
        )

        # Final Prediction Layer
        self.pred = nn.Conv2d(512, self.out_channels, kernel_size=1)
            
    def linear(self): # "linear":
        drop_rate = 0.01
        
        self.model = nn.Sequential(
            nn.Linear(self.linear_input_dim, 128),   # Input layer
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            
            nn.Linear(128, 256), 
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            
            nn.Linear(128, self.linear_output_dim)   # Output layer
        )

        # Print the number of training parameters in the model
        self.count_params(self.model)
            
    # Resnet18
    def resnet(): # "resnet"
        self.model = resnet18(weights=False)
        
        self.model.conv1 = nn.Conv2d(in_channels=3,
            out_channels=64,
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False
        )
        
        self.model.fc = nn.Linear(512, self.conv_output_dim)
        
        # Print the number of training parameters in the model
        self.count_params(self.model)
        
    # Alexnet
    def alexnet(): # "alexnet"
        self.model = alexnet(weights=False)
        
        self.model.features[0] = nn.Conv2d(in_channels=1,
            out_channels=64,
            kernel_size=2,
            stride=1,
            padding=0,
            bias=False
        )
    
        self.model.classifier[6] = nn.Linear(4096, self.conv_output_dim)
        
        # Print the number of training parameters in the model
        self.count_params(self.model)
                    
    # Custom CNN
    def customcnn(): # "customcnn"
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            
            # Conv layer, 2D becuase input is image matrix
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=1,padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
                        
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(512),
            
            # nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,stride=2,padding=1),
            # nn.BatchNorm2d(1024),
            # nn.LeakyReLU(),
            # nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,stride=1,padding=1),
            # nn.BatchNorm2d(1024),
            
            nn.AdaptiveAvgPool2d((1,1)),
        )
    
        self.fully_connected = nn.Sequential(
            nn.Linear(512,out_features=10)
        )
            
        # Custom CNN based on YOLO... comments are output sizes of that layer (chanels, height, width)
    def customcnn_kitti(): # "customcnn_kitti"
        self.flatten = nn.Flatten()
        self.model = nn.Sequential(
            
            # Conv layer, 2D because input is image matrix
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3), # 64x365x1220
            nn.MaxPool2d(kernel_size=2, stride=2), # 64x182x610
            
            nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1), # 192x182x610
            nn.MaxPool2d(kernel_size=2, stride=2), # 192x91x305
            
            nn.Conv2d(192, 128, kernel_size=1, stride=1), # 128x91x305
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 256x91x305
            nn.Conv2d(256, 256, kernel_size=1, stride=1), # 256x91x305
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 512x91x305
            nn.MaxPool2d(kernel_size=2, stride=2), # 512x45x152
            
            nn.Conv2d(512, 256, kernel_size=1, stride=1), # 256x45x152
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 512x45x152
            nn.Conv2d(512, 256, kernel_size=1, stride=1), # 256x45x152
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 512x45x152
            nn.Conv2d(512, 256, kernel_size=1, stride=1), # 256x45x152
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 512x45x152
            nn.Conv2d(512, 256, kernel_size=1, stride=1), # 256x45x152
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 512x45x152
            
            nn.Conv2d(512, 512, kernel_size=1, stride=1), # 512x45x152
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), # 1024x45x152
            nn.Conv2d(1024, 512, kernel_size=1, stride=1), # 512x45x152
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), # 1024x45x152
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), # 1024x45x152
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1), # 1024x23x76
            
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), # 1024x23x76
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1), # 1024x11x38
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1), # 1024x6x19
        )
    
        # for KITTI object detection, out should be [batch_size, num_objects_Detected, 5]
        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(6*19*1024, 2052), # was 6*19*2*4*5 ... 6x19 is the grid placed on the image
            # nn.Linear(4560, 6*19*2*4*5) # (2 bounding boxes per cell * 4 coordinates and 1 confidence score) + 4 number of classes = 14
        )        
            
    def forward(self, data):
        '''
        Forward pass of the model. Modify as other models are added
        
        Args:
            data
            
        Returns:
            output of model
        '''
        
        # Custom linear model
        if self.model_type == "linear":
            out = self.model(data.view(data.size(0),-1))
            return out.reshape(self.batch_size,self.linear_output_dim)
        
        # Custom CNN model
        elif self.model_type == "customcnn":
            out = self.model(data)
            out = out.view(out.size(0), -1)
            out = self.fully_connected(out)
            return out
        
        # Custom model for kitti (copied from YOLO paper)
        elif self.model_type == "customcnn_kitti":
            out = self.model(data)
            out = out.view(out.size(0), -1)
            out = self.fully_connected(out)
            return out
        
    def count_params(Self, model):
        # Print the number of training parameters in the model
        num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"This model has {num_param} parameters")