# I modified the parameters of decoder so that it accommodates a 128*128 output from the decoder;
# Other important modifications I made are in /run/train_params.yml
#!/bin/python
#-----------------------------------------------------------------------------
#----------------------------------------------------------------------------- 
import sys
import os

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory and append it to sys.path
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

# Now you can import the function from the decolle package
from decolle.base_model import *
from decolle.lenet_decolle_model import LenetDECOLLE
 
from collections import OrderedDict

import pdb

class SpikingLenetEncoder(LenetDECOLLE):
    """
        Decolle Spiking Encoder portion of Hybrid VAE
    """
    def build_conv_stack(self, Nhid, feature_height, feature_width, pool_size, kernel_size, stride, out_channels):
        """
            build Decolle convolution layers
             
            
        """
        output_shape = None
        padding = (np.array(kernel_size) - 1) // 2          
        for i in range(self.num_conv_layers):
            feature_height, feature_width = get_output_shape(
                [feature_height, feature_width], 
                kernel_size = kernel_size[i],
                stride = stride[i],
                padding = padding[i],
                dilation = 1)
            feature_height //= pool_size[i] 
            feature_width //= pool_size[i]
            base_layer = nn.Conv2d(Nhid[i], Nhid[i + 1], kernel_size[i], stride[i], padding[i])
            layer = self.lif_layer_type[i](base_layer,
                             alpha=self.alpha[i],
                             beta=self.beta[i],
                             alpharp=self.alpharp[i],
                             deltat=self.deltat,
                             do_detach= True if self.method == 'rtrl' else False)
            pool = nn.MaxPool2d(kernel_size=pool_size[i])
            self.LIF_layers.append(layer)
            self.pool_layers.append(pool)
        return (Nhid[-1],feature_height, feature_width)

    def build_mlp_stack(self, Mhid, out_channels): 
        output_shape = None
        if self.with_output_layer:
            Mhid += [out_channels]
            self.num_mlp_layers += 1
            self.num_layers += 1
        for i in range(self.num_mlp_layers):
            base_layer = nn.Linear(Mhid[i], Mhid[i+1])
            layer = self.lif_layer_type[i+self.num_conv_layers](base_layer,
                         alpha=self.alpha[i],
                         beta=self.beta[i],
                         alpharp=self.alpharp[i],
                         deltat=self.deltat,
                         do_detach=True if self.method == 'rtrl' else False)
            output_shape = Mhid[i+1]

            self.LIF_layers.append(layer)
            self.pool_layers.append(nn.Sequential())
        return (output_shape,)

    def build_output_layer(self, Mhid, out_channels):
        i = self.num_mlp_layers
        base_layer = nn.Linear(Mhid[i], out_channels)
        layer = self.lif_layer_type[-1](base_layer,
                     alpha=self.alpha[i],
                     beta=self.beta[i],
                     alpharp=self.alpharp[i],
                     deltat=self.deltat,
                     do_detach=True if self.method == 'rtrl' else False)
        output_shape = out_channels
        return (output_shape,)

    def step(self, input, *args, **kwargs):
        s_out = []
        r_out = []
        u_out = []
        i = 0
        for lif, pool in zip(self.LIF_layers, self.pool_layers):
            if i == self.num_conv_layers: 
                input = input.view(input.size(0), -1)
            s, u = lif(input)
            u_p = pool(u)
            if i+1 == self.num_layers and self.with_output_layer:
                s_ = sigmoid(u_p)
                sd_ = u_p
            else:
                s_ = lif.sg_function(u_p)

            s_out.append(s_) 
            u_out.append(u_p)
            input = s_.detach() if lif.do_detach else s_
            i+=1
        return s_out, r_out, u_out

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)
    
    
class CLS_SQ(nn.Module):
    def __init__(self,encoder_params):
        super(CLS_SQ,self).__init__()

        self.cls_sq = OrderedDict([])

        for i,size in enumerate(encoder_params['cls_sq_layers'][:-1]):
            if i == 0:
                self.cls_sq[f'lin{i}'] = nn.Linear(encoder_params['num_classes'],size)
            else:
                self.cls_sq[f'lin{i}'] = nn.Linear(encoder_params['cls_sq_layers'][i-1],size)
            self.cls_sq[f'norm{i}'] = nn.BatchNorm1d(size)
            self.cls_sq[f'relu{i}'] = nn.LeakyReLU(negative_slope=0.2,inplace=True)

        self.cls_sq[f'lin{i+1}'] = nn.Linear(encoder_params['cls_sq_layers'][-1],encoder_params['out_channels'])
            #('lin4', nn.Linear(layer_size3, encoder_params['num_classes']))

        self.cls_sq = nn.Sequential(self.cls_sq)

        # init model weights
        for l in self.cls_sq:
            if isinstance(l, nn.Linear):
                torch.nn.init.kaiming_uniform_(l.weight, nonlinearity='leaky_relu')
                
    def forward(self, z):
        for layer in self.cls_sq:
            z = layer(z)

        return z
    
    
    def custom_adamax(self):
        pass
        

class VAE(nn.Module):
    def __init__(self, input_shape, ngf=16, out_features=128, seq_len=300, dimz=64, encoder_params={}):
        super(VAE, self).__init__()

        self.input_shape = input_shape
        self.seq_len = seq_len
        self.dimz = dimz
        self.num_classes = encoder_params['num_classes']
        self.device = encoder_params['device']

        self.encoder = SpikingLenetEncoder( 
                            out_channels= out_features,
                            Nhid=encoder_params['Nhid'],
                            Mhid=encoder_params['Mhid'],
                            kernel_size=encoder_params['kernel_size'],
                            pool_size=encoder_params['pool_size'],
                            input_shape=encoder_params['input_shape'],
                            alpha=encoder_params['alpha'],
                            alpharp=encoder_params['alpharp'],
                            dropout=encoder_params['dropout'],
                            beta=encoder_params['beta'],
                            num_conv_layers=encoder_params['num_conv_layers'],
                            num_mlp_layers=encoder_params['num_mlp_layers'],
                            lif_layer_type = LIFLayer,
                            method='bptt',
                            with_output_layer=True)

        self.encoder_head = nn.ModuleDict({'mu':nn.Linear(out_features, self.dimz), 
                                       'logvar':nn.Linear(out_features, self.dimz)})
        
        # for 128x128
        self.decoder = nn.Sequential(
            nn.Linear(self.dimz, out_features),
            Reshape(-1,out_features,1,1),
            nn.ConvTranspose2d(out_features, ngf * 8, 8, 4, 0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 4, 0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 2, 2, 0, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2,     2, 2, 2, 0, bias=False),
            nn.ReLU())
        
        self.dimz=dimz
        self.init_parameters(self.seq_len, self.input_shape)
        
        self.cls_sq = CLS_SQ(encoder_params).to(self.device)
        

    def init_parameters(self, seq_len, input_shape):
        self.encoder_head['logvar'].weight.data[:] *= 1e-16
        self.encoder_head['logvar'].bias.data[:] *= 1e-16 
        # self.encoder_head['mu'].weight.data[:] *= 1e-16
        # self.encoder_head['mu'].bias.data[:] *= 1e-16 
        return 
    
    def encoder_forward(self, x):
        return self.encoder(x)

    def encode(self, x):
        s = self.encoder(x)[0]
        h1 = torch.nn.functional.leaky_relu(s)                   
        return self.encoder_head['mu'](h1), self.encoder_head['logvar'](h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)# - 1
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        return self.decoder(z)
    
    def excite_z(self,z,num_classes=16):
        exc_z = torch.zeros((z.shape[0],num_classes))
        for i in range(z.shape[0]):
            exc_z[i] = z[i,:num_classes]#[t[i]]
        
        return exc_z

    def forward(self, x):
        mu, logvar = self.encode(x)
        
        z = self.reparameterize(mu, logvar)

        excite_z = self.excite_z(z,self.num_classes).to(self.device)

        clas = self.cls_sq(excite_z)

        return self.decode(z), mu, logvar, clas

    
class CustomLIFLayer(LIFLayer):
    sg_function = torch.sigmoid
        
    
class SpikeClassifier(nn.Module):
    def __init__(self, input_shape, ngf=16, out_features=128, seq_len=300, dimz=32, encoder_params={}, burnin=0):
        super(SpikeClassifier, self).__init__()

        self.input_shape = input_shape
        self.seq_len = seq_len
        self.dimz = dimz
        
        self.classifier = LenetDECOLLE( 
                            out_channels= out_features,
                            Nhid=encoder_params['Nhid'],
                            Mhid=encoder_params['Mhid'],
                            kernel_size=encoder_params['kernel_size'],
                            pool_size=encoder_params['pool_size'],
                            input_shape=encoder_params['input_shape'],
                            alpha=encoder_params['alpha'],
                            alpharp=encoder_params['alpharp'],
                            dropout=encoder_params['dropout'],
                            beta=encoder_params['beta'],
                            num_conv_layers=encoder_params['num_conv_layers'],
                            num_mlp_layers=encoder_params['num_mlp_layers'],
                            lif_layer_type = LIFLayer,
                            method='bptt',
                            with_output_layer=True,
                            burnin=burnin)

