class GMAT_Net(nn.Module):
    def __init__(self, 
                 model_type, 
                 num_nodes, 
                 device, 
                 predefined_A=None, 
                 dropout=0.3, 
                 conv_channels=32, 
                 residual_channels=32, 
                 skip_channels=64, 
                 end_channels=128, 
                 seq_length=12, 
                 in_dim=2, 
                 out_dim=12, 
                 layers=3, 
                 layer_norm_affline=True):
        super(GMAT_Net, self).__init__()

        self.model_type = model_type

        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_A = predefined_A
        self.layers = layers
        self.seq_length = seq_length

        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        
        self.a_gmat_list = nn.ModuleList()
        in_channel = 32
        n_heads = 8
        dropout = 0
        alpha = 0.2
        self.a_gmat_list.append(
            A_GMAT_module(
              n_heads=n_heads, in_channel= in_channel, num_nodes=num_nodes, mlp=[n_heads],mlp2=[32], dropout=dropout, alpha=alpha
            )
        )

        self.t_gmat_list_1 = nn.ModuleList()
        self.t_gmat_list_2 = nn.ModuleList()

        self.s_gmat_list = nn.ModuleList() # dual GMAT Blocks
        
        # Modified from: https://github.com/nnzhan/MTGNN
        kernel_size = 7
        dilation_exponential = 1
        if dilation_exponential>1:
            self.receptive_field = int(1+(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            self.receptive_field = layers*(kernel_size-1) + 1
        
        print("# Model Type", self.model_type)
        print("# receptive_field", self.receptive_field)
        self.receptive_field = 13
        i=0
        if dilation_exponential>1:
            rf_size_i = int(1 + i*(kernel_size-1)*(dilation_exponential**layers-1)/(dilation_exponential-1))
        else:
            rf_size_i = i*layers*(kernel_size-1)+1
        new_dilation = 1
        target_len = self.receptive_field

        for j in range(1,layers+1):
           
            if dilation_exponential > 1:
                rf_size_j = int(rf_size_i + (kernel_size-1)*(dilation_exponential**j-1)/(dilation_exponential-1))
            else:
                rf_size_j = rf_size_i+j*(kernel_size-1)

            dilation_factor = 1
            kern = 5

            in_channel = 32
            n_heads = 8
            dropout = 0
            alpha = 0.2
            self.t_gmat_list_1.append(
                T_GMAT_module(
                  kern= kern, dilation_factor=dilation_factor, temporal_len = target_len, n_heads=n_heads, in_channel= in_channel, num_nodes=num_nodes, mlp=[n_heads],mlp2=[32], dropout=dropout, alpha=alpha
                )
            )
            
            self.t_gmat_list_2.append(
                T_GMAT_module(
                  kern= kern, dilation_factor=dilation_factor, temporal_len = target_len, n_heads=n_heads, in_channel= in_channel, num_nodes=num_nodes, mlp=[n_heads],mlp2=[32], dropout=dropout, alpha=alpha
                )
            )
            
            target_len -= 4

            in_channel = 32
            n_heads = 8
            dropout = 0
            alpha = 0.2
            
            depth = 2
            self.s_gmat_list.append(
                S_GMAT_module(
                  depth=depth, temporal_len = target_len, n_heads=n_heads, in_channel= in_channel, num_nodes=num_nodes, mlp=[n_heads],mlp2=[32], dropout=dropout, alpha=alpha
                )
            )
            # 1x1 convolution for skip connection
            self.skip_convs.append(nn.Conv2d(in_channels=conv_channels,
                                                out_channels=skip_channels,
                                                kernel_size=(1, target_len)))
            
            self.norm.append(LayerNorm((residual_channels, num_nodes, target_len),elementwise_affine=layer_norm_affline))
            
            new_dilation *= dilation_exponential
    
        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                             out_channels=end_channels,
                                             kernel_size=(1,1),
                                             bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                             out_channels=out_dim,
                                             kernel_size=(1,1),
                                             bias=True)

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length-self.receptive_field+1), bias=True)

        else:
            self.skip0 = nn.Conv2d(in_channels=in_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        self.idx = torch.arange(self.num_nodes).to(device)


    def forward(self, input, input_1,input_2,input_3,input_4, idx=None):
        seq_len = input.size(3)
        assert seq_len==self.seq_length, 'input sequence length not equal to preset sequence length'

        # Step0: 檢查receptive_field, 不足則padding0
        if self.seq_length<self.receptive_field:
            input = nn.functional.pad(input,(self.receptive_field-self.seq_length,0,0,0))
            input_1 = nn.functional.pad(input_1,(self.receptive_field-self.seq_length,0,0,0))
            input_2 = nn.functional.pad(input_2,(self.receptive_field-self.seq_length,0,0,0))
            input_4 = nn.functional.pad(input_4,(self.receptive_field-self.seq_length,0,0,0))
            
            input_3 = nn.functional.pad(input_3,(self.receptive_field-self.seq_length,0,0,0))


        # Step1: turn([64, 2, 207, 13]) to ([64, 32, 207, 13]) => 固定用同一conv
        x = self.start_conv(input) 
        x_1 = self.start_conv(input_1)  
        x_2 = self.start_conv(input_2)
        
        x_4 = self.start_conv(input_4) 
        x_3 = self.start_conv(input_3)  

        x = self.a_gmat_list[0](x,x_1,x_2,x_3,x_4)

        skip = self.skip0(F.dropout(input, self.dropout, training=self.training))
        
        for i in range(self.layers):
            
            residual = x    
            
            filter = self.t_gmat_list_1[i](x)
            filter = torch.tanh(filter)

            gate = self.t_gmat_list_2[i](x)
            gate = torch.sigmoid(gate)

            x = filter * gate
            x = F.dropout(x, self.dropout, training=self.training)

            s = x
            
            s = self.skip_convs[i](s)    

            skip = s + skip
            # Two GMAT Block of different directions implemented in S_GMAT_module 
            x = self.s_gmat_list[i](x, self.predefined_A[0], self.predefined_A[1])

            x = x + residual[:, :, :, -x.size(3):]
            x = self.norm[i](x,self.idx)
            
        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x
