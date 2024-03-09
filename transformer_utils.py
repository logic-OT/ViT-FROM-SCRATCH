import torch


class pos_encoding(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
    '''
    Here, I implemented the positional encoder the from attention is all you need paper:
    which is the sine and cosine position encoder
    given by:
    PE(pos,2i) = sin(pos/1000**(2*i)/dmodel
    PE(pos,2i) = cos(pos/1000**(2*i)/dmodel
    where pos = position of a vector/token
    '''
    def forward(self, x):
        batch_size, max_seq_length, dmodel = x.shape
        pe = torch.zeros_like(x) #position encoding matrix

        # Compute the positional encoding values
        for pos in range(max_seq_length):
            for i in range(0, dmodel):
                if i % 2 == 0:
                    pe[:, pos, i] = torch.math.sin(pos / (10000 ** (2 * i / dmodel)))
                else:
                    pe[:, pos, i] = torch.math.cos(pos / (10000 ** (2 * i / dmodel)))

        x = x + pe
        return x

class single_head_attention(torch.nn.Module):
        """
        Initializes a Single-Head Attention module. This module computes the scaled dot-product attention
        over the inputs, and is a simplified version of the attention mechanism used in the Transformer model.
        It applies linear transformations to the input to create queries (Q), keys (K), and values (V), and then
        uses these to compute attention scores.

        Parameters:
            shape (tuple): A tuple (seq_length, dmodel) indicating the shape of the input tensor,
                           where `seq_length` is the sequence length of the input, and `dmodel` is the
                           dimensionality of the input feature vectors.

        The forward pass computes the attention scores based on the queries and keys, applies these scores to the values,
        and returns the weighted sum along with the original input added (residual connection). This module also returns
        the attention scores for potential use in visualization or further analysis.
        """
        def __init__(self,shape:tuple):
                super().__init__()
                self.seq_length,self.dmodel = shape
                self.QW = torch.nn.Linear(self.dmodel,self.dmodel)
                self.KW = torch.nn.Linear(self.dmodel,self.dmodel)
                self.VW =  torch.nn.Linear(self.dmodel,self.dmodel)
                self.softmax = torch.nn.Softmax(dim=-1)

        def forward(self,x):
                Q = self.QW(x)
                K = self.KW(x)
                V = self.VW(x)

                # Calculate attention values by the dot product of Q and K => (Q.K)
                atten_values = torch.matmul(Q,K.transpose(-2, -1))/torch.sqrt(torch.tensor(self.dmodel))

                # Final linear layer(apply attention mask to V) => (Q.K).V
                final_tensor = torch.matmul(self.softmax(atten_values),V)

                return (final_tensor + x, self.softmax(atten_values)) #residual connection.
                #I return both softmax(QxK) and the final tensor. softmax(QxK) is for the attention visualisation in the latter parts of the notebook

    
class multi_head_attention(torch.nn.Module):
        def __init__(self,no_of_heads: int ,shape: tuple):
                '''
        Initializes a Multi-Head Attention module as described in the "Attention is all you need" paper. 
        This module splits the input into multiple heads to allow the model to jointly attend to information 
        from different representation subspaces at different positions. After attention is applied independently 
        on each head, the module concatenates and linearly transforms the results.

        Parameters:
            no_of_heads (int): Number of attention heads.
            shape (tuple): A tuple (seq_length, dmodel) where `seq_length` is the length of the input sequence,
                           and `dmodel` is the dimensionality of the input feature space.

        The forward pass computes the multi-head attention for input `x` and returns the transformed output.
                '''
                super().__init__()
                self.h = no_of_heads
                self.seq_length,self.dmodel = shape
                self.dk = self.dmodel//self.h
                self.softmax = torch.nn.Softmax(dim=-1)
                self.mQW = torch.nn.ModuleList([torch.nn.Linear(self.dmodel,self.dk) for i in range(self.h)])
                self.mKW = torch.nn.ModuleList([torch.nn.Linear(self.dmodel,self.dk) for i in range(self.h)])
                self.mVW = torch.nn.ModuleList([torch.nn.Linear(self.dmodel,self.dk) for i in range(self.h)])
                self.output_linear = torch.nn.Linear(self.dmodel,self.dmodel)

        def forward(self, x):
            heads = []
            for i in range(self.h):
                # Apply linear projections in batch from dmodel => h x d_k
                q = self.mQW[i](x)
                k = self.mKW[i](x)
                v = self.mVW[i](x)


                # Calculate attention using the projected vectors q, k, and v
                scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.dk, dtype=torch.float32))
                attn = self.softmax(scores)
                head_i = torch.matmul(attn, v)

                heads.append(head_i)

            # Concatenate all the heads together
            multi_head = torch.cat(heads, dim=-1)
            # Final linear layer
            output = self.output_linear(multi_head)

            return output + x  # Residual connection

            

class encoder_layer(torch.nn.Module):
    def __init__(self,shape: tuple,multi_head: bool = False, no_of_heads:int = 8):
        '''
        Implementation of Transformer Encoder
        Parameters:
            shape (tuple): The shape (H, W) of the input tensor
            multi_head (bool): Whether to use multi-head attention or not
            no_of_heads (int): Use this o set the number of heads if multi_head=True

        Returns:
            Tensor: The output of the encoder layer after applying attention, feedforward network, and normalization. 
        '''
        super().__init__()

        self.ismulti_head = multi_head
        self.max_seq_length,self.dmodel = shape
        def ff_weights():
            layer1 =  torch.nn.Linear(self.dmodel,200)
            layer2 = torch.nn.Linear(200,200)
            layer3 = torch.nn.Linear(200,self.dmodel)
            return layer1,layer2,layer3

        self.no_of_heads = no_of_heads
        self.single_head= single_head_attention(shape=shape)

        if self.ismulti_head == True:
            self.multi_head =  multi_head_attention(no_of_heads=no_of_heads,
                                                    shape=(self.dmodel,self.max_seq_length))

        self.layer1,self.layer2,self.layer3 = ff_weights()
        self.softmax = torch.nn.Softmax(dim=-1)
        self.layerNorm = torch.nn.LayerNorm(shape)
        self.relu1 = torch.nn.ReLU()
        self.relu2 = torch.nn.ReLU()

    def feed_forward(self,x):
        f = self.layer1(x)
        f = self.relu1(f)
        f = self.layer2(f)
        f = self.relu2(f)
        f = self.layer3(f)

        return self.layerNorm(f  + x) #residual connection

    def forward(self,x):
        if self.ismulti_head == True:
            x = self.multi_head(x)
        else:
            x,atten = self.single_head(x)
        x = self.layerNorm(x)
        x = self.feed_forward(x)
        x = self.layerNorm(x)
        return x
