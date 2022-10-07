# Copyright 2022 VMware, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
@author: Shay Vargaftik (VMware Research), shayv@vmware.com; vargaftik@gmail.com
@author: Yaniv Ben-Itzhak (VMware Research), ybenitzhak@vmware.com; yaniv.benizhak@gmail.com
"""

"""EdenPipeline module."""

"""
EDEN is an unbiased lossy compression method that uses a random rotation followed by deterministic quantization and scaling. 
EDEN provides strong theoretical guarantees, as described in the following ICML 2022 paper: 

"EDEN: Communication-Efficient and Robust Distributed Mean Estimation for Federated Learning" 
Shay Vargaftik, Ran Ben Basat, Amit Portnoy, Gal Mendelson, Yaniv Ben Itzhak, Michael Mitzenmacher, 
Proceedings of the 39th International Conference on Machine Learning, PMLR 162:21984-22014, 2022.

https://proceedings.mlr.press/v162/vargaftik22a.html

------------------------------------------------------------------------------------------------------------------------

In order to use and configure EDEN use the following lines in plan.yaml:
 
compression_pipeline :
  defaults : plan/defaults/compression_pipeline.yaml
  template : openfl.pipelines.EdenPipeline
  settings :
    n_bits : <number of bits per coordinate>
    device: <cpu|cuda:0|cuda:1|...>
    dim_threshold: 1000 #EDEN compresses layers that their dimension is above the dim_threshold, use 1000 as default
"""

import torch
import copy as co
import numpy as np

from .pipeline import TransformationPipeline
from .pipeline import Transformer
from .pipeline import Float32NumpyArrayToBytes

class Eden:
    
    
    def __init__(self, nbits=8, device='cpu'):
        
        def gen_normal_centroids_and_boundaries(device):
            
            ### half-normal lloyd-max centroids
            centroids = {}
            centroids[1] = [0.7978845608028654]
            centroids[2] = [0.4527800398860679, 1.5104176087114887]
            centroids[3] = [0.24509416307340598, 0.7560052489539643, 1.3439092613750225, 2.151945669890335]
            centroids[4] = [0.12839501671105813, 0.38804823445328507, 0.6567589957631145, 0.9423402689122875, 1.2562309480263467, 1.6180460517130526, 2.069016730231837, 2.732588804065177]
            centroids[5] = [0.06588962234909321, 0.1980516892038791, 0.3313780514298761, 0.4666991751197207, 0.6049331689395434, 0.7471351317890572, 0.89456439585444, 1.0487823813655852, 1.2118032120324, 1.3863389353626248, 1.576226389073775, 1.7872312118858462, 2.0287259913633036, 2.3177364021261493, 2.69111557955431, 3.260726295605043]
            centroids[6] = [0.0334094558802581, 0.1002781217139195, 0.16729660990171974, 0.23456656976873475, 0.3021922894403614, 0.37028193328115516, 0.4389488009177737, 0.5083127587538033, 0.5785018460645791, 0.6496542452315348, 0.7219204720694183, 0.7954660529025513, 0.870474868055092, 0.9471530930156288, 1.0257343133937524, 1.1064859596918581, 1.1897175711327463, 1.2757916223519965, 1.3651378971823598, 1.458272959944728, 1.5558274659528346, 1.6585847114298427, 1.7675371481292605, 1.8839718992293555, 2.009604894545278, 2.146803022259123, 2.2989727412973995, 2.471294740528467, 2.6722617014102585, 2.91739146530985, 3.2404166403241677, 3.7440690236964755]
            centroids[7] = [0.016828143177728235, 0.05049075396896167, 0.08417241989671888, 0.11788596825032507, 0.1516442630131618, 0.18546025708680833, 0.21934708340331643, 0.25331807190834565, 0.2873868062260947, 0.32156710392315796, 0.355873075050329, 0.39031926330596733, 0.4249205523979007, 0.4596922300454219, 0.49465018161031576, 0.5298108436256188, 0.565191195643323, 0.600808970989236, 0.6366826613981411, 0.6728315674936343, 0.7092759460939766, 0.746037126679468, 0.7831375375631398, 0.8206007832455021, 0.858451939611374, 0.896717615963322, 0.9354260757626341, 0.9746074842160436, 1.0142940678300427, 1.054520418037026, 1.0953237719213182, 1.1367442623434032, 1.1788252655205043, 1.2216138763870124, 1.26516137869917, 1.309523700469555, 1.3547621051156036, 1.4009441065262136, 1.448144252238147, 1.4964451375010575, 1.5459387008934842, 1.596727786313424, 1.6489283062238074, 1.7026711624156725, 1.7581051606756466, 1.8154009933798645, 1.8747553268072956, 1.9363967204122827, 2.0005932433837565, 2.0676621538384503, 2.1379832427349696, 2.212016460501213, 2.2903268704925304, 2.3736203164211713, 2.4627959084523208, 2.5590234991374485, 2.663867022558051, 2.7794919110540777, 2.909021527386642, 3.0572161028423737, 3.231896182843021, 3.4473810105937095, 3.7348571053691555, 4.1895219330235225]
            centroids[8] = [0.008445974137017219, 0.025338726226901278, 0.042233889994651476, 0.05913307399220878, 0.07603788791797023, 0.09294994306815242, 0.10987089037069565, 0.12680234584461386, 0.1437459285205906, 0.16070326074968388, 0.1776760066764216, 0.19466583496246115, 0.21167441946986007, 0.22870343946322488, 0.24575458029044564, 0.2628295721769575, 0.2799301528634766, 0.29705806782573063, 0.3142150709211129, 0.3314029639954903, 0.34862355883476864, 0.3658786774238477, 0.3831701926964899, 0.40049998943716425, 0.4178699650069057, 0.4352820704086704, 0.45273827097956804, 0.4702405882876, 0.48779106011037887, 0.505391740756901, 0.5230447441905988, 0.5407522460590347, 0.558516486141511, 0.5763396823538222, 0.5942241184949506, 0.6121721459546814, 0.6301861414640443, 0.6482685527755422, 0.6664219019236218, 0.684648787627676, 0.7029517931200633, 0.7213336286470308, 0.7397970881081071, 0.7583450032075904, 0.7769802937007926, 0.7957059197645721, 0.8145249861674053, 0.8334407494351099, 0.8524564651728141, 0.8715754936480047, 0.8908013031010308, 0.9101374749919184, 0.9295877653215154, 0.9491559977740125, 0.9688461234581733, 0.9886622867721733, 1.0086087121824747, 1.028689768268861, 1.0489101021225093, 1.0692743940997251, 1.0897875553561465, 1.1104547388972044, 1.1312812154370708, 1.1522725891384287, 1.173434599389649, 1.1947731980672593, 1.2162947131430126, 1.238005717146854, 1.2599130381874064, 1.2820237696510286, 1.304345369166531, 1.3268857708606756, 1.349653145284911, 1.3726560932224416, 1.3959037693197867, 1.419405726021264, 1.4431719292973744, 1.4672129964566984, 1.4915401336751468, 1.5161650628244996, 1.541100284490976, 1.5663591473033147, 1.5919556551358922, 1.6179046397057497, 1.6442219553485078, 1.6709244249695359, 1.6980300628044107, 1.7255580190748743, 1.7535288357430767, 1.7819645728459763, 1.81088895442524, 1.8403273195729115, 1.870306964218662, 1.9008577747790962, 1.9320118435829472, 1.9638039107009146, 1.9962716117712092, 2.0294560760505993, 2.0634026367482017, 2.0981611002741527, 2.133785932225919, 2.170336784741086, 2.2078803102947337, 2.2464908293749546, 2.286250990303635, 2.327254033532845, 2.369604977942217, 2.4134218838650208, 2.458840003415269, 2.506014300608167, 2.5551242195294983, 2.6063787537827645, 2.660023038604595, 2.716347847697055, 2.7757011083910723, 2.838504606698991, 2.9052776685316117, 2.976670770545963, 3.0535115393558603, 3.136880130166507, 3.2282236667414654, 3.3295406612081644, 3.443713971315384, 3.5751595986789093, 3.7311414987004117, 3.9249650523739246, 4.185630113705256, 4.601871059539151]
                           
            ### normal centroids
            for i in centroids:
                centroids[i] = torch.Tensor([-j for j in centroids[i][::-1]] + centroids[i]).to(device)
            
            ### centroids to bin boundaries
            def gen_boundaries(centroids):
                return [(a + b) / 2 for a, b in zip(centroids[:-1], centroids[1:])]
        
            ### bin boundaries
            boundaries = {}
            for i in centroids:
                boundaries[i] = torch.Tensor(gen_boundaries(centroids[i])).to(device)
                    
            return centroids, boundaries

        # device ('cpu' or 'cuda')
        self.device = device
        
        # generate metadata for compression
        self.centroids, self.boundaries = gen_normal_centroids_and_boundaries(self.device)

        # bits per coordinate
        if nbits not in [1,2,3,4,5,6,7,8]:
            raise Exception("nbits value is not supported")
        self.nbits = int(nbits)
        
        # shared randomness
        self.sender_prng = torch.Generator(device=device)
        self.receiver_prng = torch.Generator(device=device)

        # number of hadamard transforms to employ
        self.num_hadamard = 2 # TODO: add as a configurable parameter to YAML? 2 is default and always works, but 1 can be used for non-adversarial inputs and thus run faster
        
        
    def hadamard(self, vec):
        
        d = vec.numel()
        if d & (d-1) != 0:
            raise Exception("input numel must be a power of 2")
          
        h = 2
        while h <= d:        
            hf = h//2
            vec = vec.view(d//h,h)
            vec[:,:hf]  = vec[:,:hf] + vec[:,hf:2*hf]
            vec[:,hf:2*hf] = vec[:,:hf] - 2*vec[:,hf:2*hf]
            h *= 2   
        vec /= np.sqrt(d)
        
        return vec.view(-1)


    # randomized Hadamard transform
    def rht(self, vec, seed):
        
        self.sender_prng.manual_seed(seed)
        
        vec = vec * (2 * torch.bernoulli(torch.ones(vec.numel(), device=vec.device) / 2, generator=self.sender_prng) - 1)
        vec = self.hadamard(vec)
        
        return vec

    # inverse randomized Hadamard transform
    def irht(self, vec, seed):
        
        self.receiver_prng.manual_seed(seed)
        
        vec = self.hadamard(vec)
        vec = vec * (2 * torch.bernoulli(torch.ones(vec.numel(), device=vec.device) / 2, generator=self.receiver_prng) - 1)
        
        return vec    
    
    
    def quantize(self, vec):
        
        vec_norm = torch.norm(vec,2)
        
        if vec_norm > 0:
        
            bins = torch.bucketize(vec * (vec.numel()**0.5) / vec_norm, self.boundaries[self.nbits])
            scale = vec_norm**2 / torch.dot(torch.take(self.centroids[self.nbits], bins), vec)
            
            if not torch.isnan(scale):
                return bins, scale              
        
        return torch.zeros(vec.numel(), device=vec.device), torch.tensor([0])


    def compress(self, vec, seed):
        
        vec = torch.Tensor(vec.flatten()).to(self.device)   
        dim = vec.numel()

        if not dim & (dim - 1) == 0 or dim < 8:
            
            padded_dim = max(int(2**(np.ceil(np.log2(dim)))), 8)
            padded_vec = torch.zeros(padded_dim, device=self.device)
            padded_vec[:dim] = vec
            
            vec = self.rht(padded_vec, seed)
            for i in range(1, self.num_hadamard):
                vec = self.rht(vec, seed+i)
                
        else:
            
            for i in range(self.num_hadamard):
                vec = self.rht(vec, seed+i)
            
        bins, scale = self.quantize(vec)
        bins = self.toBits(bins)

        return bins.cpu().numpy(), float(scale.cpu().numpy()), dim
    

    def decompress(self, bins, scale, dim, seed):
        
        bins = self.fromBits(torch.Tensor(bins).to(self.device)).long()
        vec = torch.take(self.centroids[self.nbits], bins)
        
        for i in range(self.num_hadamard):
            vec = self.irht(vec, int(seed+(self.num_hadamard-1)-i))
        
        return (scale * vec)[:int(dim)].cpu().numpy()
 
    
    # packing the quantization values to bytes
    def toBits(self, intBoolVec):
        
        def toBits_h(ibv):
            
            n = ibv.numel()  
            ibv = ibv.view(n // 8, 8).int()
            
            bv = torch.zeros(n // 8, dtype=torch.uint8, device=ibv.device)
            for i in range(8):
                bv += ibv[:, i] * (2**i)

            return bv
        
        Lunit =  intBoolVec.numel() // 8       
        bitVec = torch.zeros(Lunit * self.nbits, dtype=torch.uint8)
        
        for i in range(self.nbits):
            bitVec[Lunit*i:Lunit*(i+1)] = toBits_h((intBoolVec % 2 != 0).int())
            intBoolVec = torch.div(intBoolVec, 2, rounding_mode='floor')
        
        return bitVec
    

    # unpacking bytes to quantization values       
    def fromBits(self, bitVec):
    
        def fromBits_h(bv, device):
            
            n = bv.numel()
            
            iv = torch.zeros((8, n)).to(device)
            for i in range(8):
                temp = bv.clone()
                iv[i] = (torch.div(temp, 2 ** i, rounding_mode='floor') % 2 != 0).int()

            return iv.T.reshape(-1)  
        
        bitVec = bitVec.view(self.nbits, bitVec.numel() // self.nbits).to(self.device)
        intVecs = torch.zeros(bitVec.numel() // self.nbits * 8).to(self.device)
        
        for i in range(self.nbits):
            intVecs += 2**i * fromBits_h(bitVec[i], self.device)
        
        return intVecs.reshape(1, -1) 

    
class EdenTransformer(Transformer):
    """Eden transformer class to quantize input data."""

    def __init__(self, n_bits=8, dim_threshold=100, device='cpu'):
        """Class initializer.
        """
        self.lossy = True
        self.eden = Eden(nbits=n_bits, device=device)

        print('*** Using EdenTransformer with params: {} bits, dim_threshold: {}, {} device ***'.format(n_bits, dim_threshold, device))
        
        self.dim_threshold = dim_threshold
        self.no_comp = Float32NumpyArrayToBytes()
        

    def forward(self, data, **kwargs):
        """
        Quantize data.
        """
        
        seed = (hash(sum(data.flatten())*13+7)+np.random.randint(1,2**16))%(2**16) #TODO: can be simplified if have access to a unique feature of the participant (e.g., ID)
        seed = int(float(seed))
        metadata = {'int_list': list(data.shape)}
        
        if data.size > self.dim_threshold:
            int_array, scale, dim = self.eden.compress(data, seed)
            metadata['int_to_float'] = {0 : scale, 1 : float(dim), 2 : float(seed)} #TODO: workaround: using the int to float dictionary to pass eden's metadata

            return_values = int_array.astype(np.uint8).tobytes(), metadata

        else:

            no_comp_data, metadata = self.no_comp.forward(data)
            return_values =  no_comp_data, metadata

        return return_values


    def backward(self, data, metadata, **kwargs):
        """Recover data array back to the original numerical type and the shape.

        Args:
            data: an flattened numpy array
            metadata: dictionary to contain information for recovering to original data array

        Returns:
            data: Numpy array with original numerical type and shape
        """
        
        if np.prod(metadata['int_list']) >= self.dim_threshold: # compressed data
            data = np.frombuffer(data, dtype=np.uint8)
            data = co.deepcopy(data)
            data = self.eden.decompress(data,
                                    metadata['int_to_float'][0], 
                                    metadata['int_to_float'][1], 
                                    metadata['int_to_float'][2])
            data_shape = list(metadata['int_list'])
            data = data.reshape(data_shape)
        else:
            data = self.no_comp.backward(data, metadata)
        data = data.astype(np.float32)

        return data


class EdenPipeline(TransformationPipeline):
    """A pipeline class to compress data lossy using EDEN."""

    def __init__(self, n_bits=8, dim_threshold=100, device='cpu', **kwargs):
        """Initialize a pipeline of transformers.

        Args:
            n_bits (int): Number of bits per coordinate (1-8 bits are supported)
            dim_threshold (int): Layers with less than dim_threshold params are not compressed
            device: Device for executing the compression and decompression (e.g., 'cpu', 'cuda:0', 'cuda:1')

        Return:
            Transformer class object
        """

        # instantiate each transformer
        transformers = [EdenTransformer(n_bits, dim_threshold, device)]
        super(EdenPipeline, self).__init__(transformers=transformers, **kwargs)
