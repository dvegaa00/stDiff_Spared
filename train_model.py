import numpy as np
import pandas as pd
import os
from scipy.stats import wasserstein_distance
import pandas as pd
import scanpy as sc
import warnings
from scipy.stats import spearmanr, pearsonr
from scipy.spatial import distance_matrix
from sklearn.metrics import matthews_corrcoef
import torch
from scipy.spatial.distance import cdist
import sys
from os.path import join
from IPython.display import display

from model_stDiff.stDiff_model import DiT_stDiff
from model_stDiff.stDiff_scheduler import NoiseScheduler
from model_stDiff.stDiff_train import normal_train_stDiff
from model_stDiff.sample import sample_stDiff
from process_stDiff.result_analysis import clustering_metrics

warnings.filterwarnings('ignore')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

from process_stDiff.data import *


# ******** preprocess ********
adata_spatial = sc.read_h5ad('datasets/sp/' + 'dataset2_spatial_33.h5ad')
#adata_spatial.obs["n_genes"] = genes por spot que no son cero
#adata_spatial.obs["x_coord"] y adata_spatial.obs["y_coord"] = coordenada central de los spots
#adata_spatial.var["n_cells"] = celulas por genes (coonsidera que el nombre de los genes es diferente al de Visium)

adata_seq = sc.read_h5ad('datasets/sc/'+ 'dataset2_seq_33.h5ad')
#adata_seq.obs["n_genes"] = genes por celula que no son cero
#adata_seq.var["n_cells"] = celulas por genes (coonsidera que el nombre de los genes es diferente al de Visium)

adata_seq2 = data_augment(adata_seq.copy(), True, noise_std=10)
#duplican el adata.X agregando el mismo adata mas el noise_std (no se porque)
adata_spatial2 = adata_spatial.copy()

#Podemos usar nuestro propio procesamiento
sc.pp.normalize_total(adata_seq2, target_sum=1e4)
sc.pp.log1p(adata_seq2)
adata_seq2 = scale(adata_seq2) # stDiff need
data_seq_array = adata_seq2.X

sc.pp.normalize_total(adata_spatial2, target_sum=1e4)
sc.pp.log1p(adata_spatial2)
#adata_spatial2.X = filas son los spots y columnas son los genes
#funcion scale: normaliza por spot (normalizar filas)
adata_spatial2 = scale(adata_spatial2)
#Convertimos el X a un array (se mantienen dimensiones)
data_spatial_array = adata_spatial2.X

sp_genes = np.array(adata_spatial.var_names)
sp_data = pd.DataFrame(data=data_spatial_array, columns=sp_genes)
#Dataframe con columans igual a genes (nombres de los genes) y filas son los spots (una fila corresponde a la expresion genica de un spot)
sc_data = pd.DataFrame(data=data_seq_array, columns=sp_genes)

#Training
lr = 0.00016046744893538737 
depth = 6 
num_epoch = 900 
diffusion_step = 1500 
batch_size = 2048 
hidden_size = 512 
head = 16

# mask
cell_num = data_spatial_array.shape[0]
#numero de spots
gene_num = data_spatial_array.shape[1]
#numero de genes
mask = np.ones((gene_num,), dtype='float32')
#mask de unos de shape num_genes

# gene_id_test
train_size = 0.8
gene_names_rnaseq = sp_genes 
np.random.seed(0)
n_genes = len(gene_names_rnaseq)
gene_ids_train = sorted(
    np.random.choice(range(n_genes), int(n_genes * train_size), False)
)
gene_ids_test = sorted(set(range(n_genes)) - set(gene_ids_train)) # test

#Mascara con unos menos los genes que tapan (los genes de test)
#En nuestro caso tendriamos una mascara fija (0,1,1,1,1,1,1) donde los 1 son los vecinos
mask[gene_ids_test] = 0

seq = data_seq_array
st = data_spatial_array
data_seq_masked = seq * mask
data_spatial_masked = st * mask

seq = seq * 2 - 1
# seq.shape: (11226, 33)
data_seq_masked = data_seq_masked * 2 - 1
# data_seq_masked.shape: (11226, 33)
st = st * 2 - 1
# st.shape: (3405, 33)
data_spatial_masked = data_spatial_masked * 2 - 1
# data_spatial_masked.shape: (3405, 33)

#entrada son dos arrays
#seq = adata.X
#data_seq_masked = adata.X maskeado
breakpoint()
dataloader = get_data_loader(
    seq, # all gene
    data_seq_masked, # test gene = 0
    batch_size=batch_size, 
    is_shuffle=True)

seed = 1202
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

model = DiT_stDiff(
    input_size=gene_num,  
    hidden_size=hidden_size, 
    depth=depth,
    num_heads=head,
    classes=6, 
    mlp_ratio=4.0,
    dit_type='dit'
)

device = torch.device('cuda')
model.to(device)

diffusion_step = diffusion_step

save_path_prefix = 'ckpt/demo.pt'
# train
model.train()
if not os.path.isfile(save_path_prefix):

    normal_train_stDiff(model,
                            dataloader=dataloader,
                            lr=lr,
                            num_epoch=num_epoch,
                            diffusion_step=diffusion_step,
                            device=device,
                            pred_type='noise',
                            mask=mask)

    torch.save(model.state_dict(), save_path_prefix)
else:
    model.load_state_dict(torch.load(save_path_prefix))


# sample
pdb.set_trace()
gt = data_spatial_masked
#adata.X maskeado
noise_scheduler = NoiseScheduler(
    num_timesteps=diffusion_step,
    beta_schedule='cosine'
)

dataloader = get_data_loader(
    data_spatial_masked, # test gene = 0
    data_spatial_masked, # test gene = 0
    batch_size=batch_size, 
    is_shuffle=False)


model.eval()
imputation = sample_stDiff(model,
                            device=device,
                            dataloader=dataloader,
                            noise_scheduler=noise_scheduler,
                            mask=mask,
                            gt=gt,
                            num_step=diffusion_step,
                            sample_shape=(cell_num, gene_num),
                            is_condi=True,
                            sample_intermediate=diffusion_step,
                            model_pred_type='noise',
                            is_classifier_guidance=False,
                            omega=0.2)

#sample_shape=(spots, genes)
data_spatial_masked[:, gene_ids_test] = imputation[:, gene_ids_test]

#devolverlo a los valores normales
impu = (data_spatial_masked  + 1) / 2

# ********** metrics **********
def imputation_metrics(original, imputed):
    absolute_error = np.abs(original - imputed)
    relative_error = absolute_error / np.maximum(
        np.abs(original), np.ones_like(original)
    )
    spearman_gene = []
    for g in range(imputed.shape[1]):
        if np.all(imputed[:, g] == 0):
            correlation = 0
        else:
            correlation = spearmanr(original[:, g], imputed[:, g])[0]
        spearman_gene.append(correlation)

    return {
        "median_absolute_error_per_gene": np.median(absolute_error, axis=0),
        "mean_absolute_error_per_gene": np.mean(absolute_error, axis=0),
        "mean_relative_error": np.mean(relative_error, axis=1),
        "median_relative_error": np.median(relative_error, axis=1),
        "spearman_per_gene": np.array(spearman_gene),

        # Metric we report in the GimVI paper:
        "median_spearman_per_gene": np.median(spearman_gene),
    }

tmp = imputation_metrics(np.array(data_spatial_array[:, gene_ids_test]), impu[:, gene_ids_test])
print(tmp['median_spearman_per_gene'])

#dataset2: tensor(0.8385, device='cpu')
#dataset5: 