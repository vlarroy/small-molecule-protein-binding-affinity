# %% [markdown]
# # **Data statistics and visualization**

# %%


# %%
import numpy as np
import pandas as pd
import duckdb
import math
import random
import kaleido
import dask.dataframe as dd
import py3Dmol
import pickle as pickle
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import Image
from plotly.subplots import make_subplots
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit import RDLogger

# %%
train_path_rw = 'train.parquet'
test_path_rw = 'test.parquet'

con = duckdb.connect()

# %%
n_limit = 3000000

sample = con.query(f'SELECT * FROM parquet_scan(\'{train_path_rw}\') ORDER BY random() LIMIT {n_limit}').df()

sample_bind = con.query(f""" SELECT *
                        FROM parquet_scan('{train_path_rw}')
                        WHERE binds = 1
                        ORDER BY random()
                        LIMIT {n_limit}
                        """).df()

sample_no_bind = con.query(f""" SELECT *
                        FROM parquet_scan('{train_path_rw}')
                        WHERE binds = 0
                        ORDER BY random()
                        LIMIT {n_limit}
                        """).df()

sample_test= con.query(f""" SELECT *
                        FROM parquet_scan('{test_path_rw}')
                        ORDER BY random()
                        LIMIT {n_limit}
                        """).df()

sample_balanced = pd.concat([sample_bind, sample_no_bind])

# %% [markdown]
# ## **Basic statistics**

# %%
nb_rows = con.query(f""" SELECT COUNT(*)
                    FROM parquet_scan('{train_path_rw}')
                    """).df().loc[0, "count_star()"]

print(f"Number of rows in the dataset: {nb_rows}")

# %%
nb_HSA = con.query(f"""SELECT COUNT(*)
                        FROM parquet_scan('{train_path_rw}')
                        WHERE protein_name = 'HSA'
                        """).df().loc[0,"count_star()"]

nb_BRD4 = con.query(f"""SELECT COUNT(*)
                        FROM parquet_scan('{train_path_rw}')
                        WHERE protein_name = 'BRD4'
                        """).df().loc[0,"count_star()"]

nb_sEH = con.query(f"""SELECT COUNT(*)
                        FROM parquet_scan('{train_path_rw}')
                        WHERE protein_name = 'sEH'
                        """).df().loc[0,"count_star()"]

print(f"Number of rows with HSA binding: {nb_HSA}")
print(f"Number of rows with BRD4 binding: {nb_BRD4}")
print(f"Number of rows with sEH binding: {nb_sEH}")


# %% [markdown]
# ## **Distributons**

# %%
blocks_lengths = [sample_no_bind[f'buildingblock{i+1}_smiles'].apply(lambda x: len(x)) for i in range(3)]

fig = go.Figure()

for i in range(3):
  fig.add_trace(go.Box(y=blocks_lengths[i], name=f'buildingblock{i+1}_smiles'))

fig.update_layout(title= "Distribution of length of building blocks")

fig.show()

# %% [markdown]
# ## **Balance of the datasheet**
# Balance of the datasheet by protein

# %%
# Get representation of each protein by binds count

fig = px.histogram(sample, x="protein_name", color = "binds", barmode = "group", title="Distribution of proteins by binds in log scale", log_y=True)

fig.show()

# %% [markdown]
# We can see that the distribution between 'non-binding' and 'binding' for all proteins is highly imbalanced.
# This is important for later model traning. We can try to use over/under sampling techniques, but it can be tricky because can lead to overfitting.. 

# %% [markdown]
# ## **Balance of building blocks by binding**

# %% [markdown]
# The cumulative probability refers to the probability that a smiles BB will be less than or equal to a certain value when the values are ordered.

# %%
buildingblock_nb = [sample_no_bind[f'buildingblock{i+1}_smiles'].value_counts().to_list() for i in range(3)]
buildingblock_nb += [sample_bind[f'buildingblock{i+1}_smiles'].value_counts().to_list() for i in range(3)]
cumulative_probs = []

for i in range(6):
  counts = buildingblock_nb[i]
  total_count = sum(counts)
  cumulative_sum = np.cumsum(counts)
  cumulative_prob = cumulative_sum / total_count
  cumulative_probs.append(cumulative_prob)

bindings_label =["no_binding"]*3 + ["binding"]*3
fig = go.Figure()

for i in range(6):
  fig.add_trace(go.Scatter(x=np.arange(len(cumulative_probs[i]))+1,
                           y=cumulative_probs[i], mode='lines',
                           name= f'buildingblock{(i%3)+1}_smiles_{bindings_label[i]}',
                           legendgroup = i%3
                           ))
  
fig.update_layout(title= 'Cumulative distribution of building blocks in the training set',
                    xaxis_title= 'Building block count',
                    yaxis_title = 'Cumulative probability')

fig.show()

# %% [markdown]
# For BB1 & BB2, the 'binding' conditions seem to have more concentrated distributions (steeper curves) compared to the 'non-binding' conditions. This could imply that, in the presence of binding, fewer building blocks dominate the distribution.
# 
# For BB3, both for the 'binding' and 'non-binding' conditions have more similar and flatter distributions, suggesting a more uniform spread of bb counts. 
# 
# The distinct shapes and steepness of the curves for each BB indicate that the 'binding' and 'non-binding' behaviour differs significantly across different BBs. 

# %%
buildingblock_test = [sample_test[f"buildingblock{i+1}_smiles"].value_counts().to_list() for i in range(3)]
cumulative_probs = []

for i in range(3):
    counts = buildingblock_nb[i]
    total_count = sum(counts)
    cumulative_sum = np.cumsum(counts)
    cumulative_prob = cumulative_sum / total_count
    cumulative_probs.append(cumulative_prob)
bindings_label = ["no_binding"]*3 + ["binding"]*3
fig = go.Figure()

for i in range(3):
    fig.add_trace(go.Scatter(x=np.arange(len(cumulative_probs[i])) + 1, 
                             y=cumulative_probs[i], mode='lines', 
                             name=f'buildingblock{(i)+1}_smiles'))



    
fig.update_layout(title='Cumulative Distribution of Building Blocks in the test set',
                  xaxis_title='Building Block Count',
                  yaxis_title='Cumulative Probability')

fig.show()

# %% [markdown]
# In the test set, it seems that representativity of molecules os more or less linear, at least for BB1. 

# %% [markdown]
# ## **Overlapping between train and test set**
# Here, we check for the BBs that are within the train and test set or both.
# Spoiler: we will see that all the blocks in the training set are present in the test set, but not the contrary. Modeling can be thus tricky since models will have to generalize well.

# %%
#list initialization
fq_builds_intersect = []
fq_builds_train = []
fq_builds_test = []

for i in range(1,4):
    buildi_train_set = con.query(f"""SELECT DISTINCT buildingblock{i}_smiles
                                 FROM parquet_scan('{train_path_rw}')
                                 """).df()
    
    buildi_test_set = con.query(f"""SELECT DISTINCT buildingblock{i}_smiles
                                 FROM parquet_scan('{test_path_rw}')
                                 """).df()

    intersect_buildi = set(list(buildi_train_set.values.squeeze())).intersection(list(buildi_test_set.values.squeeze()))

    set_train_build_i = set(list(buildi_train_set.values.squeeze()))
    set_test_build_i = set(list(buildi_test_set.values.squeeze()))

    intersect_buildi = set_train_build_i.difference(set_test_build_i)

    train_only_buildi = set_train_build_i.difference(intersect_buildi)
    test_only_buildi = set_test_build_i.difference(intersect_buildi)

    n_buildi_intersect = len(intersect_buildi)
    n_buildi_test = len(test_only_buildi)
    n_buildi_train = len(train_only_buildi)
    n_buildi_total = n_buildi_test + n_buildi_train + n_buildi_intersect

    fq_buildi_intersect = n_buildi_intersect / n_buildi_total
    fq_buildi_train = n_buildi_train / n_buildi_total
    fq_buildi_test = n_buildi_test / n_buildi_total

    fq_builds_intersect.append(fq_buildi_intersect)
    fq_builds_train.append(fq_buildi_train)
    fq_builds_test.append(fq_buildi_test)

# %%
#fig = go.Figure()
circles = []
subplot_titles = [f"Overlapping of building blocks {i+1}" for i in range(3)]
fig = make_subplots(rows=1, cols=3, subplot_titles=subplot_titles)
for i in range(3):

    text = [f"Train set <br>{int(100*fq_builds_train[i])}%", 
            f"Intersection <br>{int(100*fq_builds_intersect[i])}%",
            f"Test set <br>{int(100*fq_builds_test[i])}%"]
    # Create scatter trace of text labels
    fig.add_trace(go.Scatter(
        x=[1, 1.75, 2.5],
        y=[1, 1, 1],
        text=text,
        mode="text",
        textfont=dict(
            color="black",
            size=18,
            family="Arail",
        )
    ),row=1, col=i+1)

    # Update axes properties
    fig.update_xaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )

    fig.update_yaxes(
        showticklabels=False,
        showgrid=False,
        zeroline=False,
    )


    # Add circles
    circles.append(dict(type="circle",
        line_color="gray", fillcolor="green",
        x0=0, y0=0, x1=2, y1=2,opacity=0.3,xref=f"x{i+1}",yref=f"y{i+1}",
    ))

    circles.append(dict(type="circle",
        line_color="gray", fillcolor="orange",
        x0=1.5, y0=0, x1=3.5, y1=2,opacity=0.3,xref=f"x{i+1}",yref=f"y{i+1}",
    ))


#fig.update_shapes(opacity=0.3, xref="x", yref="y")
fig.update_layout(
    shapes=circles)

fig.update_layout(
    margin=dict(l=20, r=20, b=100),
    height=600, width=800,
    plot_bgcolor="white"
)

fig.update_layout(title="Overlapping of building blocks between train and test sets")

fig.update_layout(
    autosize=False,
    width=2000,
    height=500)

#fig.show()
Image(fig.to_image(format="png", width=2000, height=500, scale=1))

# %% [markdown]
# # **Visualization**

# %% [markdown]
# ### 2D visualization
# Let's visualize the first molecule in molecule_smiles in 2D from our sample df. 

# %%
first_smiles = sample['molecule_smiles'].iloc[0]
first_smiles

# %%
first_molecule = Chem.MolFromSmiles(first_smiles)
first_molecule

# %%
Draw.MolToImage(first_molecule, size=(700, 700))

# %% [markdown]
# ### 3D visualization
# 

# %% [markdown]
# Let's visualize the first molecule in molecule_smiles in 3D from our sample df.

# %%
#Add hydrogen atoms to the molecule
first_molecule = Chem.AddHs(first_molecule)

#Embed the molecule in 3D space
AllChem.EmbedMolecule(first_molecule)

#3D visualization
view = py3Dmol.view(width=800, height=600)
pbd_block_molecule = Chem.MolToPDBBlock(first_molecule)
view.addModel(pbd_block_molecule, 'pdb')
view.setStyle({'stick': {}})
view.zoomTo()
view.show()

# %%


# %% [markdown]
# #### BB1 smiles

# %%
bb1_smiles = sample["buildingblock1_smiles"].iloc[0]
bb1_molecule = Chem.MolFromSmiles(bb1_smiles)

#Add hydrogen atoms to the molecule
bb1_molecule = Chem.AddHs(bb1_molecule)

#Embed the molecule in 3D space
AllChem.EmbedMolecule(bb1_molecule)

#3D visualization
view = py3Dmol.view(width=800, height=600)
pbd_block_bb_1 = Chem.MolToPDBBlock(bb1_molecule)
view.addModel(pbd_block_bb_1, 'pdb')
view.setStyle({'stick': {}})
view.zoomTo()
view.show()

# %% [markdown]
# #### BB2 smmiles
# The same but with BB2

# %%
bb2_smiles = sample["buildingblock2_smiles"].iloc[0]
bb2_molecule = Chem.MolFromSmiles(bb2_smiles)

#Add hydrogen atoms to the molecule
bb1_molecule = Chem.AddHs(bb2_molecule)

#Embed the molecule in 3D space
AllChem.EmbedMolecule(bb2_molecule)

#3D visualization
view = py3Dmol.view(width=800, height=600)
pbd_block_bb_2 = Chem.MolToPDBBlock(bb2_molecule)
view.addModel(pbd_block_bb_2, 'pdb')
view.setStyle({'stick': {}})
view.zoomTo()
view.show()

# %% [markdown]
# #### BB3 Smiles
# Lastly, with BB3

# %%
bb3_smiles = sample["buildingblock3_smiles"].iloc[0]
bb3_molecule = Chem.MolFromSmiles(bb3_smiles)

#Add hydrogen atoms to the molecule
bb3_molecule = Chem.AddHs(bb3_molecule)

#Embed the molecule in 3D space
AllChem.EmbedMolecule(bb3_molecule)

#3D visualization
view = py3Dmol.view(width=800, height=600)
pbd_block_bb_3 = Chem.MolToPDBBlock(bb3_molecule)
view.addModel(pbd_block_bb_3, 'pdb')
view.setStyle({'stick': {}})
view.zoomTo()
view.show()


