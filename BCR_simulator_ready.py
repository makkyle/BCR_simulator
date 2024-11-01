## load in the binding model and its pretrained weights 

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow.keras.layers as layers
from tensorflow.keras import activations
# from tensorflow.keras.layers import Layer, Input, Embedding, LSTM, Dense, Attention
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

n_classes = 1

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res
def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,):
    
    inputs = keras.Input(shape=input_shape)
    inputs2 = keras.Input(shape=input_shape)
    inputs3 = keras.Input(shape=input_shape)
    x = inputs
    y = inputs2
    z = inputs3

    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        y = transformer_encoder(y, head_size, num_heads, ff_dim, dropout)
        z = transformer_encoder(z, head_size, num_heads, ff_dim, dropout)


    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    y = layers.GlobalAveragePooling1D(data_format="channels_last")(y)
    z = layers.GlobalAveragePooling1D(data_format="channels_last")(z)

    x = layers.Concatenate()([x,y,z])
    # for dim in mlp_units:
    x = layers.Dense(mlp_units, activation="relu")(x)
    x = layers.Dropout(mlp_dropout)(x)

    outputs = layers.Dense(n_classes, activation="sigmoid")(x)

    model = keras.Model([inputs,inputs2,inputs3],outputs)
    
#     model.compile(loss="mse",
#     optimizer=tf.keras.optimizers.SGD(learning_rate=1e-04),
#     metrics=["mse"],)

    return model

input_shape = (3000,21)

model = build_model(
    input_shape,
    head_size=64,
    num_heads=6,
    ff_dim=6,
    num_transformer_blocks=1,
    mlp_units=80,
    mlp_dropout=0.1,
    dropout=0.25,
)

# Load the weights
model.load_weights('transformer_binding_affinity_300_21.h5')

# Compile the model
model.compile(
    loss="mse",
    optimizer=keras.optimizers.SGD(learning_rate=1e-5),
    metrics=["mse"],
)
print("Model loaded and compiled successfully.")

#######################################################
##            one-hot encoding (sequence)            ##
#######################################################
import numpy as np

# Define the amino acids
ALPHABET = 'ACDEFGHIKLMNPQRSTVWY*'

def one_hot_encoder(s, alphabet=ALPHABET, max_length=300):
    # Build dictionary
    d = {a: i for i, a in enumerate(alphabet)}
    ## output: {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}

    # Encode
    x = np.zeros((max_length, len(d)), dtype=int)
    for idx, c in enumerate(s[:max_length]):
        if c in d:
            x[idx, d[c]] = 1
    return x



def np_generator(data, sequence_length=3000, feature_dim=21):
    np_input = []
    for index in range(len(data)):
        ex = one_hot_encoder(data[index])
        seq_v = np.zeros((sequence_length, feature_dim), dtype=int)
        seq_v[:min(ex.shape[0], sequence_length), :] = ex[:sequence_length, :]
        np_input.append(seq_v)
        print('shape of current index:', np.shape(ex))
        print('shape of embedding:', np.shape(seq_v))
        print('shape of numpy list:', np.shape(np_input))
    
    np_input_array = np.array(np_input)
    print('Final shape of numpy array:', np.shape(np_input_array))
    
    return np_input_array


# np_generator(["VVICGEHVVE"])

def generate_nucleotide_sequence(length):
    return ''.join(np.random.choice(list('ACGT'), length))

# Function to translate nucleotide sequence to amino acid sequence
def translate_sequence(nucleotide_seq):
    return ''.join(genetic_code[nucleotide_seq[i:i+3]] for i in range(0, len(nucleotide_seq), 3))

#######################################################
##         Architecture of BCR simulator             ##
#######################################################

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ete3 import Tree, TreeStyle
from tqdm import tqdm
from collections import Counter

# Parameters
initial_cells_range = [50, 100, 200]  # Different initial number of B cells
CDR_length_H = 120  # Length of CDR sequence in nucleotides (must be a multiple of 3)
CDR_length_L = 120
base_P_survive_range = np.arange(0.001, 0.98, 0.001).tolist()  # Different base probabilities of survival
theta_range = [0.02, 0.05, 0.1]  # Different increases in survival probability per mutation
cycles = 20  # Number of cycles (increase this number for more generations)
mutation_rate_range = [0.1, 0.3, 0.5]  # Different mutation rates
num_simulations = 10  # Number of simulations for each parameter set
proliferate_no = 10

# Genetic code dictionary
genetic_code = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'Z', 'TAG':'Z',
    'TGC':'C', 'TGT':'C', 'TGA':'Z', 'TGG':'Z',
}

## input the antigen sequence here
Ag_seq = "KEKQDVFCDSKLMSAAIKDNRAVHADMGYWIESALNDTWKIEKASFIEVKSCHWPKSHTLWSNGVLESEMIIPKNFAGPVSQHNYRPGYHTQTAGPWHLGSLEMDFDFCEGTTVVVTEDCGNRGPSLRTTTASGKLITEWCCRSCTLPPLRYRGEDGCWYGMEIRPLKEKEENLVNSLVTA"


# Function to generate a random nucleotide sequence
def generate_nucleotide_sequence(length):
    return ''.join(np.random.choice(list('ACGT'), length))

# Function to translate nucleotide sequence to amino acid sequence
def translate_sequence(nucleotide_seq):
    return ''.join(genetic_code[nucleotide_seq[i:i+3]] for i in range(0, len(nucleotide_seq), 3))
    
# Function to calculate affinity (higher is better)
def calculate_affinity(Ag_seq,HC_seq,LC_seq):
    ## convert the sequences into np.array with shape of (1,300,20)
    print('converting numpy array for Ag_seq........')
    Ag_seq = np_generator([Ag_seq])
    # np_input = []
    print('converting numpy array for HC_seq........')
    HC_seq = list(HC_seq)
    HC_seq = np_generator([HC_seq])
    # np_input = []
    print('converting numpy array for LC_seq........')
    LC_seq = list(LC_seq)
    print('LC_seq', LC_seq)
    LC_seq = np_generator([LC_seq])
    # np_input = []
    print(np.shape(Ag_seq))
    print(np.shape(HC_seq))
    print(np.shape(LC_seq))
    # print(model.summary())
    affinity_score = model.predict([HC_seq,LC_seq,Ag_seq])
    print('affinity_score',affinity_score)
    return affinity_score
    
# Function to calculate Shannon Entropy
def calculate_shannon_entropy(sequences):
    counts = Counter(sequences)
    total = sum(counts.values())
    entropy = -sum((count / total) * np.log2(count / total) for count in counts.values())
    return entropy

def extract_same_CDR_sequences(list_of_dicts,key,value):
    # Key and value to filter by
    key_to_filter = key
    value_to_filter = value
    
    # Extract dictionaries where the value for the specified key matches the given value
    filtered_dicts = [d for d in list_of_dicts if d.get(key_to_filter) == value_to_filter]
    
    # Print the result
    print(filtered_dicts)

    return filtered_dicts

# Function to simulate a cycle
def simulate_cycle(B_cells, base_P_survive , proliferate_no):
    new_B_cells = []
    old_B_cells = B_cells.copy()
    for cell in B_cells:
        
        # ## proliferate (based on itself) ie. clone itself in sequence-level
        # for _ in range(proliferate_no):
        #     B_cells.append(cell)
        
        ## measure binding affinity between the B cell receptor & antigen 
        affinity = calculate_affinity(Ag_seq,cell['CDRH'],cell['CDRL'])
        # P_survive = min(base_P_survive + theta * affinity, 1.0)
        P_survive = base_P_survive
        
        if affinity > P_survive:
            new_amino_acid_seq_HC = cell['CDRH']
            new_amino_acid_seq_LC = cell['CDRL']
            # Introduce mutations (the proliferation step ie. step before binding affinity -- also included here) 
            for _ in range(proliferate_no):
                mutation_index_HC = np.random.randint(0, len(new_amino_acid_seq_HC))
                new_amino_acid_seq_HC = new_amino_acid_seq_HC[:mutation_index_HC] + np.random.choice(list('ACDEFGHIKLMNPQRSTVWY')) + new_amino_acid_seq_HC[mutation_index_HC+1:]
                
                mutation_index_LC = np.random.randint(0, len(new_amino_acid_seq_LC))
                new_amino_acid_seq_LC = new_amino_acid_seq_LC[:mutation_index_LC] + np.random.choice(list('ACDEFGHIKLMNPQRSTVWY')) + new_amino_acid_seq_LC[mutation_index_LC+1:]
                
                new_B_cells.append({'CDRH': new_amino_acid_seq_HC, 'CDRL': new_amino_acid_seq_LC, 'parent_H': cell['CDRH'] , 'parent_L': cell['CDRL']})
            # new_B_cells.append({'CDR': new_amino_acid_seq, 'parent': cell['CDR']})
        
    return new_B_cells

# Function to run the simulation with given parameters
def run_simulation(initial_cells, base_P_survive):
    ## for each B cell, a randomly generated CDR sequence with particular length is generated and sequence of its parent (which is None)
    B_cells = [{'CDRH': translate_sequence(generate_nucleotide_sequence(CDR_length_H)), 'CDRL': translate_sequence(generate_nucleotide_sequence(CDR_length_L)) ,'parent_H': None, 'parent_L': None} for _ in range(initial_cells)]
    ## generate a dictionary called 'lineage' where each B cell randomly generated sequence put into ['CDR'] and None for ['parent']
    lineage = { (cell['CDRH'],cell['CDRL']) : (cell['parent_H'],cell['parent_L']) for cell in B_cells}
    shannon_entropy_over_time = []

    for cycle in range(cycles):
        cells = simulate_cycle(B_cells, base_P_survive, proliferate_no)
        for cell in B_cells:
            lineage[cell['CDRH']] = cell['parent_H']
            lineage[cell['CDRL']] = cell['parent_L']
        
        # Calculate Shannon Entropy for the current generation
        CDR_sequences = [cell['CDRH'] for cell in B_cells]
        entropy = calculate_shannon_entropy(CDR_sequences)
        shannon_entropy_over_time.append(entropy)
    
    return shannon_entropy_over_time

# Run simulations for different parameter sets and collect results
results = []

for initial_cells in initial_cells_range:
    for base_P_survive in base_P_survive_range:
        ## display it as progress bar 
        for _ in tqdm(range(num_simulations), desc=f"Simulating for initial_cells={initial_cells}, base_P_survive={base_P_survive}"):
            entropy_over_time = run_simulation(initial_cells, base_P_survive)
            for generation, entropy in enumerate(entropy_over_time):
                results.append({
                    'initial_cells': initial_cells,
                    'base_P_survive': base_P_survive,
                    'generation': generation,
                    'entropy': entropy
                })




