import random

from rdkit import Chem


def contains_benzene(smiles):
    # Create an RDKit molecule object from the SMILES string
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # Define the SMILES for benzene
    benzene_smiles = 'c1ccccc1'
    benzene = Chem.MolFromSmiles(benzene_smiles)

    # Check if the molecule contains a benzene ring
    return mol.HasSubstructMatch(benzene)


def contains_halogen(smiles):
    # Create an RDKit molecule object from the SMILES string
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    # List of halogens: F, Cl, Br, I
    halogens = ['F', 'Cl', 'Br', 'I']

    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in halogens:
            return True

    return False


def trim_dataset(df, amt):
    rows_to_remove = random.sample(df.index.tolist(), amt)
    df.drop(rows_to_remove, inplace=True)


def filter_smiles_idxs(smiles, contains_filter):

    contains_indices = [i for i, item in enumerate(smiles) if contains_filter(item)]
    not_contains_indices = [i for i, item in enumerate(smiles) if not contains_filter(item)]

    # contains_subset = df[df['contains_filter'] == True].copy()
    # not_contains_subset = df[df['contains_filter'] == False].copy()
    #
    # contains_subset.drop(columns=['contains_filter'], inplace=True)
    # not_contains_subset.drop(columns=['contains_filter'], inplace=True)

    # if trim:
    #     if len(contains_subset) > len(not_contains_subset):
    #         trim_dataset(contains_subset, len(contains_subset) - len(not_contains_subset))
    #     elif len(not_contains_subset) > len(contains_subset):
    #         trim_dataset(not_contains_subset, len(not_contains_subset) - len(contains_subset))

    return contains_indices, not_contains_indices
