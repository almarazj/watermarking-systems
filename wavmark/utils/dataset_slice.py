import os
import argparse
from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from typing import Tuple

def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="Obtain a portion of the ASVspoof dataset maintaining the proportions of attack types."
    )
    parser.add_argument(
        "--data_path", required=True, type=str, help="Path to the dataset location.",
    )
    parser.add_argument(
        "--protocol_path", required=True, type=str, help="Path to the protocol text file.",
    )
    parser.add_argument(
        "--dataset_split", required=False, type=str, help="Specify train, dev or eval split."
    )
    parser.add_argument(
        "--percentage", required=True, type=int, help="Percentage of the dataset to be reduced into."
    )

    args = parser.parse_args()
    return args

def slice_dataset(dataset_path: Path,
                  percentage_of_dataset: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the dataset into a smaller sample while maintaining the proportions of the attack types.

    Parameters
    ----------
    dataset_path : Path 
        Path to the dataset file.
    percentage_of_dataset : int
        The percentage of the dataset to be selected for the sample.

    Returns
    -------
    dataset_df : pd.DataFrame 
        Dataframe containing the original dataset
    dataset_sample_df : pd.DataFrame
        Dataframe containing the reduced sample.
    """
    
    dataset_df = pd.read_csv(dataset_path, sep=' ', header=None)
    stratify_column = dataset_df.iloc[:, 2]
    _, dataset_sample_df = train_test_split(dataset_df, test_size=percentage_of_dataset/100, stratify=stratify_column, random_state=42)
    
    return dataset_df, dataset_sample_df

def check_proportions(dataset: pd.DataFrame,
                      dataset_sample: pd.DataFrame) -> None:
    """
    Checks and prints the proportions of attack types in both the original dataset and the sample.

    Parameters
    ----------
    dataset : pd.DataFrame 
        The original dataset.
    dataset_sample : pd.DataFrame 
        The reduced sample from the dataset.

    Returns
    -------
    None : 
        Prints a comparison table of the proportions for each attack type.
    """
    original_proportions = dataset.iloc[:, 2].value_counts(normalize=True)
    sample_proportions = dataset_sample.iloc[:, 2].value_counts(normalize=True)
    
    comparison_table = pd.DataFrame({
        'Original Proportion': original_proportions,
        'Sample Proportion': sample_proportions
    })
    
    print("\nProportion comparison for each attack type:")
    print(comparison_table)

def get_paths(args) -> dict:
    """
    Generates and returns the necessary paths for the dataset and protocol files.

    Parameters
    ----------
    args : argparse.Namespace 
        Parsed command-line arguments.

    Returns
    -------
    dict : 
        A dictionary containing paths for the dataset, output dataset, protocol files, and output protocol files.
    """

    protocol_path = Path(args.protocol_path).resolve()
    data_path = Path(args.data_path).resolve()

    # Ensure the paths are correctly resolved
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {protocol_path}")
    
    if not data_path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {data_path}")

    protocol_filename = protocol_path.stem
    protocol_extension = protocol_path.suffix

    output_protocol_filename = f"{protocol_filename}-{args.percentage}{protocol_extension}"

    data_folder_name = data_path.name

    
    return {
        "dataset_path": data_path,
        "output_dataset_path": data_path.parent / f'{data_folder_name}-{args.percentage}',
        "protocols_path": protocol_path,
        "output_protocols_path": protocol_path.parent / output_protocol_filename,
    }
    
def save_new_dataset(dataset_sample, paths) -> None:
    """
    Copies the files from the dataset sample to the new output folder and saves the corresponding protocol file.

    Parameters
    ----------
    dataset_sample : pd.DataFrame
        The reduced sample from the dataset.
    paths : dict
        A dictionary containing paths for the dataset and protocol files.

    Returns
    -------
    None : 
        Copies files and saves the protocol file to the specified output path.
    """
    os.makedirs(paths["output_dataset_path"], exist_ok=True)
    
    for filename in tqdm(dataset_sample.iloc[:, 1], total=len(dataset_sample.iloc[:, 1])):
        source = os.path.join(paths["dataset_path"], f'{filename}.wav')
        output = os.path.join(paths["output_dataset_path"], f'{filename}.wav')
        shutil.copyfile(source, output)
    print(f"Files copied to {paths['output_dataset_path']}")
    
    np.savetxt(paths["output_protocols_path"], dataset_sample, fmt='%s', delimiter=' ')
    print(f'Saved {paths["output_protocols_path"]}.')
    
def main():
    
    args = get_args()
    paths = get_paths(args)
    
    dataset, dataset_sample = slice_dataset(paths["protocols_path"], args.percentage)
    
    check_proportions(dataset, dataset_sample)
    
    save_new_dataset(dataset_sample, paths)

if __name__ == "__main__":
    main()