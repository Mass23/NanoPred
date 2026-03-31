import argparse
from src import data_processing

def main():
    parser = argparse.ArgumentParser(description="Get data from sequences efficiently")
    parser.add_argument("-f", "--fasta", required=True, help="Path to input fasta file")
    parser.add_argument("-p1", "--primer1", required=True, help="Forward primer sequence")
    parser.add_argument("-p2", "--primer2", required=True, help="Reverse primer sequence")
    args = parser.parse_args()
    
    # Call your src/data_processing functions here
    data_processing.process_fasta_for_benchmark(args.fasta, args.primer1, args.primer2)
    
if __name__ == "__main__":
    main()
