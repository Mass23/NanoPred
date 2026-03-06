def iupac_code_handling(seq):
    # Handle IUPAC codes in the sequence
    # Example implementations
    # A = 'A', T = 'T', G = 'G', C = 'C', R = 'AG', Y = 'CT', S = 'GC', W = 'AT', K = 'GT', M = 'AC', B = 'CGT', D = 'AGT', H = 'ACT', V = 'ACG', N = 'ACGT'

    return seq


def reverse_complement(seq):
    complements = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    return ''.join(complements[base] for base in reversed(seq.upper()))


def primer_matching(seq, primer):
    # Check if the primer matches the sequence
    return primer in seq

# Example usage:
# print(reverse_complement('ATGC'))
# print(primer_matching('ATGC', 'AT'))
# print(iupac_code_handling('ATGRY'))
