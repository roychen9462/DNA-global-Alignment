#!/usr/bin/env python3
import os
import sys
import time
import psutil

import numpy as np

class SeqAlign(object):
    def __init__(self, seq1, seq2, delta=30):
        self.delta = delta

        self.seq1 = seq1
        self.seq2 = seq2
        self.dp = np.zeros([len(seq1)+1, len(seq2)+1], dtype=np.int32)

    def run(self):
        self._compute_array()
        self._trace_back()

    def score(self):
        return self.dp[len(self.seq1), len(self.seq2)]

    def print_aligned_seqs(self, n=500, head=True):
        if head:
            return (self.aligned_seq1[:n], self.aligned_seq2[:n])
        else:
            return (self.aligned_seq1[-n:], self.aligned_seq2[-n:])

    def _alpha(self, letter_1, letter_2):
        # Adding the ASCII code of two letters as key value
        alpha_table = {132: 110,         # (A,C), (C,A)
                       136: 48,          # (A,G), (G,A)
                       149: 94,          # (A,T), (T,A)
                       138: 118,         # (C,G), (G,C)
                       151: 48,          # (C,T), (T,C)
                       155: 110}         # (G,T), (T,G)

        if letter_1 == letter_2:
            return 0
        else:
            return alpha_table[ord(letter_1)+ord(letter_2)]

    def _compute_array(self):
        # Initialization
        for i in range(len(self.seq1)+1):
            self.dp[i, 0] = i*self.delta
        for i in range(len(self.seq2)+1):
            self.dp[0, i] = i*self.delta

        # Compute array
        for i in range(1, len(self.seq1)+1):
            for j in range(1, len(self.seq2)+1):
                self.dp[i, j] = min(self._alpha(self.seq1[i-1], self.seq2[j-1]) + self.dp[i-1, j-1],
                                    self.delta + self.dp[i-1, j],
                                    self.delta + self.dp[i, j-1])

    def _trace_back(self):
        self.aligned_seq1 = ""
        self.aligned_seq2 = ""

        i = len(self.seq1)
        j = len(self.seq2)
        while i > 0 and j > 0:
            if self.dp[i, j] == self._alpha(self.seq1[i-1], self.seq2[j-1]) + self.dp[i-1, j-1]:
                self.aligned_seq1 += self.seq1[i-1]
                self.aligned_seq2 += self.seq2[j-1]
                i -= 1
                j -= 1
            elif self.dp[i, j] == self.delta + self.dp[i-1, j]:
                self.aligned_seq1 += self.seq1[i-1]
                self.aligned_seq2 += "_"
                i -= 1
            elif self.dp[i, j] == self.delta + self.dp[i, j-1]:
                self.aligned_seq1 += "_"
                self.aligned_seq2 += self.seq2[j-1]
                j -= 1

        if i > 0:
            self.aligned_seq1 += self.seq1[0:i-1]
            self.aligned_seq2 += "_"*(i-1)

        if j > 0:
            self.aligned_seq1 += "_"*(j-1)
            self.aligned_seq2 += self.seq2[0:j-1]

        self.aligned_seq1 = self.aligned_seq1[::-1]
        self.aligned_seq2 = self.aligned_seq2[::-1]


class EfficientSeqAlign(object):
    def __init__(self, seq1, seq2, delta=30):
        self.delta = delta
        if len(seq1) <= len(seq2):
            self.seq1 = seq1
            self.seq2 = seq2
            self.flag = 0
        else:
            self.seq1 = seq2
            self.seq2 = seq1
            self.flag = 1
        self.aligned_seq1 = ""
        self.aligned_seq2 = ""
        self.p = []
    
    def run(self):
        self._divide_and_conquer_align(self.seq1, self.seq2)
        if self.flag:
            self.seq1, self.seq2 = self.seq2, self.seq1

    def print_aligned_seqs(self, n=500, head=True):
        if head:
            return (self.aligned_seq1[:n], self.aligned_seq2[:n])
        else:
            return (self.aligned_seq1[-n:], self.aligned_seq2[-n:])
    def score(self):
        c = 0
        for i, j in zip(self.aligned_seq1, self.aligned_seq2):
            if i == "_" or j == "_":
                c += self.delta
            else:
                c += self._alpha(i, j)

        return c

    def _alpha(self, letter_1, letter_2):
        # Adding the ASCII code of two letters as key value
        alpha_table = {132: 110,         # (A,C), (C,A)
                       136: 48,          # (A,G), (G,A)
                       149: 94,          # (A,T), (T,A)
                       138: 118,         # (C,G), (G,C)
                       151: 48,          # (C,T), (T,C)
                       155: 110}         # (G,T), (T,G)

        if letter_1 == letter_2:
            return 0
        else:
            return alpha_table[ord(letter_1)+ord(letter_2)]

    def _divide_and_conquer_align(self, seq1, seq2, s_seq1=0, s_seq2=0):
        m = len(seq1)
        n = len(seq2)

        if m <= 2 or n <= 2:
            align_seq1, align_seq2 = self._basic_align(seq1, seq2)
            self.aligned_seq1 += align_seq1
            self.aligned_seq2 += align_seq2

            return

        idx_seq2 = int(n/2)
        f_dp = self._space_efficient_align(seq1, seq2[0:idx_seq2])
        g_dp = self._backward_space_efficient_align(seq1, seq2[idx_seq2:])
        q = np.argmin(f_dp + g_dp)
        self.p.append((q+s_seq1, idx_seq2+s_seq2))

        self._divide_and_conquer_align(seq1[:q], seq2[:idx_seq2], s_seq1=s_seq1, s_seq2=s_seq2)
        self._divide_and_conquer_align(seq1[q:], seq2[idx_seq2:], s_seq1=q+s_seq1, s_seq2=idx_seq2+s_seq2)

    def _basic_align(self, seq1, seq2):
        align = SeqAlign(seq1, seq2, self.delta)
        align.run()

        return align.aligned_seq1, align.aligned_seq2

    def _space_efficient_align(self, seq1, seq2):
        # Initialization
        dp = np.zeros([len(seq1)+1, 2])
        for i in range(len(seq1)+1):
            dp[i, 0] = i*self.delta

        # Compute array
        for i in range(1, len(seq2)+1):
            dp[0, 1] = i*self.delta
            for j in range(1, len(seq1)+1):
                dp[j, 1] = min(self._alpha(seq1[j-1], seq2[i-1]) + dp[j-1, 0],
                               self.delta + dp[j-1, 1],
                               self.delta + dp[j, 0])
            dp[:, 0] = dp[:, 1]

        return dp[:, 1]

    def _backward_space_efficient_align(self, seq1, seq2):
        # Initialization
        dp = np.zeros([len(seq1)+1, 2])
        for i in range(len(seq1)+1):
            dp[i, 0] = (len(seq1)-i)*self.delta

        # Compute array
        for i in range(len(seq2)-1, -1, -1):
            dp[len(seq1), 1] = (len(seq2)-i)*self.delta
            for j in range(len(seq1)-1, -1, -1):
                dp[j, 1] = min(self._alpha(seq1[j], seq2[i]) + dp[j+1, 0],
                               self.delta + dp[j+1, 1],
                               self.delta + dp[j, 0])
            dp[:, 0] = dp[:, 1]

        return dp[:, 1]

def gen_sequences(infile):
    with open(infile, 'r') as infile:
        seqs = []
        for line in infile:
            if not line[0].isdigit():
                seqs.append(line[:-1])
            else:
                idx = int(line)
                seqs[-1] = seqs[-1][:idx+1] + seqs[-1] + seqs[-1][idx+1:]

    return (seqs[0], seqs[1])


if __name__ == "__main__":
    # File to write time data to
    ofile = open(sys.argv[2], 'w')

    mem = psutil.Process(os.getpid())

    delta = 30
    seq1, seq2 = gen_sequences(sys.argv[1])

    t = time.time()
    mem_start = mem.memory_info().rss

    align = EfficientSeqAlign(seq1, seq2, delta)
    align.run()

    total_mem = mem.memory_info().rss - mem_start

    ofile.write(f"{align.print_aligned_seqs(50)[0]} {align.print_aligned_seqs(50, head=False)[0]}\n")
    ofile.write(f"{align.print_aligned_seqs(50)[1]} {align.print_aligned_seqs(50, head=False)[1]}\n")
    ofile.write(f"{time.time() - t}\n")
    ofile.write(f"{total_mem/1000}\n")
    ofile.close()
