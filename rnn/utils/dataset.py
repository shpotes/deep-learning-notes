from typing import Callable

import numpy as np
import torch

class TemporalOrder(torch.utils.data.IterableDataset):
    def __init__(self,
                 difficulty_level: str = 'hard',
                 transforms: Callable = None,
                 seed: int = 42):
        
        super(TemporalOrder).__init__()
        self.classes = ['Q', 'R', 'S', 'U']

        self.relevant_symbols = ['X', 'Y']
        self.distraction_symbols = ['a', 'b', 'c', 'd']

        self.start_symbol = 'B'
        self.end_symbol = 'E'

        if seed is not None:
            np.random.seed(seed)

        all_symbols = self.relevant_symbols + self.distraction_symbols + \
            [self.start_symbol] + [self.end_symbol]

        self.sym_to_idx = {s: n for n, s in enumerate(all_symbols)}
        self.idx_to_sym = {n: s for n, s in enumerate(all_symbols)}

        self.class_to_idx = {c: n for n, c in enumerate(self.classes)}
        self.idx_to_class = {n: c for n, c in enumerate(self.classes)}

        self.set_difficulty(difficulty_level)

        if transform is None:
            self.transforms = lambda x: x
        else:
            self.transforms = lambda x, y: transforms(x, y)


    def set_difficulty(self, difficulty_level: str):
        if difficulty_level == 'easy':
            self.sim_params = {
                'length_range': (7, 9),
                't1_range': (1, 3),
                't2_range': (4, 6),
            }
        elif difficulty_level == 'normal':
            self.sim_params = {
                'length_range': (30, 41),
                't1_range': (2, 6),
                't2_range': (20, 28),
            }
        elif difficulty_level == 'moderate':
            self.sim_params = {
                'length_range': (60, 81),
                't1_range': (10, 21),
                't2_range': (45, 55),
            }
        elif difficulty_level == 'hard':
            self.sim_params = {
                'length_range': (100, 111),
                't1_range': (10, 21),
                't2_range': (50, 61),
            }
        elif difficulty_level == 'nightmare':
            self.sim_params = {
                'length_range': (300, 501),
                't1_range': (10, 81),
                't2_range': (250, 291),
            }
        else:
            raise NotImplementedError

    def __iter__(self):
        while True:
            length = np.random.randint(*self.sim_params['length_range'])
            t1 = np.random.randint(*self.sim_params['t1_range'])
            t2 = np.random.randint(*self.sim_params['t2_range'])

            x = np.random.choice(self.distraction_symbols, length)
            x[0] = self.start_symbol
            x[-1] = self.end_symbol

            y = np.random.choice(self.classes)

            if y == 'Q':
                x[t1], x[t2] = self.relevant_symbols[0], self.relevant_symbols[0]
            elif y == 'R':
                x[t1], x[t2] = self.relevant_symbols[0], self.relevant_symbols[1]
            elif y == 'S':
                x[t1], x[t2] = self.relevant_symbols[1], self.relevant_symbols[0]
            else:
                x[t1], x[t2] = self.relevant_symbols[1], self.relevant_symbols[1]

            x = ''.join(x)

            yield self.transforms(x, y)
