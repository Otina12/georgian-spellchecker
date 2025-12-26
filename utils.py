import random

# map of neighboring georgian letters on a standard keyboard. None = Backspace, e. g. skip current
georgian_keyboard_map = [
    ['ქ', 'წჭ', 'ე', 'რღ', 'ტთ', 'ყ', 'უ', 'ი', 'ო', 'პ'],
    ['ა', 'სშ', 'დ', 'ფ', 'გ', 'ჰ', 'ჯჟ', 'კ', 'ლ', None],
    ['ზძ', 'ხ', 'ცჩ', 'ვ', 'ბ', 'ნ', 'მ', None, None, None],
]

n, m = len(georgian_keyboard_map), len(georgian_keyboard_map[0])

dirs = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1), ( 0, 0), ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

step_probabilities = [
    0.00005, 0.00020, 0.00005,
    0.00020, 0.99900, 0.00020,
    0.00005, 0.00200, 0.00005,
]

def keyboard_typo(word, shift_change_prob = 0.05):
    char_to_pos = {}

    for i, row in enumerate(georgian_keyboard_map):
        for j, cell in enumerate(row):
            if cell is not None:
                for shift_idx, char in enumerate(cell):
                    char_to_pos[char] = (i, j, shift_idx)
    
    corrupted_chars = []
    
    for char in word:
        if char not in char_to_pos:
            corrupted_chars.append(char)
            continue
        
        row, col, shift_idx = char_to_pos[char]
        
        direction_idx = random.choices(range(len(dirs)), weights = step_probabilities)[0]
        dr, dc = dirs[direction_idx]
        
        new_row = row + dr
        new_col = col + dc
        
        if 0 <= new_row < n and 0 <= new_col < m:
            target_cell = georgian_keyboard_map[new_row][new_col]
            
            if target_cell is None:  # None = Backspace = skip
                continue

            if random.random() < shift_change_prob and len(target_cell) > 1:
                new_shift_idx = 1 - shift_idx if shift_idx < 2 else 0
                new_shift_idx = min(new_shift_idx, len(target_cell) - 1)
            else:
                new_shift_idx = min(shift_idx, len(target_cell) - 1)
            
            corrupted_chars.append(target_cell[new_shift_idx])
        else:
            corrupted_chars.append(char)
    
    return ''.join(corrupted_chars)

def swap_adjacent_chars(word, swap_prob = 0.005):
    chars = list(word)
    current_prob = swap_prob
    i = 0
    
    while i < len(chars) - 1:
        if random.random() < current_prob:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            current_prob /= 10 # decrease probability for less swaps
            i += 2
        else:
            i += 1
    
    return ''.join(chars)

def double_char(word, double_prob = 0.005):
    chars = []
    current_prob = double_prob
    
    for char in word:
        chars.append(char)
        if random.random() < current_prob:
            chars.append(char)
            current_prob /= 10 # decrease probability for less duplications
    
    return ''.join(chars)

def corrupt_word(word, shift_change_prob = 0.05, swap_prob = 0.005, double_prob = 0.005):
    word = keyboard_typo(word, shift_change_prob)
    word = swap_adjacent_chars(word, swap_prob)
    word = double_char(word, double_prob)
    return word

# function that returns edit distance (Levenshtein distance) between two words
def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[n][m]

def encode_word(word, stoi, unk_idx, eos_idx, add_eos = True):
    indices = []
    for ch in word:
        indices.append(stoi.get(ch, unk_idx))
    if add_eos:
        indices.append(eos_idx)
    return indices

def decode_indices(indices, itos, eos_idx, pad_idx):
    chars = []
    for idx in indices:
        if idx == eos_idx or idx == pad_idx:
            break
        chars.append(itos[idx])
    return ''.join(chars)