import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple, Any
from itertools import combinations
import time
import numpy as np

class DataLoader:
    def __init__(self, pokemon_file: str, moves_file: str, learnsets_file: str, type_chart_file: str):
        self.pokemon_df = pd.read_csv(pokemon_file)
        self.moves_df = pd.read_csv(moves_file)
        self.learnsets_df = pd.read_csv(learnsets_file)
        self.type_chart = pd.read_csv(type_chart_file, index_col=0)
        self.min_max_power = [0,999]
        self.att = [1,1]

    def get_pokemon_by_generation(self, generation: str) -> List[str]:
        # Filter Pokémon based on the selected generation
        filtered_pokemon = self.learnsets_df[self.learnsets_df['generation'] == int(generation)]['pokemon'].unique()
        return list(filtered_pokemon)
    
    def get_pokemon_list(self) -> List[str]:
        #a = sorted(self.pokemon_df['name'].unique().tolist())
        a = self.pokemon_df['name'].unique().tolist()
        return a
    def get_move_list(self)->List[str]:
        a = self.moves_df['name'].unique().tolist()
        return a
        
    def get_pokemon_types(self, name: str) -> List[str]:
        row = self.pokemon_df[self.pokemon_df['name'] == name].iloc[0]
        return [row['type1']] + ([row['type2']] if pd.notna(row['type2']) else [])

    def get_pokemon_learnset(self, name: str, generation: str = None) -> List[str]:
        if generation:
            moves = self.learnsets_df[(self.learnsets_df['pokemon'] == name) & (self.learnsets_df['generation'] == int(generation))]['move'].unique().tolist()
        else:
            # If no generation is specified, return all available moves
            moves = self.learnsets_df[self.learnsets_df['pokemon'] == name]['move'].unique().tolist()
        return moves
    
    def get_damaging_learnset(self, name: str, generation: str = None) -> List[str]:
        moves = self.learnsets_df[(self.learnsets_df['pokemon'] == name) & (self.learnsets_df['generation'] == int(generation))]['move'].unique().tolist()
        moves = [move for move in moves if self.is_damaging_move(move)]
        moves = [move for move in moves if self.power_min_max(move)]
        return moves

    def is_damaging_move(self, move_name: str) -> bool:
        row = self.moves_df[self.moves_df['name'] == move_name]
        return not row.empty and row.iloc[0]['is_damaging'] == 1
    
    def power_min_max(self, move_name: str) -> bool:
        row = self.moves_df[self.moves_df['name'] == move_name]
        return not row.empty and row.iloc[0]['power'] >= self.min_max_power[0] and row.iloc[0]['power'] <=self.min_max_power[1]

    def get_move_data(self, move_name: str) -> Tuple[str, int]:
        row = self.moves_df[self.moves_df['name'] == move_name]
        if row.empty:
            return ('', 0)
        return row.iloc[0]['type'], row.iloc[0]['power'], row.iloc[0]['dtype']

    def get_type_effectiveness(self, move_type: str, target_types: List[str]) -> float:
        multiplier = 1.0
        for t in target_types:
            multiplier *= self.type_chart.loc[move_type, t]
        return multiplier

    def get_all_target_types(self) -> Dict[str, List[str]]:
        return {
            row['name']: [row['type1']] + ([row['type2']] if pd.notna(row['type2']) else [])
            for _, row in self.pokemon_df.iterrows()
        }
    def calculate_type_effectiveness(self) -> pd.DataFrame:
    # Lowercase type chart
        type_chart = {
        'normal':   {'rock': 0.5, 'ghost': 0.0, 'steel': 0.5},
        'fire':     {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 2, 'bug': 2, 'rock': 0.5, 'dragon': 0.5, 'steel': 2},
        'water':    {'fire': 2, 'water': 0.5, 'grass': 0.5, 'ground': 2, 'rock': 2, 'dragon': 0.5},
        'electric': {'water': 2, 'electric': 0.5, 'grass': 0.5, 'ground': 0, 'flying': 2, 'dragon': 0.5},
        'grass':    {'fire': 0.5, 'water': 2, 'grass': 0.5, 'poison': 0.5, 'ground': 2, 'flying': 0.5, 'bug': 0.5, 'rock': 2, 'dragon': 0.5, 'steel': 0.5},
        'ice':      {'fire': 0.5, 'water': 0.5, 'grass': 2, 'ice': 0.5, 'ground': 2, 'flying': 2, 'dragon': 2, 'steel': 0.5},
        'fighting': {'normal': 2, 'ice': 2, 'poison': 0.5, 'flying': 0.5, 'psychic': 0.5, 'bug': 0.5, 'rock': 2, 'ghost': 0, 'dark': 2, 'steel': 2, 'fairy': 0.5},
        'poison':   {'grass': 2, 'poison': 0.5, 'ground': 0.5, 'rock': 0.5, 'ghost': 0.5, 'steel': 0, 'fairy': 2},
        'ground':   {'fire': 2, 'electric': 2, 'grass': 0.5, 'poison': 2, 'flying': 0, 'bug': 0.5, 'rock': 2, 'steel': 2},
        'flying':   {'electric': 0.5, 'grass': 2, 'fighting': 2, 'bug': 2, 'rock': 0.5, 'steel': 0.5},
        'psychic':  {'fighting': 2, 'poison': 2, 'psychic': 0.5, 'dark': 0, 'steel': 0.5},
        'bug':      {'fire': 0.5, 'grass': 2, 'fighting': 0.5, 'poison': 0.5, 'flying': 0.5, 'psychic': 2, 'ghost': 0.5, 'dark': 2, 'steel': 0.5, 'fairy': 0.5},
        'rock':     {'fire': 2, 'ice': 2, 'fighting': 0.5, 'ground': 0.5, 'flying': 2, 'bug': 2, 'steel': 0.5},
        'ghost':    {'normal': 0, 'psychic': 2, 'ghost': 2, 'dark': 0.5},
        'dragon':   {'dragon': 2, 'steel': 0.5, 'fairy': 0},
        'dark':     {'fighting': 0.5, 'psychic': 2, 'ghost': 2, 'dark': 0.5, 'fairy': 0.5},
        'steel':    {'fire': 0.5, 'water': 0.5, 'electric': 0.5, 'ice': 2, 'rock': 2, 'fairy': 2, 'steel': 0.5},
        'fairy':    {'fire': 0.5, 'fighting': 2, 'poison': 0.5, 'dragon': 2, 'dark': 2, 'steel': 0.5},
        }

        all_types = list(type_chart.keys())

    # Load and lowercase the data
        df = self.pokemon_df
        df = df[['type1', 'type2']]

    # Normalize and sort type combinations
        def sort_types(row):
            types = [row['type1'], row['type2']]
            types = [t for t in types if pd.notna(t)]
            types.sort()
            return pd.Series(types + [None] * (2 - len(types)))

        df[['sorted1', 'sorted2']] = df.apply(sort_types, axis=1)

    # Get unique combinations
        unique_combos = df.groupby(['sorted1', 'sorted2']).size().reset_index(name='count')

    # Helper function to calculate damage multiplier
        def calc_multiplier(attack_type, defend1, defend2):
            def_mult1 = type_chart.get(attack_type, {}).get(defend1, 1)
            def_mult2 = type_chart.get(attack_type, {}).get(defend2, 1) if defend2 else 1
            return def_mult1 * def_mult2

    # Add columns for each attacking type
        for atk_type in all_types:
            unique_combos[atk_type] = unique_combos.apply(
                lambda row: calc_multiplier(atk_type, row['sorted1'], row['sorted2']),
                axis=1
            )

        return unique_combos
def evaluate_moveset(data: DataLoader, attacker_name: str, selected_moves: List[str], type_combos: pd.DataFrame, generation: str) -> Dict[str, Any]:
    pre_split = False
    special = ['fire', 'water', 'electric', 'grass', 'ice', 'psychic', 'dragon', 'dark']
    physical = ['normal', 'fighting', 'poison', 'ground', 'flying', 'bug', 'rock', 'ghost', 'steel']
    if int(generation)<4:
        pre_split = True
    attacker_types = data.get_pokemon_types(attacker_name)
    type_shape = type_combos['normal'].to_numpy().shape
    moves_arrays = np.zeros((len(selected_moves), type_shape[0]))
    for i in range(0, len(selected_moves)):
        att = 0
        move_type, power, move_d_type = data.get_move_data(selected_moves[i])
        stab = 1
        if move_type in attacker_types:
            stab = 1.5
        if pre_split:
            if move_type in physical:
                att = data.att[0]
            elif move_type in special:
                att = data.att[1]
        else:
            if move_d_type == 2:
                att = data.att[0]
            elif move_d_type == 3:
                att = data.att[1]
        moves_arrays[i] = type_combos[move_type].to_numpy()*power*stab*att
        print(selected_moves[i])
    count = type_combos['count'].to_numpy().reshape((1,type_shape[0]))
    col_max = np.max(moves_arrays, axis=0)
    row_counts = np.zeros(len(selected_moves), dtype=int)
    for col in range(0,type_shape[0]):
        for row in range(0,len(selected_moves)):
            if moves_arrays[row, col] == col_max[col]:
                row_counts[row] += 1
    print(np.sum(col_max))
    col_max = count * np.array(col_max).reshape(1,type_shape[0])
    return  np.sum(col_max)

def evaluate_moveset1(data: DataLoader, attacker_name: str, selected_moves: List[str], type_combos: pd.DataFrame) -> Dict[str, Any]:
    print(f'start: {time.time()}')
    attacker_types = data.get_pokemon_types(attacker_name)
    target_type_map = data.get_all_target_types()
    print(time.time())    
    effectiveness_counts = Counter()
    type_hit_map = {
        'super_effective': set(),
        'normal_effective': set(),
        'not_very_effective': set(),
        'ineffective': set(),
        'power': set()
    }
    move_scores = {}
    move_n = ''
    for target_name, target_types in target_type_map.items():
        best_effectiveness = 0
        for move_name in selected_moves:
            move_type, power = data.get_move_data(move_name)
            if power < 0:
                continue
            multiplier = data.get_type_effectiveness(move_type, target_types)
            stab = 1.5 if move_type in attacker_types else 1.0
            effective_power = power * stab * multiplier
            if effective_power >= best_effectiveness:
                move_n = move_name
                best_effectiveness = effective_power
        try:
            move_scores[move_n] += best_effectiveness
        except:
            move_scores[move_n] = best_effectiveness   
        
        if best_effectiveness > 0:
            if multiplier == 0:
                effectiveness_counts['ineffective'] += 1
                type_hit_map['ineffective'].update(target_name)
            elif multiplier > 1:
                effectiveness_counts['super_effective'] += 1
                type_hit_map['super_effective'].update(target_name)
            elif multiplier == 1:
                effectiveness_counts['normal_effective'] += 1
                type_hit_map['normal_effective'].update(target_name)
            elif multiplier < 1:
                effectiveness_counts['not_very_effective'] += 1
                type_hit_map['not_very_effective'].update(target_name)
    print(f'end: {time.time()}') 
    return {
        'effectiveness_counts': effectiveness_counts,
        'type_hit_map': {k: sorted(v) for k, v in type_hit_map.items()},
        'scores': move_scores
    }

def recommend_moveset(data: DataLoader, attacker_name: str, current_moves: List[str], locked_moves: List[str], banned_moves: List[str], max_changes: int, generation: str, att: List[int]) -> List[str]:
    type_combos = data.calculate_type_effectiveness()
    data.att = att
    learnset = set(data.get_damaging_learnset(attacker_name, generation))
    substring = 'hidden-power'
    c = sum(1 for item in banned_moves if substring in item)
    if c>0:
        banned_moves.extend([
        'hidden-power-fig', 'hidden-power-fly', 'hidden-power-poi',
        'hidden-power-gro', 'hidden-power-roc', 'hidden-power-bug',
        'hidden-power-gho', 'hidden-power-ste', 'hidden-power-fir',
        'hidden-power-wat', 'hidden-power-gra', 'hidden-power-ele',
        'hidden-power-psy', 'hidden-power-ice', 'hidden-power-dra',
        'hidden-power-dar', 'hidden-power-fai'
    ])
    available_moves = list(learnset - set(locked_moves)-set(banned_moves))
    num_to_change = min(max_changes, 4 - len(locked_moves))
    best_score = 0
    best_moveset = current_moves

    for new_moves in combinations(available_moves, num_to_change):
        trial_moveset = locked_moves + list(new_moves)
        count = sum(1 for item in trial_moveset if substring in item)
        if count > 1:
            continue
        if len(trial_moveset) > 4:
            continue
        evaluation = evaluate_moveset(data, attacker_name, trial_moveset, type_combos, generation)
        #counts = evaluation['effectiveness_counts']
        #score = counts['super_effective'] * 2 + counts['normal_effective']
        counts = evaluation
        score = counts
        if score > best_score:
            best_score = score
            best_moveset = trial_moveset
    return list(best_moveset)
