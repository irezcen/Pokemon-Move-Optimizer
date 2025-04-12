import tkinter as tk
from tkinter import ttk, messagebox
from data_logic import DataLoader, evaluate_moveset, recommend_moveset
import numpy as np

class MovesetApp:
    def __init__(self, root, data: DataLoader):
        self.data = data
        self.root = root
        self.root.title("Pokémon Moveset Optimizer")

        self.pokemon_var = tk.StringVar()
        self.move_vars = tk.StringVar()
        self.locked_vars = []
        self.banned_vars = []
        self.max_changes_var = tk.IntVar(value=1)
        
        self.min_power_var = tk.IntVar(value=0)
        self.max_power_var = tk.IntVar(value=999)  # Arbitrary high value for max power
        self.generation_var = tk.StringVar()
        self.attack_var = tk.IntVar(value=1)
        self.special_var = tk.IntVar(value=1)
        
        self.debounce_delay = 300  # 500 ms delay (0.5 seconds)
        self.debounce_timer = None  # To store the scheduled function call

        self.create_widgets()

    def create_widgets(self):
        ttk.Label(self.root, text="Select Pokémon:").grid(row=0, column=0, sticky="w")
        ttk.Label(self.root, text="Attack/Special Attack stat").grid(row=1, column=0, sticky="w")
        
        ttk.Entry(self.root, textvariable=self.attack_var, width=3).grid(row=1, column=0, sticky="e")
        ttk.Entry(self.root, textvariable=self.special_var, width=3).grid(row=1, column=1, sticky="w")
        
        ttk.Label(self.root, text="Gen:").grid(row=2, column=0, sticky="w")
        
        self.generation_combo = ttk.Combobox(self.root, textvariable=self.generation_var, values=[str(i) for i in range(1, 10)], state="normal")
        self.generation_combo.grid(row=2, column=1, sticky="n")
        
        self.pokemon_combo = ttk.Combobox(self.root, textvariable=self.pokemon_var, values=self.data.get_pokemon_list(), state="normal")
        self.pokemon_combo.grid(row=0, column=1)
        
        
        
        self.pokemon_combo.bind("<<ComboboxSelected>>", self.update_moves)
        self.generation_combo.bind("<<ComboboxSelected>>", self.update_pokemon_list)
        
        self.pokemon_combo.bind("<KeyRelease>", lambda event: self.debounce_update_combobox(self.pokemon_combo, self.data.get_pokemon_list(), event))
        self.generation_combo.bind("<KeyRelease>", lambda event: self.debounce_update_combobox(self.generation_combo, [str(i) for i in range(1, 10)], event))
        
        

        self.moves_frame = ttk.LabelFrame(self.root, text="Select Moves")
        self.moves_frame.grid(row=3, column=0, columnspan=2, pady=10)

        ttk.Label(self.root, text="Max Moves to Change:").grid(row=6, column=0, sticky="w")
        ttk.Entry(self.root, textvariable=self.max_changes_var, width=5).grid(row=6, column=1, sticky="w")
        
                # Add Min Power and Max Power fields
        ttk.Label(self.root, text="Min Power:").grid(row=7, column=0, sticky="w")
        ttk.Entry(self.root, textvariable=self.min_power_var, width=5).grid(row=7, column=1, sticky="w")

        ttk.Label(self.root, text="Max Power:").grid(row=8, column=0, sticky="w")
        ttk.Entry(self.root, textvariable=self.max_power_var, width=5).grid(row=8, column=1, sticky="w")

        ttk.Button(self.root, text="Recommend Moveset", command=self.recommend).grid(row=4, column=0, columnspan=2, pady=5)

        self.results_text = tk.Text(self.root, height=30, width=60)
        self.results_text.grid(row=9, column=0, columnspan=2)
        
    def update_pokemon_list(self, event=None):
        selected_generation = self.generation_var.get()
        filtered_pokemon = self.data.get_pokemon_by_generation(selected_generation)
        
        self.pokemon_combo['values'] = filtered_pokemon
        self.pokemon_combo.set('')  # Clear the selection

        # Clear the moves section when the generation changes
        for widget in self.moves_frame.winfo_children():
            widget.destroy()
            
    def debounce_update_combobox(self, combobox, options, event=None):
        if self.debounce_timer:
            self.root.after_cancel(self.debounce_timer)  # Cancel the previous scheduled call if a new key is pressed

        self.debounce_timer = self.root.after(self.debounce_delay, lambda: self.update_combobox_autocomplete(combobox, options, event))

    def update_combobox_autocomplete(self, combobox, options, event=None):
        typed_value = combobox.get().lower()  # Get the current text in the combobox
        
        if typed_value == "":
            # If the input is empty, show all options
            combobox['values'] = options
            combobox.set('')  # Optionally clear the selection
        else:
            # Filter the options based on the user's input
            filtered_options = [option for option in options if typed_value in option.lower()]
            combobox['values'] = filtered_options
            if filtered_options:
                combobox.set(filtered_options[0])  # Optionally auto-select the first match

            
    def update_moves(self, event=None):
        for widget in self.moves_frame.winfo_children():
            widget.destroy()

        selected = self.pokemon_var.get()
        selected_generation = self.generation_var.get()
        moves = self.data.get_pokemon_learnset(selected, selected_generation)
        moves = [str(i) for i in moves]
        moves.sort()
        self.move_vars = []
        self.locked_vars = []
        self.banned_vars = []

        for i in range(0, 10):
            move_var = tk.StringVar(value=moves[i])
            lock_var = tk.BooleanVar(value=False)
            ban_var = tk.BooleanVar(value=False)  # New variable for banning moves
            self.move_vars.append(move_var)
            self.locked_vars.append(lock_var)
            self.banned_vars.append(ban_var)  # Add to banned list

            ttk.Combobox(self.moves_frame, textvariable=move_var, values=moves, state='normal').grid(row=i, column=0)
            if i < 4:
                ttk.Checkbutton(self.moves_frame, text="Lock", variable=lock_var).grid(row=i, column=2)
            ttk.Checkbutton(self.moves_frame, text="Ban", variable=ban_var).grid(row=i, column=1)

    def recommend(self):
        type_combos = data.calculate_type_effectiveness()
        selected = self.pokemon_var.get()
        moves = [var.get() for var in self.move_vars if var.get()]
        locked = [var.get() for i, var in enumerate(self.move_vars) if self.locked_vars[i].get()]
        banned = [var.get() for i, var in enumerate(self.move_vars) if self.banned_vars[i].get()]
        max_changes = self.max_changes_var.get()
        generation = self.generation_var.get()
        
                # Get the min and max power values
        min_power = self.min_power_var.get()
        max_power = self.max_power_var.get()

        if len(moves) == 0:
            messagebox.showerror("Error", "Select at least one move.")
            return
        
        self.data.min_max_power = [min_power, max_power]

        att = [self.attack_var.get(), self.special_var.get()]
        result_moveset = recommend_moveset(self.data, selected, moves, locked, banned, max_changes, generation, att)
        #evaluation = evaluate_moveset(self.data, selected, result_moveset, type_combos)

        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Recommended Moveset:\n{result_moveset}\n\n")
        #self.results_text.insert(tk.END, f"Effectiveness Counts:\n{evaluation['effectiveness_counts']}\n\n")

        # Show the effectiveness of each move against Pokémon types
        self.results_text.insert(tk.END, f"Type Coverage:\n")
        
        # Use `calculate_type_effectiveness` to get detailed type coverage
        type_effectiveness_df = self.data.calculate_type_effectiveness()

        types = type_effectiveness_df['sorted1'].unique()
        
        multiplier = np.zeros(18)
        for move in result_moveset:
            self.results_text.insert(tk.END, f"{move}:\n")            
            move_type, power, dtype = data.get_move_data(move)
            
            for i in range(0, len(types)):
                type1 = [types[i]]
                
                a = self.data.get_type_effectiveness(move_type, type1)
                if a > multiplier[i]:
                    multiplier[i] = a
        for i in range(0, len(types)):
            if multiplier[i] == 2:  # Super effective
                effectiveness_str = "Super Effective"
            elif multiplier[i] == 1:  # Normal
                effectiveness_str = "Effective"
            elif multiplier[i] == 0.5:  # Weak
                effectiveness_str = "Not very effective"
            elif multiplier[i] == 0:  # No effect
                effectiveness_str = "No Effect"
            self.results_text.insert(tk.END, f"  {types[i]}: {effectiveness_str}\n")
            self.results_text.insert(tk.END, "\n")
                
            
            

if __name__ == "__main__":
    data = DataLoader("pokemon.csv", "moves.csv", "learnsets.csv", "type_chart.csv")
    root = tk.Tk()
    app = MovesetApp(root, data)
    root.mainloop()
