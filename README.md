# Pokemon-Move-Optimizer
A Python application that helps users select the most effective offensive moveset for a given Pokémon. This tool uses real game data (provided via CSV files) and type matchups to analyze and recommend moves that maximize type coverage and damage potential.

Features:
Choose generetion you want to optimize moveset for(this changes learnsets and move types pre gen 4, it does not change move power, it's updated to gen 9)

Select any Pokémon from a dataset(gen 1-9, no alternative formes)

Input attack/special attack stat(program does not provide IV/Ev calculations, you have to calculate it somewhere else)

Choose up to 4 damaging moves from their learnset

Evaluate selected moveset based on:

  Effective damage dealt to all pokemons(gen 1-9)
  
  STAB
  
  Type coverage for each category
  
  It does NOT consider abilities, weather, accuracy, etc.
  
This optimizer only chooses damaging moves, does not take into consideration extra effects(also negative like recharge turn)

Lock specific moves and limit the number of suggested changes

Ban specific moves and limit the number of suggested changes(especially banning hidden-power will ban every hidden power type)

Limit calculations by specific moves' minimum and maximum power

Simple GUI

Data Requirements

The app expects the following CSV files:

  pokemon.csv – Pokémon list with types
  
  Columns: name, type1, type2
  
  
  moves.csv – Move information
  
  Columns: name, type, power, is_damaging (1 if damaging, 0 otherwise)
  

  learnsets.csv – Which Pokémon can learn which moves
  
  Columns: pokemon, move
  

  type_chart.csv – Type effectiveness matrix
  
  Rows and columns as types(rows attacker, columns defender)
  
  Each cell represents the effectiveness multiplier (e.g. 2.0, 0.5, 1.0)
  

  You can provide your own files with data if you wish to apply changes(for example limit targets, revert move changes from gen 9, etc.)
  

  Program is poorly optimized right now, if you wish to let program select 4 moves without any restrictions it can take even few minutes to calculate, locking at lest one move and banning hidden power reduces this time to few seconds.
  

  How to run:
  
  Download and extract .zip file.

  Run moveset_optimizer.exe

  Make sure all csv files are in the same folder that .exe
  

  Terminal will print some data for debugging purposes
  

  Feel free to post your own, better version of this program.
  
