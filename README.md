# RNN-Haiku-Generator
Practice using a GRU RNN to synthesize Haikus using various levels of Temperature.
Uses a 2-layer bidirectional GRU to generate Haikus letter by letter. Bidirectionality is used to attempt to create continuity from beginning to end. 
Haikus generally fit a 5/7/5 structure, however there is no coherent sentiment within the words. Training data was downloaded from https://www.kaggle.com/datasets/bfbarry/haiku-dataset which collected and cleaned 11,265 Haikus from r/Haiku. The model takes in a starting word or characters then attempts to produce a Haiku based on the initial input using varius levels of Temperature. 
The resulting program output looks as follows:

Enter starting word for Haiku: ready 

Generated Haiku (Temperature = 0.1)
['ready and spines / a sunding a still the day /  a sunding ']

Generated Haiku (Temperature = 0.2)
['ready and stell / a sunding of the sould / i will the say ']

Generated Haiku (Temperature = 0.3)
['ready see think / light and and light the snow /  by with the beat ']

Generated Haiku (Temperature = 0.4)
['ready in the day / in a coud i wont a hang / a lay of sun ']
