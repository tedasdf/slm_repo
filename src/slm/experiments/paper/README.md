###

Train of thought:

first set for 12 runs 

1. with pre set depth list and width list and head dim, fixed flop and dataset set 

2. there are architecture 2ith overlapping ratios and compute budget 

dont do that 



process:

1. do a frontier illustration with only 3 compute budgets, spaced far apart and chosen from the higher end of my range.
    a. Chinchilla's core result is that under a fixed comptue budget, the compute -optimal model size and training tokens should scale roughly equally, so this part is really about showing that trend, not re-fitting a new law from scratch. 
    b. Kaplan also reported that width/depth details have relativelyt small effect compared with overall scale across a borad range, 
conclusion:
    using a compact architecture family for this illustration is a reaonsable compromize.

2. Second, do a **fixed dataset sweep** with 9 molde size. 
