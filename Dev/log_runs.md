#original - LLR_PARITY = 20.0, num_iter=20, max_k=6000, target_fer=1e-3,sweep_factor=1.62, max_sweep_iter=10,hash_bits=512;

#old1 - LLR_PARITY = 50.0,num_iter=20, max_k=8000, target_fer=1e-2,sweep_factor=1.12, max_sweep_iter=10,hash_bits=32; oct20 night, the ldpc skr crosses at 7.5% qber
#old0 - LLR_PARITY = 20.0,  num_iter=30, max_k=8000, target_fer=1e-2, sweep_factor=1.08, max_sweep_iter=10, hash_bits=32; oct21 night, the ldpc skr only crosses at 6.5% qber
#new2 - LLR_PARITY = 20.0, num_iter=60, max_k=8448, target_fer=0.2, sweep_factor=1.01, max_sweep_iter=30, hash_bits=32 ; argon advice, 10avgruns, performs too good, passes at 8% qber, performs better than cascade at high qber
#new2- LLR_PARITY = 20.0, num_iter=40, max_k=8000, target_fer=0.2, sweep_factor=1.01, max_sweep_iter=20, hash_bits=32; 10avgruns, still performs too good, passes at 8% qber, 
#new2- LLR_PARITY = 20.0, num_iter=40, max_k=7000, target_fer=0.2, sweep_factor=1.01, max_sweep_iter=20, hash_bits=32; 10avgruns, performs even better? passes at 8% qber,
#new2- LLR_PARITY = 20.0, num_iter=40, max_k=7000, target_fer=0.1, sweep_factor=1.12, max_sweep_iter=20, hash_bits=32; 10avgruns, similar 8%qber 
#new2 - LLR_PARITY = 20.0,  num_iter=30, max_k=7000, target_fer=0.1, sweep_factor=1.12, max_sweep_iter=10, hash_bits=32; 10avgruns, to see if performance degrades, it does now passes at 7.5% qber ; lets run for more avg to see it performs similarly or falls more
#new2 - LLR_PARITY = 20.0,  num_iter=30, max_k=7000, target_fer=0.1, sweep_factor=1.12, max_sweep_iter=10, hash_bits=32; 40avgruns, performs too well, passes at 8% qber
#new2 - LLR_PARITY = 20.0,  num_iter=20, max_k=7000, target_fer=0.1, sweep_factor=1.12, max_sweep_iter=10, hash_bits=32; 10avgruns, to see if performance degrades, yes it degrades passes at around 7%qber
#new2 - LLR_PARITY = 20.0,  num_iter=20, max_k=7000, target_fer=0.1, sweep_factor=1.12, max_sweep_iter=10, hash_bits=32; 50avgruns, to see if the performance stays stable and fluctuations smoothen, performs good passes around 7% qber, however at high qber it surpasses all 
#new2 - LLR_PARITY = 20.0,  num_iter=20, max_k=7000, target_fer=0.1, sweep_factor=1.12, max_sweep_iter=5, hash_bits=32; 10avgruns, to see if the performance degrades only in high qber, does reduce in high qber, the pass at 6.9% qber
#new2 - LLR_PARITY = 20.0,  num_iter=20, max_k=7000, target_fer=0.1, sweep_factor=1.12, max_sweep_iter=5, hash_bits=32; 50avgruns, to see if the performance stays stable, best performance so far passes at 8%qber
#new2 - LLR_PARITY = 20.0,  num_iter=20, max_k=6000, target_fer=0.1, sweep_factor=1.12, max_sweep_iter=5, hash_bits=32; 10avgruns, to see if it affects the high qber performance at all, outperforms cascade qber passes at 8.1-8.2%qber
#new2 - LLR_PARITY = 20.0,  num_iter=20, max_k=7000, target_fer=0.1, sweep_factor=1.22, max_sweep_iter=5, hash_bits=32; 10avgruns,to see if it affects the high qber performance at all,passes at around 7.5%qber
#new2 - LLR_PARITY = 20.0,  num_iter=20, max_k=7000, target_fer=1e-3, sweep_factor=1.22, max_sweep_iter=5, hash_bits=32; 10avgruns, very bad pass at 5.9%qber
#new2 - LLR_PARITY = 20.0,  num_iter=20, max_k=7000, target_fer=0.1, sweep_factor=1.22, max_sweep_iter=5, hash_bits=32; 100avgruns, oct22 night, best results so far,passes at 7.5%qber

#new1 - running to new plot of realized secret key fraction against qber, 20avg runs, seems odd, ldpc seems to be fluctuating a lot

#new1 - running to new plot of realized secret key fraction against qber, 10avg runs, with new mism>0 condn at the bottom of ldpc run function. hell no fluctuations increased, no positive skr

#new1 - running to new plot of realized secret key fraction against qber, 10avg runs, with new mism>0 condn at the bottom of ldpc run function and llr inversion at end. fail no ldpc values

#new1 - running to new plot of realized secret key fraction against qber,10000signals 10avg runs, with new partial chunk discard technique from argon, ldpc values are there but the values on the extreme, extemely high leakage, no positive skr etc etc

#new1 - running to new plot of realized secret key fraction against qber, 10avg runs, with new partial chunk discard technique from argon, ldpc values are there but the values on the extreme, moderate leakage, no positive skr etc etc

#new1 - LLR_PARITY = 20.0, num_iter=60, max_k=8448, target_fer=0.2, sweep_factor=1.01, max_sweep_iter=30, hash_bits=32 ;100avg runs oct24 night, with new partial chunk discard technique from argon, 

#new1 - LLR_PARITY = 20.0,  num_iter=20, max_k=7000, target_fer=0.1, sweep_factor=1.22, max_sweep_iter=5, hash_bits=32; 100avgruns,new plot of realized secret key fraction against qber, other plots are obv ok, but the realized secure key fraction ldpc doesnt go to zero

#new3 - new argon ldpc code, new bb84 edits, 10avg runs, very high leakage in ldpc, no positive skr

#new3 - new argon ldpc modified code, 10avg runs, very high leakage in ldpc, no positive skr
#new3 - new argon ldpc modified code, 8448 block size, 10avg runs,very high leakage in ldpc, no positive skr
#new3 - new argon ldpc modified code 2, 10avg runs, better performance, actually surpases cascade, suprising !! will change the parameters
#new3 - new argon ldpc modified code 2, 50000signals , 10avg runs, LLR_PARITY = 20.0,  num_iter=20, max_k=3840, target_fer=0.1, sweep_factor=1.08   , max_sweep_iter=5, hash_bits=32; good but still looks like overperformance of ldpc
#new3 - new argon ldpc modified code 2, 100000signals , 100avg runs, LLR_PARITY = 20.0,  num_iter=15, max_k=3840, target_fer=0.1, sweep_factor=1.08   , max_sweep_iter=5, hash_bits=128;seems good, cascade is below zero in reconcilliation efficiency, ldpc is not going to zero in realized skr fraction

#new3 - new argon ldpc modified code 3 no cascade fallback, 100000signals , 10avg runs, LLR_PARITY = 20.0,  num_iter=15, max_k=3840, target_fer=0.1, sweep_factor=1.08   , max_sweep_iter=5, hash_bits=128;
