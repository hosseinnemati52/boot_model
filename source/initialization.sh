#!/bin/bash

# Read the value of n (line 2) and m (line 3) from init_steps_data.txt
n=$(sed -n '2p' init_steps_data.txt)  # Extract the second line
m=$(sed -n '3p' init_steps_data.txt)  # Extract the third line

# Loop from 1 to n
for ((i=1; i<=n; i++))
do
    # Step 1: Run "mechanically_relax.exe"
    ./mechanically_relax.exe
    
    # Step 2: Run "equilibration_sim.exe". Inside this function, the final distributions are written.
    ./equilibration_sim.exe

    # Step 3: Run the Python script "cond_check.py". Inside this, the file "condition.txt" is written.
    python3 eq_cond_check.py    
    
    # Step 4: Check the condition
    condition=$(cat eq_condition.txt)
    if [[ "$condition" -eq 1 && "$i" -gt "$m" ]]; then
        echo "Equilibration condition met (condition=1 and i>$m)!"
        
        # Run the script "make_phi_F.py"
        python3 make_phi_F.py

        # Run the file "mechanically_relax.exe"
        ./mechanically_relax.exe

        # Exit the loop
        break
    else
        # Run the script "make_phi_F.py"
        python3 make_phi_F.py
    fi
done

