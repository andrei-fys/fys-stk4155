#!/bin/bash

progname="b.py"
echo "degree, noise, Mean, R2, MSE"
for (( i=1; i < 5; i++))
    do
#      echo "With noice"
#      echo "Poly is $i"
	  ./$progname $i True
#      echo "Without noice"
#      echo "Poly is $i"
	  ./$progname $i False
done
